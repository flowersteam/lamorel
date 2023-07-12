import torch
from math import ceil

import logging
lamorel_logger = logging.getLogger('lamorel_logger')

from .utils import load_hf_model_and_tokenizer
from .base_llm import BaseLLM

# Accelerate
from accelerate import Accelerator, infer_auto_device_map, init_empty_weights
accelerator = Accelerator()

from contextlib import nullcontext


class HF_LLM(BaseLLM):
    """
    This class is a wrapper around the language model, and handles distributing
    and/or compiling the model.
    For each task the default is text in, text out.
    """

    def __init__(self, args, devices, use_cpu):
        super().__init__(args, devices, use_cpu)
        print("Parallelising HF LLM on {} devices".format(len(self.devices)))
        # Load model and tokenizer
        self._LLM_tokenizer, _model_constructor, num_layers = load_hf_model_and_tokenizer(
            args.model_type, args.model_path, args.pretrained)

        if use_cpu:
            # Current version of the lib does not support parallelization with cpu
            self._LLM_model = _model_constructor().to('cpu')
        else:
            # Set model parallelism
            with init_empty_weights():
                self._LLM_model = _model_constructor()
                self._LLM_model.tie_weights()
                device_map = infer_auto_device_map(
                    model=self._LLM_model,
                    max_memory={
                        _device: torch.cuda.mem_get_info(f'cuda:{_device}')[0]
                        for _device in devices
                    }
                )

            self._LLM_model = _model_constructor(device_map=device_map)

        # Set minibatch generation
        self.__input_encoder = None
        self._scoring_minibatch_size = args.minibatch_size
        if args.model_type == "causal":
            self.__minibatch_generator = self.__build_decoder_minibatch
            self.model_type = "causal"
        elif args.model_type == "seq2seq":
            self.__minibatch_generator = self.__build_encoder_decoder_minibatch
            if args.pre_encode_inputs:
                self.__input_encoder = lambda input: self.__get_input_embedding_from_encoder(input)
            self.model_type = "seq2seq"
        else:
            raise NotImplementedError()

        if self._LLM_tokenizer.pad_token is not None:
            self.pad_token = 0  # self._LLM_tokenizer(self._LLM_tokenizer.pad_token)
        else:
            self.pad_token = 0  # self._LLM_tokenizer(" ")
        self.__synchronize_gpus_after_scoring = args.parallelism.synchronize_gpus_after_scoring
        self.__empty_cuda_cache_after_scoring = args.parallelism.empty_cuda_cache_after_scoring

    def get_model_config(self):
        return self._LLM_model.config

    def register_module_functions(self, module_functions):
        self._module_functions = torch.nn.ModuleDict(module_functions)

    def __pad_sequence(self, sequence, size):
        sequence_size = len(sequence["input_ids"])
        ids = sequence["input_ids"] + [
            self.pad_token
            for _ in range(size - sequence_size)]
        mask = sequence["attention_mask"] + [0 for _ in range(size - sequence_size)]
        sequence["input_ids"] = ids
        sequence["attention_mask"] = mask
        return sequence

    def get_trainable_module(self):
        return self._LLM_model

    def __concat_sequences(self, sequence_a, sequence_b, sequence_size=None):
        if sequence_size is None: sequence_size = len(sequence_a['input_ids']) + len(sequence_b['input_ids'])
        return self.__pad_sequence({
            "input_ids": sequence_a['input_ids'] + sequence_b['input_ids'],
            "attention_mask": sequence_a['attention_mask'] + sequence_b['attention_mask']
        }, sequence_size)

    def __build_decoder_minibatch(self, outputs, inputs, inputs_representation=None):
        '''
            Concat state and output
        '''
        batch = {
            "input_ids": [],
            "attention_mask": []
        }
        output_max_size = max([len(_i['input_ids']) + len(_o['input_ids']) for _i, _o in zip(inputs, outputs)])
        for input, output in zip(inputs, outputs):
            concatenation = self.__concat_sequences(input, output, output_max_size)
            batch["input_ids"].append(concatenation["input_ids"])
            batch["attention_mask"].append(concatenation["attention_mask"])

        return batch

    def __build_encoder_decoder_minibatch(self, outputs, inputs=None, inputs_representation=None):
        '''
            Set state as encoder's input and action as decoder's input
        '''
        assert inputs is not None or inputs_representation is not None
        batch = {
            "decoder_input_ids": [],
            "decoder_attention_mask": []
        }

        output_max_size = max([len(o['input_ids']) for o in outputs])
        for idx, output in enumerate(outputs):
            _output = self.__concat_sequences(
                {"input_ids": [self.pad_token], "attention_mask": [1]},
                output,
                output_max_size + 1
            )
            batch["decoder_input_ids"].append(_output["input_ids"])
            batch["decoder_attention_mask"].append(_output["attention_mask"])
            if idx == 0:
                batch["attention_mask"] = []
            batch["attention_mask"].append(inputs[idx]["attention_mask"])

            if inputs_representation is None:
                if idx == 0:
                    batch["input_ids"] = []
                batch["input_ids"].append(inputs[idx]["input_ids"])

        if inputs_representation is not None:
            batch["encoder_outputs"] = tuple([inputs_representation])

        return batch

    def __get_input_embedding_from_encoder(self, input):
        input_batch = {}
        for key, value in input.items():
            input_batch[key] = torch.tensor(value, device=self.device)
        return self._LLM_model.encoder(**input_batch, return_dict=False)[0].to(self.device)

    def generate(self, contexts, **kwargs):
        generations = []
        for text_input in contexts:
            encoded_input = self._LLM_tokenizer.encode(text_input, return_tensors='pt').to(self.device)
            results = self._LLM_model.generate(
                encoded_input,
                return_dict_in_generate=True,
                output_scores=True,
                **kwargs
            )
            if self.model_type == "causal":  # hence input should be removed from result
                generated_sequences = results.sequences[:, encoded_input.shape[-1]:]
            else:
                generated_sequences = results.sequences[:, 1:]
            _generated_texts = self._LLM_tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)
            probabilities = torch.stack(results.scores, dim=1).softmax(-1)
            texts_probabilities = torch.gather(probabilities, 2, generated_sequences[:, :, None]).squeeze(-1)
            _scores = texts_probabilities.prod(-1)

            generations.append([
                {
                    "text": _text,
                    "score": _score.detach().cpu().numpy()
                }
                for _text, _score in zip(_generated_texts, _scores)
            ])
        return generations

    def forward(self, module_function_keys, contexts, candidates=None, require_grad=False, minibatch_size=None,
                **kwargs):
        _forward_results = [[] for _ in range(len(contexts))]
        if candidates is None:
            candidates = [[""] for _ in range(len(contexts))]

        _ids_tables = {}
        with torch.no_grad() if not require_grad else nullcontext():
            batch_inputs, batch_input_representations, batch_outputs = [], None, []
            tokenized_contexts = [self._LLM_tokenizer(context) for context in contexts]
            contexts_max_size = max([len(i['input_ids']) for i in tokenized_contexts])

            # 1) Concat all samples to prepare batches
            for _w, _candidates in enumerate(candidates):
                _ids_tables[_w] = [i for i in range(len(batch_inputs), len(batch_inputs) + len(_candidates))]
                if len(_candidates) == 0:
                    break

                lamorel_logger.debug(f"Tokenizing the {_w}-th batch")
                outputs = [self._LLM_tokenizer(output) for output in _candidates]
                if self.model_type == "seq2seq":
                    padded_input = self.__pad_sequence(tokenized_contexts[_w], contexts_max_size)
                else:
                    padded_input = tokenized_contexts[_w]

                batch_inputs.extend([padded_input for _ in range(len(outputs))])
                batch_outputs.extend(outputs)
                _w += 1

            # 2) If needed, first encode inputs
            _minibatch_size = minibatch_size if minibatch_size is not None else self._scoring_minibatch_size
            lamorel_logger.debug(
                f"Preparing to process {len(batch_inputs)} examples with a batch size of {_minibatch_size}...")
            if self.__input_encoder is not None:
                batch_input_representations = []
                for step in range(len(contexts) // _minibatch_size + 1):
                    _input = {}
                    for _i in range(step*_minibatch_size, min((step+1)*_minibatch_size, len(contexts))):
                        if len(_ids_tables[_i]) == 0: break
                        _current_context = batch_inputs[_ids_tables[_i][0]]
                        for k,v in _current_context.items():
                            if k not in _input: _input[k] = []
                            _input[k].append(v)
                    if len(_input.keys()) > 0:
                        result = self.__input_encoder(_input)
                        _current_candidates = candidates[step*_minibatch_size: (step+1)*_minibatch_size]
                        batch_input_representations.append(
                            torch.repeat_interleave(result,
                                                    torch.tensor([len(_c) for _c in _current_candidates], device=self.device),
                                                    dim=0)
                        )
                if len(batch_input_representations) > 0:
                    batch_input_representations = torch.cat(batch_input_representations)
                else:
                    batch_input_representations = None

            # 3) Use decoder + custom modules
            batch_results = {k: [] for k in module_function_keys}
            for step in range(len(batch_inputs) // _minibatch_size + 1):
                lamorel_logger.debug(f"Processing minibatch n°{step}...")
                step_idx = step * _minibatch_size
                current_minibatch_size = min(_minibatch_size, len(batch_inputs) - step_idx)
                if current_minibatch_size <= 0: break
                _inputs_representation = batch_input_representations[step_idx:step_idx + current_minibatch_size] \
                    if batch_input_representations is not None else None
                minibatch_inputs = self.__minibatch_generator(
                    batch_outputs[step_idx:step_idx + current_minibatch_size],
                    inputs=batch_inputs[step_idx:step_idx + current_minibatch_size],
                    inputs_representation=_inputs_representation
                )
                if not module_function_keys == ['__score']:
                    minibatch_inputs["output_hidden_states"] = True

                # Transform it to tensors on the right device
                lamorel_logger.debug(f"Putting it on device {self.device}")
                minibatch = {}
                for key, value in minibatch_inputs.items():
                    if key == "encoder_outputs":
                        minibatch[key] = value
                    else:
                        minibatch[key] = torch.tensor(value, device=self.device)

                lamorel_logger.debug(f"Calling forward on process {accelerator.process_index}")
                _outputs = self._LLM_model(**minibatch)  # Get scores before softmax
                lamorel_logger.debug(f"Forward succeeded on process {accelerator.process_index}")

                for _key in module_function_keys:
                    lamorel_logger.debug(f"Computing {_key} function")
                    _fn = self._module_functions[_key]
                    if _key == "__score":
                        results = _fn(_outputs, minibatch, batch_inputs[step_idx:step_idx + current_minibatch_size])
                    else:
                        results = _fn(_outputs,
                                      minibatch=minibatch,
                                      tokenized_contexts=batch_inputs[step_idx:step_idx + current_minibatch_size],
                                      **kwargs)
                    batch_results[_key].append(results)

            for k, _ in batch_results.items():
                if len(batch_results[k]) > 0:
                    batch_results[k] = torch.cat(batch_results[k])

            for idx in range(len(contexts)):
                _forward_results[idx] = {}
                for k, v in batch_results.items():
                    indices = _ids_tables[idx]
                    if len(indices) > 0:
                        _forward_results[idx][k] = v[indices]
                    else:
                        _forward_results[idx][k] = torch.tensor([])

        if self.__synchronize_gpus_after_scoring:
            lamorel_logger.debug(f"Synchronizing GPUs on process {accelerator.process_index}")
            for device in self.devices:
                torch.cuda.synchronize(device)
        if self.__empty_cuda_cache_after_scoring:
            lamorel_logger.debug(f"Emptying CUDA cache on process {accelerator.process_index}")
            torch.cuda.empty_cache()

        lamorel_logger.debug(f"Scoring finished on process {accelerator.process_index}")
        return _forward_results
