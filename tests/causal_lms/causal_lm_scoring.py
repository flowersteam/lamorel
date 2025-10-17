import os
visible_device = str(max(0, int(os.environ.get("RANK")) - 1))
print(f"Setting visible devices to be: {visible_device}")
os.environ['CUDA_VISIBLE_DEVICES'] = visible_device
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "0"

try:
    from unsloth import FastLanguageModel
    print(f"Successfully imported unsloth!")
except Exception as err:
    print("Failed to import unsloth.")

import unittest
import torch
import hydra
import numpy

from torch.nn.functional import log_softmax

from lamorel import Caller, BaseModuleFunction, BaseModelInitializer


class LogScoringModuleFn(BaseModuleFunction):
    def __init__(self):
        super().__init__()
        self._pad_token = 0

    def initialize(self):
        pass

    def forward(self, forward_outputs, minibatch, tokenized_contexts, **kwargs):
        if self.llm_config.model_type == "causal":
            if self.llm_config.pre_encode_inputs:
                logits = forward_outputs["logits"][:, :-1, :]
                output_tokens = minibatch["input_ids"][:, 1:]
            else:  # hence inputs should be removed from result
                end_of_context_positions = [len(_context["input_ids"]) for _context in tokenized_contexts]
                if len(set(end_of_context_positions)) == 1:  # inputs all have the same size (probably due to `pad_contexts=True`)
                    logits = forward_outputs["logits"][:, end_of_context_positions[0]:-1, :]
                    output_tokens = minibatch["input_ids"][:, end_of_context_positions[0]+1:]
                else:
                    raw_logits, raw_output_tokens = [], []
                    max_len = 0
                    for i in range(len(tokenized_contexts)):
                        raw_logits.append(forward_outputs["logits"][i, end_of_context_positions[i]:-1, :])
                        raw_output_tokens.append(minibatch["input_ids"][i, end_of_context_positions[i]+1:])
                        if len(raw_output_tokens[-1]) > max_len:
                            max_len = len(raw_output_tokens[-1])

                    logits = torch.stack([
                        torch.nn.functional.pad(torch.tensor(_logits), (0, 0, 0, max_len - len(_logits)), value=0)
                        for _logits in raw_logits
                    ])
                    output_tokens = torch.stack([
                        torch.nn.functional.pad(torch.tensor(_tokens), (0, max_len - len(_tokens)), value=self._pad_token)
                        for _tokens in raw_output_tokens
                    ])
        else:
            logits = forward_outputs["logits"][:, :-1, :]  # skip </s> token appended by tokenizer
            output_tokens = minibatch["decoder_input_ids"][:, 1:]  # skip pad token

        logits = log_softmax(logits, dim=-1)
        tokens_logprobs = \
            torch.gather(logits, 2, output_tokens[:, :, None]).squeeze(-1).to(torch.float32)  # filter with sequence tokens

        # Compute mask to assign probability 1 to padding tokens
        mask = torch.ones(tokens_logprobs.shape, dtype=torch.bool, device=self.device)
        for i, _output in enumerate(output_tokens):
            for j, _token in enumerate(_output):
                if _token != self._pad_token:
                    mask[i, j] = False
        masked_token_probs = tokens_logprobs.masked_fill(mask, 0.0)  # apply mask
        minibatch_probs = masked_token_probs.sum(-1)  # compute final sequences' probability

        return minibatch_probs.cpu()


class UnslothInferenceInitializer(BaseModelInitializer):
    def __init__(self, use_unsloth):
        super().__init__()
        self._use_unsloth = use_unsloth

    def initialize_model(self, model):
        if self._use_unsloth:
            print("Initializing unsloth model.")
            FastLanguageModel.for_inference(model._modules['_LLM_model'])

        return model

lm_server = None
generated = None
PROMPTS = ["The capital of France is", "How are you"]
pad_contexts = True

class CausalLMScoring(unittest.TestCase):
    def test_vanilla_scoring(self):
        global lm_server, generated, pad_contexts
        scores = []
        for idx, _prompt in enumerate(PROMPTS):
            _scores = lm_server.custom_module_fns(['score'],
                                                contexts=[_prompt],
                                                candidates=[
                                                    [_text['text'] for _text in generated[idx]]
                                                ],
                                                llm_to_call="main_llm", pad_contexts=pad_contexts)

            scores.extend([_s['score'].numpy() for _s in _scores])

        scores = numpy.array(scores)
        generated_scores = numpy.array([
            [_text['text_logprob'] for _text in _result] for _result in generated
        ])
        try:
            numpy.testing.assert_array_almost_equal(scores, generated_scores, 1)
        except AssertionError as e:
            self.fail()

    def test_batched_scoring(self):
        global lm_server, generated, pad_contexts
        _scores = lm_server.custom_module_fns(['score'],
                                            contexts=PROMPTS,
                                            candidates=[
                                                [_text['text'] for _text in _result]
                                                for _result in generated
                                            ], llm_to_call="main_llm", pad_contexts=pad_contexts)

        scores = numpy.array([_s['score'].numpy() for _s in _scores])
        generated_scores = numpy.array([
            [_text['text_logprob'] for _text in _result] for _result in generated
        ])
        try:
            numpy.testing.assert_array_almost_equal(scores, generated_scores, 1)
        except AssertionError as e:
            self.fail()

    # def test_minibatch_batched_scoring(self):
    #     global lm_server
    #     generated = lm_server.generate(contexts=self.PROMPTS, num_return_sequences=10, return_logprobs=True,
    #                                    do_sample=True, top_k=None, max_new_tokens=5, temperature=1, top_p=1,
    #                                    llm_to_call="main_llm")
    #
    #     _scores = lm_server.custom_module_fns(['score'],
    #                                         contexts=self.PROMPTS,
    #                                         candidates=[
    #                                             [_text['text'] for _text in _result]
    #                                             for _result in generated
    #                                         ], minibatch_size=5, llm_to_call="main_llm")
    #     scores_v1 = numpy.array([_s['score'].numpy() for _s in _scores])
    #     _scores = lm_server.custom_module_fns(['score'],
    #                                           contexts=self.PROMPTS,
    #                                           candidates=[
    #                                               [_text['text'] for _text in _result]
    #                                               for _result in generated
    #                                           ], minibatch_size=len(self.PROMPTS), llm_to_call="main_llm")
    #     scores_v2 = numpy.array([_s['score'].numpy() for _s in _scores])
    #     _scores = lm_server.custom_module_fns(['score'],
    #                                           contexts=self.PROMPTS,
    #                                           candidates=[
    #                                               [_text['text'] for _text in _result]
    #                                               for _result in generated
    #                                           ], minibatch_size=len(self.PROMPTS)*10, llm_to_call="main_llm")
    #     scores_v3 = numpy.array([_s['score'].numpy() for _s in _scores])
    #
    #     # generated_scores = numpy.array([
    #     #     [_text['text_logprob'] for _text in _result] for _result in generated
    #     # ])
    #     try:
    #         # numpy.testing.assert_array_almost_equal(scores_v1, generated_scores, 1)
    #         numpy.testing.assert_array_almost_equal(scores_v1, scores_v2, 1)
    #         numpy.testing.assert_array_almost_equal(scores_v1, scores_v3, 1)
    #     except AssertionError as e:
    #         self.fail()

@hydra.main(config_path='config', config_name='config')
def main(config_args):
    global lm_server, generated, pad_contexts
    pad_contexts = config_args.rl_script_args.pad_contexts
    lm_server = Caller(config_args.lamorel_args,
                       custom_model_initializer={
                           "main_llm": UnslothInferenceInitializer(config_args.lamorel_args.llm_configs.main_llm.handler == "unsloth")
                       },
                       custom_module_functions={
                           "main_llm": {
                               'score': LogScoringModuleFn()
                           }
                       })
    generated = lm_server.generate(contexts=PROMPTS, num_return_sequences=3, return_logprobs=True,
                                   do_sample=True, top_k=None, max_new_tokens=2, temperature=1, top_p=1.0,
                                   llm_to_call="main_llm")

    causal_lm_scoring_suite = unittest.TestLoader() \
        .loadTestsFromTestCase(CausalLMScoring)
    runner = unittest.TextTestRunner()
    runner.run(causal_lm_scoring_suite)
    lm_server.close()


if __name__ == '__main__':
    main()

