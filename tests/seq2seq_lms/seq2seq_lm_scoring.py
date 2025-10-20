import os
visible_device = str(max(0, int(os.environ.get("RANK")) - 1))
print(f"Setting visible devices to be: {visible_device}")
os.environ['CUDA_VISIBLE_DEVICES'] = visible_device
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "0"

import unittest
import torch
from torch.nn.functional import log_softmax
import hydra
import numpy

from lamorel import Caller, BaseModuleFunction

class LogScoringModuleFn(BaseModuleFunction):
    def __init__(self):
        super().__init__()

    def initialize(self):
        pass

    def forward(self, forward_outputs, minibatch, tokenized_contexts, **kwargs):
        if self.llm_config.model_type == "causal":
            if self.llm_config.pre_encode_inputs:
                logits = forward_outputs["logits"][:, :-1, :]
                output_tokens = minibatch["input_ids"][:, 1:]
            else:  # hence input should be removed from result (and may not be all of same size)
                end_of_context_positions = [len(_context["input_ids"]) for _context in tokenized_contexts]
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
                    torch.nn.functional.pad(torch.tensor(_tokens), (0, max_len - len(_tokens)), value=self.llm_config.pad_token)
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
                if _token != self.llm_config.pad_token:
                    mask[i, j] = False
        masked_token_probs = tokens_logprobs.masked_fill(mask, 0.0)  # apply mask
        minibatch_probs = masked_token_probs.sum(-1)  # compute final sequences' probability

        return minibatch_probs.cpu()

lm_server = None

class Seq2SeqLMScoring(unittest.TestCase):
    PROMPTS = ["The capital of France is", "How are you"]

    def test_vanilla_scoring(self):
        global lm_server
        generated = lm_server.generate(contexts=self.PROMPTS, num_return_sequences=3, return_logprobs=True,
                                       do_sample=True, top_k=None, max_new_tokens=5, temperature=1, top_p=1.0)
        scores = []
        for idx, _prompt in enumerate(self.PROMPTS):
            _scores = lm_server.custom_module_fns(['score'],
                                                contexts=[_prompt],
                                                candidates=[
                                                    [_text['text'] for _text in generated[idx]]
                                                ])

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
        global lm_server
        generated = lm_server.generate(contexts=self.PROMPTS, num_return_sequences=3, return_logprobs=True,
                                        do_sample=True, top_k=None, max_new_tokens=5, temperature=1, top_p=1)

        _scores = lm_server.custom_module_fns(['score'],
                                            contexts=self.PROMPTS,
                                            candidates=[
                                                [_text['text'] for _text in _result]
                                                for _result in generated
                                            ])

        scores = numpy.array([_s['score'].numpy() for _s in _scores])
        generated_scores = numpy.array([
            [_text['text_logprob'] for _text in _result] for _result in generated
        ])
        try:
            numpy.testing.assert_array_almost_equal(scores, generated_scores, 1)
        except AssertionError as e:
            self.fail()


@hydra.main(config_path='config', config_name='config')
def main(config_args):
    global lm_server
    lm_server = Caller(config_args.lamorel_args,
                       custom_module_functions={
                           "main_llm":{
                                'score': LogScoringModuleFn()
                            }
                       })
    causal_lm_scoring_suite = unittest.TestLoader() \
        .loadTestsFromTestCase(Seq2SeqLMScoring)
    runner = unittest.TextTestRunner()
    runner.run(causal_lm_scoring_suite)
    lm_server.close()


if __name__ == '__main__':
    main()

