import unittest
import torch
import hydra
import numpy

from torch.nn.functional import log_softmax

from lamorel import Caller, BaseModuleFunction, lamorel_init

lamorel_init()
class LogScoringModuleFn(BaseModuleFunction):
    def __init__(self, model_type, pre_encoded_input):
        super().__init__()
        self._model_type = model_type
        self._pad_token = 0
        self._pre_encoded_input = pre_encoded_input

    def initialize(self):
        pass

    def forward(self, forward_outputs, minibatch, tokenized_contexts, **kwargs):
        if self._model_type == "causal":
            if self._pre_encoded_input:
                end_of_context_position = 0
            else:  # hence input should be removed from result
                end_of_context_position = len(
                    tokenized_contexts[0]["input_ids"])  # inputs are padded so all of same size and last token was sent to outputs (so no -1)

            logits = forward_outputs["logits"][:, end_of_context_position:-1, :]
            output_tokens = minibatch["input_ids"][:, end_of_context_position+1:]
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

lm_server = None

class CausalLMScoring(unittest.TestCase):
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
                           'score': LogScoringModuleFn(config_args.lamorel_args.llm_args.model_type,
                                                       config_args.lamorel_args.llm_args.pre_encode_inputs)
                       })
    causal_lm_scoring_suite = unittest.TestLoader() \
        .loadTestsFromTestCase(CausalLMScoring)
    runner = unittest.TextTestRunner()
    runner.run(causal_lm_scoring_suite)
    lm_server.close()


if __name__ == '__main__':
    main()

