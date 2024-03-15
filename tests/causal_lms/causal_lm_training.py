import unittest
import hydra
import torch
import numpy as np
import math

from lamorel import Caller, lamorel_init
from lamorel import BaseUpdater, BaseModuleFunction
lamorel_init()


class SigmoidOutputModuleFn(BaseModuleFunction):
    def __init__(self, model_type):
        super().__init__()
        self._model_type = model_type

    def initialize(self):
        if 'hidden_size' in self.llm_config.attribute_map:
            _hidden_size_key = self.llm_config.attribute_map['hidden_size']
        else:
            if "word_embed_proj_dim" in self.llm_config.to_dict():
                _hidden_size_key = "word_embed_proj_dim"
            elif "hidden_size" in self.llm_config.to_dict():
                _hidden_size_key = "hidden_size"
            else:
                print(self.llm_config.to_dict())
                raise NotImplementedError("Unknown hidden size key")

        self._llm_hidden_size = self.llm_config.to_dict()[_hidden_size_key]
        if "num_hidden_layers" in self.llm_config.attribute_map:
            hidden_layers_config_key = self.llm_config.attribute_map["num_hidden_layers"]
        else:
            hidden_layers_config_key = "num_hidden_layers"
        self._last_hidden_layer = self.llm_config.to_dict()[hidden_layers_config_key] - 1

        self.head_op = torch.nn.Sequential(
            torch.nn.Linear(self._llm_hidden_size, 64),
            torch.nn.Sigmoid(),
            torch.nn.Linear(64, 1),
            torch.nn.Sigmoid()
        ).to(self.device)

    def forward(self, forward_outputs, minibatch, tokenized_contexts, **kwargs):
        # Get last layer's hidden from last token in context
        if self._model_type == "causal":
            model_head = forward_outputs['hidden_states'][self._last_hidden_layer][:, -1, :]  # WTF ???
        else:
            model_head = forward_outputs["decoder_hidden_states"][self._last_hidden_layer + 1][:, 0, :]

        output = self.head_op(model_head.to(torch.float32).to(self.device))
        return output.cpu()


class BCEUpdater(BaseUpdater):
    def __init__(self, model_type, minibatch_size, gradient_batch_size, gradient_minibatch_size=None,
                 use_all_params_for_optim=True):
        super(BCEUpdater, self).__init__()
        self._model_type = model_type
        self._minibatch_size = minibatch_size
        self._gradient_batch_size = gradient_batch_size
        self._gradient_minibatch_size = gradient_minibatch_size
        self._use_all_params_for_optim = use_all_params_for_optim

    def get_trainable_params(self, model, return_with_names=False):
        if return_with_names:
            return ((n, p) for n, p in model.named_parameters() if p.requires_grad)
        else:
            return (p for p in model.parameters() if p.requires_grad)

    def perform_update(self, contexts, candidates, _current_batch_ids, **kwargs):
        if not hasattr(self, 'loss_fn'):
            self.loss_fn = torch.nn.BCELoss()

        if not hasattr(self, 'optimizer'):
            if self._use_all_params_for_optim:
                # iterator_trainable_params = self._llm_module.parameters()
                for param in self._llm_module.module._LLM_model.parameters():
                    param.requires_grad = False
                iterator_trainable_params = self._llm_module.module._module_functions['sigmoid_output'].parameters()
            else:
                iterator_trainable_params = self.get_trainable_params(self._llm_module, False)

            self.optimizer = torch.optim.Adam(iterator_trainable_params, lr=5e-3)

        current_process_buffer = {}
        for k in ['labels']:
            current_process_buffer[k] = kwargs[k][_current_batch_ids]

        losses = []
        n_minibatches = math.ceil(len(contexts) / self._minibatch_size)
        for step in range(n_minibatches):
            _minibatch_start_idx = step * self._minibatch_size
            _minibatch_end_idx = min(
                (step + 1) * self._minibatch_size,
                len(contexts))

            self.optimizer.zero_grad()
            gradient_accumulation_steps = math.ceil(
                _minibatch_end_idx - _minibatch_start_idx / self._gradient_batch_size)
            for accumulated_batch in range(gradient_accumulation_steps):
                _start_idx = _minibatch_start_idx + accumulated_batch * self._gradient_batch_size
                _stop_idx = _minibatch_start_idx + min(
                    (accumulated_batch + 1) * self._gradient_batch_size, _minibatch_end_idx)
                _contexts = contexts[_start_idx:_stop_idx]
                _labels = current_process_buffer['labels'][_start_idx:_stop_idx]
                if len(_contexts) == 0: break
                if self._gradient_minibatch_size is None:
                    _batch_size = self._gradient_batch_size
                else:
                    _batch_size = self._gradient_minibatch_size

                output = self._llm_module(['sigmoid_output'], contexts=_contexts, require_grad=True,
                                          minibatch_size=_batch_size)
                predicted = torch.stack([_o["sigmoid_output"][0] for _o in output])
                loss = self.loss_fn(predicted, _labels) / gradient_accumulation_steps
                losses.append(loss.detach().cpu().item())
                loss.backward()

            self.optimizer.step()

        return {"loss": np.mean(losses)}

lm_server = None

class CausalLMTraining(unittest.TestCase):
    train_dataset = {
        "x": [
            "This sentence is true",
            "This is just true",
            "Yet another true",
            "A sentence that is true",
            "This sentence is false",
            "This is just false",
            "Yet another false",
            "Don't focus on false"
        ],
        "y": [
            [1],
            [1],
            [1],
            [1],
            [0],
            [0],
            [0],
            [0],
        ]
    }

    test_dataset = {
        "x": [
            "What about this true",
            "Or a true like this",
            "And another false",
            "Here is a sentence that is false"
        ],
        "y": [
            1,
            1,
            0,
            0,
        ]
    }

    def test_after_training(self):
        global lm_server
        for i in range(100):
            results = lm_server.update(
                contexts=self.train_dataset["x"],
                candidates=None,
                labels=torch.tensor(self.train_dataset["y"], dtype=torch.float32),
            )
            print(f"Loss at step {i}: {np.mean([l['loss'] for l in results])}")

        tests = lm_server.custom_module_fns(module_function_keys=['sigmoid_output'],
                                            contexts=self.test_dataset["x"])
        np.testing.assert_array_almost_equal([_r['sigmoid_output'].item() for _r in tests], self.test_dataset["y"],
                                             decimal=2)


@hydra.main(config_path='config', config_name='config')
def main(config_args):
    global lm_server
    # lm server
    lm_server = Caller(config_args.lamorel_args,
                       custom_updater=BCEUpdater(config_args.lamorel_args.llm_args.model_type,
                                                 config_args.rl_script_args.minibatch_size,
                                                 config_args.rl_script_args.gradient_batch_size,
                                                 config_args.rl_script_args.gradient_minibatch_size,
                                                 config_args.rl_script_args.use_all_params_for_optim),
                       custom_module_functions={
                           'sigmoid_output': SigmoidOutputModuleFn(config_args.lamorel_args.llm_args.model_type)
                       })
    causal_lm_training_suite = unittest.TestLoader() \
        .loadTestsFromTestCase(CausalLMTraining)
    runner = unittest.TextTestRunner()
    runner.run(causal_lm_training_suite)
    lm_server.close()

if __name__ == '__main__':
    main()