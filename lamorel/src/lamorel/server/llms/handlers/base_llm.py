import torch

class BaseLLM(torch.nn.Module):
    def __init__(self, args, devices, process_index, use_cpu=False):
        super().__init__()
        self.process_index = process_index
        # self.device_id = self.devices[0]
        self.use_cpu = use_cpu
        if self.use_cpu:
            self.main_device = 'cpu'
            self.devices = None
        else:
            self.main_device = torch.device(f'cuda:{devices[0]}')  # use first device of map for tensors
            self.devices = devices

    def register_module_functions(self, module_functions):
        raise NotImplementedError()

    def generate(self, contexts, **kwargs):
        raise NotImplementedError()

    def get_model_config(self):
        raise NotImplementedError()

    def get_additional_llm_config(self):
        raise NotImplementedError()

    def forward(self, module_function_keys, contexts, candidates=None, require_grad=False, **kwargs):
        raise NotImplementedError()