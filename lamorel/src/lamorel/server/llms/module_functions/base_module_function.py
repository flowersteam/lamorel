import logging
lamorel_logger = logging.getLogger('lamorel_logger')

from torch.nn import Module

class BaseModuleFunction(Module):
    def __init__(self):
        super().__init__()
        self.llm_config = None
        self.model_config = None
        self.device = None
        self.current_process_config = None

    def initialize(self):
        '''
        Use this method to initialize your module operations
        '''
        raise NotImplementedError()

    def forward(self, forward_outputs, minibatch, tokenized_contexts, current_minibatch_ids, **kwargs):
        raise NotImplementedError()
