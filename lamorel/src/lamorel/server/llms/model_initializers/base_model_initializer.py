import logging
lamorel_logger = logging.getLogger('lamorel_logger')

class BaseModelInitializer:
    def __init__(self):
        self.llm_config = None
        self.model_config = None
        self.current_process_config = None

    def initialize_model(self, model):
        raise NotImplementedError()
