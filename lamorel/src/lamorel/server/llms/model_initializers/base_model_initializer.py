import logging
lamorel_logger = logging.getLogger('lamorel_logger')

class BaseModelInitializer:
    def __init__(self):
        pass

    def initialize_model(self, model):
        raise NotImplementedError()
