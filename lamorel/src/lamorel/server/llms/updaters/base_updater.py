import logging
lamorel_logger = logging.getLogger('lamorel_logger')

class BaseUpdater:
    def __init__(self):
        self._llm_module = None

    def set_llm_module(self, llm_module):
        self._llm_module = llm_module

    def perform_update(self, contexts, candidates, _current_batch_ids, **kwargs):
        lamorel_logger.warning("Calling update but no updater was provided, ignoring.")
