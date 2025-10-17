from enum import Enum
from .hf_llm import HF_LLM
# from .unsloth_llm import UnslothLLM

class HandlersEnum(Enum):
    hf_llm = HF_LLM
    # unsloth = UnslothLLM