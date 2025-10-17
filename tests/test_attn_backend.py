import os
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "0"

from unsloth import FastLanguageModel
import time
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
torch.nn.attention.WARN_FOR_UNFUSED_KERNELS = True
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
set_seed(0)

# model_name = "/lus/work/CT10/iso1996/SHARED/LLMs/Llama-2-7b-hf"
model_name = "/lus/work/CT10/iso1996/SHARED/LLMs/llama-2-7b-chat-bnb-4bit"
# model_name = "/lustre/fsn1/projects/rech/imi/ucy39hi/LLMs/llama-2-7b-chat-bnb-4bit"

prompt = ["That is a true story.",
        "It’s true that she loves coffee.",
        "True friends are hard to find.",
        "He remained true to his word.",]
print("Loading tokenizer")
model, tokenizer = FastLanguageModel.from_pretrained(model_name, device_map="auto") #, attn_implementation="flash_attention_2")
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
# tokenizer.pad_token = tokenizer.eos_token

def test():
    _time = time.time()
    print("Natural order output:")
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True).to(model.model.embed_tokens.weight.device)
    print(model(output_hidden_states=True, **input_ids)['hidden_states'][-1][:, -1, :].sum(-1))

    print("Reverse order output:")
    input_ids = tokenizer(list(reversed(prompt)), padding=True, return_tensors="pt").to(model.model.embed_tokens.weight.device)
    print(model(output_hidden_states=True, **input_ids)['hidden_states'][-1][:, -1, :].sum(-1))
    print(f"Done in {time.time() - _time}s")

print("Default backend:")
# torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", attn_implementation="flash_attention_2",torch_dtype=torch.float16,)
# model.eval()
test()

print("MATH backend:")
with sdpa_kernel(SDPBackend.MATH):
    test()

try:
    print("XFormer backend:")
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        test()
except Exception as err:
    print(f"Failed: {err}")

try:
    print("Flash Attention backend:")
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        test()
except Exception as err:
    print(f"Failed: {err}")

# print("TEST 4:")
# torch.backends.cuda.enable_flash_sdp(False)
# test()

# print("TEST 5:")
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", attn_implementation="sdpa")
# model.eval()
# test()