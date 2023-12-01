## Context
We provide a lightweight implementation of the PPO finetuning performed in ["Grounding Large Language Models in Interactive Environments with Online Reinforcement Learning"](https://arxiv.org/abs/2302.02662).
We use [LoRA](https://arxiv.org/abs/2106.09685) through the [Peft](https://github.com/huggingface/peft) library for lightweight finetuning.

We leverage Lamorel's custom modules and updaters to add a value head on top of the LLM and finetune all the weights using the PPO loss.
Finally, using Lamorel's initializer, we add LoRA's adapters to the LLM (which are then automatically synchronized by Lamorel if multiple LLM instances are deployed).

## Installation
1. Install [BabyAI-Text](https://github.com/flowersteam/Grounding_LLMs_with_online_RL/tree/main/babyai-text) environment
2. Install required packages: `pip install -r requirements.txt`

## Launch
1. Spawn both processes (RL collecting data and LLM):
<!-- ```bash
python -m lamorel_launcher.launch \
       --config-path /home/ewanlee/Codes/lamorel/examples/PPO_LoRA_finetuning/ \ 
       --config-name /home/ewanlee/Codes/lamorel/examples/PPO_LoRA_finetuning/local_gpu_config \
       rl_script_args.path=PROJECT_PATH/examples/PPO_finetuning/main.py \
       rl_script_args.output_dir=YOUR_OUTPUT_DIR \
       lamorel_args.accelerate_args.machine_rank=0 \
       lamorel_args.llm_args.model_path=PATH_TO_YOUR_LLM
``` -->
```bash
TRANSFORMERS_OFFLINE=1 python -m lamorel_launcher.launch \
--config-path /home/ewanlee/Codes/lamorel/examples/PPO_LoRA_finetuning/ \
--config-name local_gpu_config \
rl_script_args.path=/home/ewanlee/Codes/lamorel/examples/PPO_LoRA_finetuning/main.py \
rl_script_args.output_dir=/home/ewanlee/Codes/lamorel/examples/PPO_LoRA_finetuning/logs \
```