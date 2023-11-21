## Context
We provide a lightweight implementation of the PPO finetuning performed in ["Grounding Large Language Models in Interactive Environments with Online Reinforcement Learning"](https://arxiv.org/abs/2302.02662).
We use [Spinning Up's PPO implementation](https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ppo).

We leverage Lamorel's custom modules and updaters to add a value head on top of the LLM and finetune all the weights using the PPO loss.

## Installation
1. Install [BabyAI-Text](https://github.com/flowersteam/Grounding_LLMs_with_online_RL/tree/main/babyai-text) environment
2. Install required packages: `pip install -r requirements.txt`

## Launch
To launch the example using a single GPU on a local machine:
1. Spawn a process for the RL code:
```bash
python -m lamorel_launcher.launch \
       --config-path PROJECT_PATH/examples/PPO_finetuning/ \ 
       --config-name PROJECT_PATH/examples/PPO_finetuning/local_gpu_config \
       rl_script_args.path=PROJECT_PATH/examples/PPO_finetuning/main.py \
       rl_script_args.output_dir=YOUR_OUTPUT_DIR \
       lamorel_args.accelerate_args.machine_rank=0 \
       lamorel_args.llm_args.model_path=PATH_TO_YOUR_LLM
```

2. Spawn a process for the LLM:
```bash
python -m lamorel_launcher.launch \ 
       --config-path PROJECT_PATH/examples/PPO_finetuning/ \
       --config-name PROJECT_PATH/examples/PPO_finetuning/local_gpu_config \
       rl_script_args.path=PROJECT_PATH/examples/PPO_finetuning/main.py \
       rl_script_args.output_dir=YOUR_OUTPUT_DIR \
       lamorel_args.accelerate_args.machine_rank=1 \
       lamorel_args.llm_args.model_path=PATH_TO_YOUR_LLM
```