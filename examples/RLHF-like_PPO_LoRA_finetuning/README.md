## Context
We provide a simple example of PPO finetuning on a LLM which is asked to generate specific token sequences.
As in RLHF, each token is a different action.

We use [LoRA](https://arxiv.org/abs/2106.09685) through the [Peft](https://github.com/huggingface/peft) library for lightweight finetuning.
We leverage Lamorel's custom modules and updaters to add a value head on top of the LLM and finetune all the weights using the PPO loss.

## Installation
1.Install required packages: `pip install -r requirements.txt`

## Launch
To launch the example using a single GPU on a local machine:
1. Spawn both processes (RL collecting data and LLM):
```bash
python -m lamorel_launcher.launch \
       --config-path PROJECT_PATH/examples/PPO_finetuning/ \ 
       --config-name PROJECT_PATH/examples/PPO_finetuning/local_gpu_config \
       rl_script_args.path=PROJECT_PATH/examples/PPO_finetuning/main.py \
       rl_script_args.output_dir=YOUR_OUTPUT_DIR \
```