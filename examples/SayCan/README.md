## Context
The code of the environment (along with prompt generation and affordance function) was taken from the official notebook release with SayCan: https://github.com/google-research/google-research/blob/master/saycan/SayCan-Robot-Pick-Place.ipynb

We provide in [`run_pickplace_saycan.py`](run_pickplace_saycan.py) a small script that uses Lamorel (instead of GPT3's API) to score actions.
Scoring an action with Lamorel (no matter neither which LLM is used nor whether a distributed setup is used or not) is a single line of code:
```python
lm_server.score([prompt], [possible_actions])
```

## Installation
1. Install required packages: `pip install -r requirements.txt`

## Launch
To launch the example using a single GPU on a local machine:
1. Spawn a process for the RL code:
```bash
python -m lamorel_launcher.launch \ 
       --config-path PROJECT_PATH/examples/SayCan/ \
       --config-name PROJECT_PATH/examples/SayCan/local_gpu_config \
       rl_script_args.path=PROJECT_PATH/examples/SayCan/run_pickplace_saycan.py \
       lamorel_args.accelerate_args.machine_rank=0 \
       lamorel_args.llm_args.model_path=PATH_TO_YOUR_LLM
```

2. Spawn a process for the LLM:
```bash
python -m lamorel_launcher.launch \
       --config-path PROJECT_PATH/examples/SayCan/ \
       --config-name PROJECT_PATH/examples/SayCan/local_gpu_config \
       rl_script_args.path=PROJECT_PATH/examples/SayCan/run_pickplace_saycan.py \
       lamorel_args.accelerate_args.machine_rank=1 \
       lamorel_args.llm_args.model_path=PATH_TO_YOUR_LLM
```