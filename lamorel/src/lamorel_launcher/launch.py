import os
import hydra
from omegaconf import OmegaConf,open_dict
from hydra.main import get_args_parser
from accelerate.commands.launch import launch_command_parser, launch_command

def prepare_args_for_parsing(dict_args):
    args = []
    for key, value in dict_args.items():
        args.append(f"--{key}")
        if value is not None:
            args.append(f"{value}")
    return args

@hydra.main(config_path='', config_name='')
def main(config_args):
    # Get args processed by hydra for future use
    hydra_parser = get_args_parser()
    hydra_args = hydra_parser.parse_args()

    # Compute total number of processes and give it to accelerate
    with open_dict(config_args):
        config_args.lamorel_args.accelerate_args.num_processes = \
            config_args.lamorel_args.distributed_setup_args.n_rl_processes + \
            config_args.lamorel_args.distributed_setup_args.n_llm_processes

    # Parse accelerate arguments from config for distributed launch
    accelerate_parser = launch_command_parser()
    config_args.lamorel_args.accelerate_args.config_file = os.path.join(
        hydra_args.config_path, config_args.lamorel_args.accelerate_args.config_file)
    accelerate_args_to_parse = prepare_args_for_parsing(config_args.lamorel_args.accelerate_args)
    accelerate_args_to_parse.append(config_args.rl_script_args.path)
    accelerate_args = accelerate_parser.parse_args(accelerate_args_to_parse)

    # Prepare launch
    hydra_args.overrides.append('hydra.run.dir=.') #Overwriting hydra config to avoid nested output dirs
    accelerate_args.training_script_args = hydra_args.overrides
    accelerate_args.training_script_args.extend([
        "--config-path", hydra_args.config_path,
        "--config-name", hydra_args.config_name
    ])

    if config_args.lamorel_args.llm_args.parallelism.use_gpu is False:
        os.environ["USE_CPU"] = str(True)

    launch_command(accelerate_args)


if __name__ == '__main__':
    main()