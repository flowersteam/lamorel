import os
import hydra
import logging
from omegaconf import OmegaConf,open_dict
from hydra.main import get_args_parser
from argparse import Namespace
import torch.distributed.run as distrib_run

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

    # Compute total number of processes
    n_llm_processes = 0
    for _llm_name, _llm_config in config_args.lamorel_args.distributed_setup_args.llm_processes.items():
        n_llm_processes += _llm_config.n_processes
        assert len(_llm_config.devices_per_process) == _llm_config.n_processes, f"Please provide the devices for each process for LLM {_llm_name}"

    torch_distributed_args = {}
    n_processes = n_llm_processes + config_args.lamorel_args.distributed_setup_args.n_rl_processes
    if config_args.lamorel_args.distributed_setup_args.multinode:
        logging.warning("Asking for a multinode setup, make sure you launch as many times this launches as the total number of asked processes.")
        torch_distributed_args["nnodes"] = str(n_processes)
        torch_distributed_args["nproc_per_node"] = 1
        torch_distributed_args["standalone"] = False
        torch_distributed_args["rdzv-id"] = config_args.lamorel_args.distributed_setup_args.multinode_args.experiment_id
        torch_distributed_args["rdzv-backend"] = "c10d"
        torch_distributed_args[
            "rdzv-endpoint"] = f"{config_args.lamorel_args.distributed_setup_args.multinode_args.main_process_ip}:{config_args.lamorel_args.distributed_setup_args.multinode_args.main_process_port}"
    else:
        torch_distributed_args["nnodes"] = "1"
        torch_distributed_args["nproc_per_node"] = n_processes
        torch_distributed_args["standalone"] = True

    torch_distributed_argparser = distrib_run.get_args_parser()
    parsed_args = torch_distributed_argparser.parse_args(prepare_args_for_parsing(torch_distributed_args))

    # Prepare launch
    hydra_args.overrides.append('hydra.run.dir=.')  # Overwriting hydra config to avoid nested output dirs
    for _override in hydra_args.overrides:
        current_override = _override.split("=")
        if len(current_override) > 1 and current_override[0] == "rl_script_args.path":
            setattr(parsed_args, "training_script", current_override[1])

        parsed_args.training_script_args.append(_override)

    parsed_args.training_script_args.append(f"--config-path={hydra_args.config_path}")
    parsed_args.training_script_args.append(f"--config-name={hydra_args.config_name}")

    distrib_run.run(parsed_args)


if __name__ == '__main__':
    main()