from datetime import timedelta
import os
from torch.distributed import init_process_group

import logging
lamorel_logger = logging.getLogger('lamorel_logger')

def init_distributed_setup(config):
    if "timeout" in config:
        lamorel_logger.info(f"Setting the timeout to {int(config.timeout)} seconds.")
        timeout = timedelta(seconds=int(config.timeout))
    else:
        lamorel_logger.info(f"No configuration found for the timeout, setting it to default: 1800 seconds.")
        timeout = timedelta(seconds=1800)

    if "backend" in config:
        assert config.backend in ["gloo", "nccl"], "Provided an unrecognized backend. Authorized: ['gloo', 'nccl']."
        lamorel_logger.info(f"Using {config.backend} backend.")
        backend = config.backend
    else:
        backend = "gloo"
        lamorel_logger.info(f"No backend provided, setting it to default: 'gloo'.")

    try:
        init_process_group(
            rank=int(os.environ["RANK"]),
            world_size=int(os.environ["WORLD_SIZE"]),
            backend=backend,
            timeout=timedelta(seconds=int(config.init_timeout) if "init_timeout" in config and config.init_timeout is not None else 120)
        )
        lamorel_logger.info(f"Successfully initialized distributed process on process {os.environ['RANK']}")
        return backend, timeout
    except Exception as e:
        lamorel_logger.error(f"Error: {e}")
        lamorel_logger.error(f'MASTER_ADDR: {os.environ["MASTER_ADDR"]}')
        lamorel_logger.error(f'MASTER_PORT: {os.environ["MASTER_PORT"]}')
        lamorel_logger.error(f'WORLD_SIZE: {os.environ["WORLD_SIZE"]}')
        lamorel_logger.error(f'RANK: {os.environ["RANK"]}')
        lamorel_logger.error(f'LOCAL_RANK: {os.environ["LOCAL_RANK"]}')
        raise e