# VLLM BUG details
These implementation notes are provided to **save time** and offer **context** for future Lamorel contributors working on VLLM support.
#### Activation

Once the initial VLLM integration was completed, we tested it by enabling the following configuration:

```python
config_args.use_vllm = true
```
#### Observed Behavior

Upon running a standard Lamorel script with VLLM enabled, the execution **blocked** during runtime and produced the following output:


```
python child_process_test.py
main process pid: 37424
[W socket.cpp:464] [c10d] The server socket cannot be initialized on [::]:5556 (errno: 97 - Address family not supported by protocol).
[W socket.cpp:697] [c10d] The client socket cannot be initialized to connect to [localhost]:5556 (errno: 97 - Address family not supported by protocol).
[W socket.cpp:697] [c10d] The client socket cannot be initialized to connect to [localhost]:5556 (errno: 97 - Address family not supported by protocol).
[DEBUG]: proccess with rank:1 and pid:37517 
PG world size: 2
[DEBUG]: proccess with rank:0 and pid:37516 
PG world size: 2
WARNING 08-20 11:57:51 config.py:1354] Casting torch.bfloat16 to torch.float16.
INFO 08-20 11:57:51 llm_engine.py:169] Initializing an LLM engine (v0.5.1) with config: model='/beegfs/abouyghf/Mistral_7B/', speculative_config=None, tokenizer='/beegfs/abouyghf/Mistral_7B/', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, rope_scaling=None, rope_theta=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=32768, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cuda, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None), seed=0, served_model_name=/beegfs/abouyghf/Mistral_7B/, use_v2_block_manager=False, enable_prefix_caching=False)
INFO 08-20 11:57:51 selector.py:172] Cannot use FlashAttention-2 backend due to sliding window.
INFO 08-20 11:57:51 selector.py:53] Using XFormers backend.```
```


###  Describe the bug
After some debug lines we concluded that bug happens inside the vllm.LLM constructor:  
Model loading hangs when LLM(model="...", ..) is called by a subprocess spawned with torch.multiprocessing.

The GPU memory usage remains stuck at 427 MiB / 40960 MiB (according to nvidia-smi), whereas it normally reaches around 15000 MiB when the model is successfully loaded in a single process.


Lmaorel's backend initialize  a process_group to handle communication protocol betwen the spawned prcesses.
#### Bug isolation:
We had the idea to reproduce the bug within a smaller code, that doesn't include all the lamorel's setups
and the code had the same behavior 
here's the code:

```python
import os
from vllm import LLM
from multiprocessing import Process
import time


import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank,size):
    pid = os.getpid()
    print(f"[DEBUG]: proccess with rank:{rank} and pid:{pid} ")
    print("PG world size:",  dist.get_world_size(), sep= " ")
    if rank == 0:
        llm = LLM(model="/beegfs/abouyghf/Mistral_7B/" ,dtype="float16", tensor_parallel_size= 1, pipeline_parallel_size= 1)


def init_process(rank, size, fn, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '5556'
    torch.cuda.set_device(rank)
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

if __name__ == "__main__":
    main_pid = os.getpid()
    print("main process pid:", main_pid, sep=" ")
    world_size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(world_size):
        p = mp.Process(target=init_process, args=(rank, world_size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

```
After reviewing the source code of `vllm` in `vllm-project/vllm/distributed/parallel_state.py`, it appears that `vllm` also initializes its own process group using `torch.distributed`. This raised concerns about a potential configuration conflict between the distributed settings defined in `lamorel` and those internally managed by `vllm`, which may be the root cause of the observed issue.


After setting debug levels of torch.distributed backeds some logs revealed that the `world_size` parameter passed to vllm's process group is set to 2. wich is excactly the external (lamorel) world_size parameter.
 


After a consequent time of debuguing we discovered the reason behind the bug: 

In the function in vllm/distributed/parallel_state.py
```python
def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str = "nccl",
):
    logger.debug(
        "world_size=%d rank=%d local_rank=%d "
        "distributed_init_method=%s backend=%s", world_size, rank, local_rank,
        distributed_init_method, backend)
    from vllm.config import get_current_vllm_config
    config = get_current_vllm_config()
    if config is not None and config.parallel_config.data_parallel_size > 1:
        parallel_config = config.parallel_config
        # adjust to take into account data parallelism
        # offset the rank by the data parallel rank
        rank = parallel_config.data_parallel_rank * world_size + rank
        # adjust the world size to take into account data parallelism
        world_size = parallel_config.world_size_across_dp
        ip = parallel_config.data_parallel_master_ip
        port = parallel_config.get_next_dp_init_port()
        distributed_init_method = get_distributed_init_method(ip, port)
        logger.info(
            "Adjusting world_size=%d rank=%d distributed_init_method=%s for DP",
            world_size, rank, distributed_init_method)
    if not torch.distributed.is_initialized():
        assert distributed_init_method is not None, (
            "distributed_init_method must be provided when initializing "
            "distributed environment")
        if not torch.distributed.is_backend_available(backend):
            logger.warning(
                "Distributed backend %s is not available; "
                "falling back to gloo.", backend)
            assert torch.distributed.is_gloo_available(), (
                "Fallback Gloo backend is not available.")
            backend = "gloo"
        # this backend is used for WORLD
        torch.distributed.init_process_group(
            backend=backend,
            init_method=distributed_init_method,
            world_size=world_size,
            rank=rank)
    # set the local rank
    # local_rank is not available in torch ProcessGroup,
    # see https://github.com/pytorch/pytorch/issues/122816
    if local_rank == -1:
        # local rank not set, this usually happens in single-node
        # setting, where we can use rank as local rank
        if distributed_init_method == "env://":
            local_rank = envs.LOCAL_RANK
        else:
            local_rank = rank
    global _WORLD, _NODE_COUNT
    if _WORLD is None:
        ranks = list(range(torch.distributed.get_world_size()))
        _WORLD = init_world_group(ranks, local_rank, backend)
        _NODE_COUNT = _node_count(_WORLD.cpu_group)
        logger.debug("Detected %d nodes in the distributed environment",
                     _NODE_COUNT)
    else:
        assert _WORLD.world_size == torch.distributed.get_world_size(), (
            "world group already initialized with a different world size")

```

#### Explanations

Since `torch.distributed` is already initialized by `lamorel`, the function in `vllm` directly enters the branch `if _WORLD is None:`. As a result, it attempts to create a `world_group` based on the existing process group — the one created by `lamorel` — which has ranks `[0, 1]`. This leads to a `world_size = 2`.

However, only one process is actually active, meaning the second expected process does not exist and cannot respond to the `init_world_group` function call. This mismatch causes a deadlock during process group initialization.

