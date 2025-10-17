# Language Models for Reinforcement Learning - *Lamorel*

*Lamorel* is a Python library designed for people eager to use Large Language Models (LLMs) in interactive environments (e.g. RL setups).

---
## ** *News* **
- **2025/04/26 - V0.3**:
  - Multiple LLMs can be deployed.
  - Peft support has been improved.
  - Unsloth LLMs are now supported (see this [checklist](docs/unsloth.md) for using unsloth models).
  - More control on padding options.
  - Module heads can now return dicts.
  - Follow [this quick tutorial](docs/migrate_0.2_0.3.md) to migrate from V0.2 to V0.3. 
- **2023/11/21 - V0.2**: 
  - The support of Decoder-Only models has been largely improved.
  - Optimizations:
    - contexts sent to lamorel are automatically padded, easing the use of custom modules (see [examples](examples/)).
    - batching has been improved.
    - `pre_encode_inputs: true` now works for all models, allowing one to cache contexts.
    - quantization has been added (please use `pip install .[quantization]` and set `load_in_4bit: true` in your config) using [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) through [Accelerate](https://github.com/huggingface/accelerate) and [Transformers](https://github.com/huggingface/transformers).
  - Simply setting `load_in_4bit: true` in the fongi of [PPO_LoRA_finetuning example](examples/PPO_LoRA_finetuning) results in using [QLoRA](https://arxiv.org/abs/2305.14314).
  - [Tests](tests) have been added to ensure scoring and training properly work.
  - [PPO_finetuning](examples/PPO_finetuning) and [PPO_LoRA_finetuning](examples/PPO_LoRA_finetuning) have been improved:
    - gradient accumulation has been fixed.
    - you can now load finetuned weights with `loading_path`.
    - the environment is now vectorized for faster training.
  - A new [example](examples/RLHF-like_PPO_LoRA_finetuning) shows how to consider tokens as actions as in an RLHF setup (this example can be used for RLHF purposes by modifying the reward).
- **2023/07/12**: an [example](examples/PPO_LoRA_finetuning) showing how to use [LoRA](https://arxiv.org/abs/2106.09685) through the [Peft](https://github.com/huggingface/peft) library for lightweight finetuning has been added.

---
## Why *Lamorel*?
### What is the difference between *Lamorel* and RLHF libs?

*Lamorel* was initially designed to easily use LLMs in interactive environments. 
It is especially made for high throughput using a distributed architecture.
The philosophy of *Lamorel* is to be very permissive and allow as much as possible usage of LLMs while maintaining scaling: the application should run with 1 or N LLMs.

For this reason, it is not specialised neither in RL nor in particular in RLHF. 
Our [examples](examples) illustrate how *Lamorel* can be used for various applications including [RLHF-like finetuning](examples/RLHF-like_PPO_LoRA_finetuning).
However, one must understand that *Lamorel*'s philosophy means that users must implement themselves what they want to do with the LLM(s).

This is why we advise users knowing in advance they want to do RLHF, especially without any modification of classic implementations, to use libs specialised in RLHF that already come with RL implementations (e.g. [RL4LMs](https://github.com/allenai/RL4LMs), [TRL](https://github.com/lvwerra/trl)).
On the other hand, users more inclined to experiment with implementations or looking for an LLM lib they can use in different projects may prefer *Lamorel*.

### *Lamorel*'s key features
1. Abstracts the use of LLMs (e.g. tonekization, batches) into simple calls
```python
lm_server.generate(contexts=["This is an examples prompt, continue it with"])
lm_server.score(contexts=["This is an examples prompt, continue it with"], candidates=["a sentence", "another sentence"])
```
2. Provides a method to compute the probability of token sequences (e.g. action commands) given a prompt
3. Provides access to open-sourced LLMs from the [Hugging Face's hub](https://huggingface.co/models) along with  [Model Parallelism](https://huggingface.co/docs/transformers/v4.15.0/parallelism) to use multiple GPUs for an LLM instance
```yaml
  main_llm:
    handler: hf_llm  # or unsloth
    constructor_kwargs:  # any argument you'd like to pass to the model's constructor
    model_type: causal
    model_path: distilgpt2
    pretrained: true
    minibatch_size: 256
    pad_token_id:
    pre_encode_inputs: false
    load_in_4bit: false
    synchronize_gpus_after_scoring: false
    empty_cuda_cache_after_scoring: false
```
4. Is made for scaling up your experiments by deploying multiple instances of the LLM and dispatching the computation thanks to a simple configuration file
```yaml
    distributed_setup_args:
      backend: gloo  # nccl can also be used but lamorel still uses cpu operations
      init_timeout: 120
      timeout: 1800
      multinode: false
      multinode_args:
        main_process_ip:
        main_process_port:
        experiment_id:
      n_rl_processes: 1
      llm_processes:
        main_llm:
          n_processes: 1
          devices_per_process: [[0]]  # either "cpu" or list of gpu ids
          ddp_kwargs:
```
4. Supports multiple LLMs at the same time.
```yaml
    llm_processes:
      main_llm:
        n_processes: 1
        devices_per_process: [[0]]  # either "cpu" or list of gpu ids
        ddp_kwargs:
      secondary_llm:
        n_processes: 2
        devices_per_process: ["cpu","cpu"]  # either "cpu" or list of gpu ids
        ddp_kwargs:
```
5. Allows one to give their own PyTorch modules to compute custom operations (e.g. to add new heads on top of the LLM)
6. Allows one to train the LLM (or part of it) thanks to a [Data Parallelism](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) setup where the user provides its own update method 


## Installation
1. `cd lamorel`
2. `pip install .`

## How to use *Lamorel*
*Lamorel* is built of three main components:
- a Python API to interact with LLMs
- a configuration file to set the LLM servers
- a launcher deploying the multiple LLM servers and launching your RL script

### Instantiating the server in your RL script
*Lamorel* leverages [hydra](https://hydra.cc/) for its configuration file. Because of this, you need to add the hydra decorator on top of your `main` function.
Then, you must instantiate the `Caller` class from *Lamorel* which will create the object allowing you to interact with the LLM servers.
Do not forget to initialize *Lamorel* once imported with `lamorel_init()` to initialize the communication with the servers. 
```python
import hydra
from lamorel import Caller

@hydra.main(config_path='../config', config_name='config')
def main(config_args):
    lm_server = Caller(config_args.lamorel_args)
    # Do whatever you want with your LLM
    lm_server.close()
if __name__ == '__main__':
    main()
```
Do not forget to close the connection with servers at the end of your script.

### Using the Caller
Once instantiated, you can use the different methods of the `Caller` object to send requests to your LLMs.

#### Scoring
First, we provide the `score` method to compute the probability (or log probability) of a sequence of tokens (a `candidate`) given a prompt (`context`).
*Lamorel* allows to provide multiple candidates for a single context but also to batch this computation for multiple contexts (along with their associated candidates). Using this, one can use a classic vectorized RL setup where at each step, multiple environments running in parallel return their current state and expect an action. 
```python
lm_server.score(contexts=["This is an examples prompt, continue it with"], 
                candidates=[["a sentence", "another sentence"]],
                return_logprobs=True)
```

#### Generation
*Lamorel* also provides a method for text generation. Similarly to the `score` method, one can give multiple prompts (`contexts`).
Our `generate` method can use any keyword argument from [Transformers API](https://huggingface.co/docs/transformers/main_classes/text_generation).
In addition of the generated texts, it also returns the probability of each generated sequence.

```python
lm_server.generate(contexts=["This is an examples prompt, continue it with"])
lm_server.generate(contexts=["This is an examples prompt, continue it with"], temperature=0.1, max_length=25)
```

#### Custom modules
While *Lamorel* provides two main uses of LLMs (i.e. scoring and generating), we also allow users to provide their own methods to perform custom operations using LLMs.
We expect these custom methods to be [PyTorch modules](https://pytorch.org/docs/stable/generated/torch.nn.Module.html).
![Additional modules](docs/images/additional_modules.gif)

In order to define such a custom operation, users must extend our `BaseModuleFunction` class.
For this, you must extend two main methods:
- `initialize(self)`: initialize your custom operations here.
- `forward(self, forward_outputs, minibatch, tokenized_context, **kwargs)`: perform your operations here and return the results.

*Lamorel* will give your custom module to all LLM servers and ensure your variables (e.g. weights) are the same on each server. 
See the example below where we implement a Multi Layer Perceptron (MLP) on top of our LLM.

```python
from lamorel import BaseModuleFunction

class TwoLayersMLPModuleFn(BaseModuleFunction):
    def __init__(self):
        super().__init__()

    def initialize(self):
        '''
        Use this method to initialize your module operations.
        - self.llm_config gives access to configuration of the LLM your provided in lamorel's config
        - self.model_config gives the configuration used by Transformers for the LLM (e.g. useful to know the size of representations)
        - self.device gives you access to the main device (e.g. GPU) the LLM is using
        '''
        llm_hidden_size = self.model_config.to_dict()[self.model_config.attribute_map['hidden_size']]
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(llm_hidden_size, 128),
            torch.nn.Linear(128, 128),
            torch.nn.Linear(128, self._n_outputs),
        ).to(self.device)

    def forward(self, forward_outputs, minibatch, tokenized_context, **kwargs):
        '''
        Perform your operations here.
        - forward_outputs gives access the output of the computations performed by the LLM (e.g. representations of each layer)
        - minibatch gives access to the input data (i.e. a prompt and multiple candidates) given to the LLM
        - tokenized_context gives access to the prompt used
        '''
        # Get the last layer's representation from the token right after the prompt
        if self.llm_config.model_type == "causal": # adapt to the Transformers API differing between Encoder-Decoder and Decoder-only models
            model_head = forward_outputs['hidden_states'][0][0, len(tokenized_context["input_ids"])-1, :]
        else:
            model_head = forward_outputs['encoder_last_hidden_state'][0, len(tokenized_context["input_ids"]) - 1, :]
        
        # Give representation to our MLP
        output = self.mlp(model_head)
        return output
```

Once implemented, you can give your custom module(s) to the *Lamorel* `Caller` (along with a key). It will then be possible to use your module:
- either by calling it directly using the `custom_module_fns`
- or by using it in addition of scoring

```python
lm_server = Caller(config_args.lamorel_args,
                   custom_module_functions={
                       "main_llm": {'mlp_head': TwoLayersMLPModuleFn()}
                   })
# Direct call
lm_server.custom_module_fns(module_function_keys=['mlp_head'],
                            contexts=["This is an examples prompt, continue it with"])

# Scoring
lm_server.score(additional_module_function_keys=['mlp_head'],
                contexts=["This is an examples prompt, continue it with"], 
                candidates=["a sentence", "another sentence"])
```

#### Updaters

We have seen so far how to use an LLM (along with possibly custom modules) for inference. However, *Lamorel* also provides tools to update (e.g. train) these operations with a `BaseUpdater` class which can be extended and perform any update operation. Our class gives access to the whole computation graph `self._llm_module`.
It can for instance be used to perform operations with gradient by calling `self._llm_module(['mlp_head', '__score'], ...)` with the leaf operations wanted (i.e. scoring and/or custom modules) or to select weights to train (the LLM itself `self._llm_module._LLM_model`, custom modules `self._llm_module.module._module_functions`, the whole graph `self._llm_module`).

```python
from lamorel import BaseUpdater

class TestUpdater(BaseUpdater):
    def perform_update(self, contexts, candidates, _current_batch_ids, **kwargs):
        if not hasattr(self, 'loss_fn'):
            self.loss_fn = torch.nn.CrossEntropyLoss()
        if not hasattr(self, 'optimizer'):
            # You can train:
            # 1. Only the LLM
            # self.optimizer = torch.optim.Adam(self._llm_module._LLM_model.parameters())
            
            # 2. Only some custom modules
            self.optimizer = torch.optim.Adam(self._llm_module.module._module_functions["mlp_head"].parameters())
            
            # 3. Everything
            # self.optimizer = torch.optim.Adam(self._llm_module.parameters())
        
        # Use the computational graph with gradient
        # 1. Only the LLM's scoring module
        # output = self._llm_module(['__score'], contexts=contexts, require_grad=True)
        
        # 2. Only some custom modules
        output = self._llm_module(['mlp_head'], contexts=contexts, require_grad=True)
        
        # 3. Both
        # output = self._llm_module(['__score', 'mlp_head'], contexts=contexts, require_grad=True)
        
        # Stack outputs to batch loss computation
        stacked_output = torch.stack([_o["mlp_head"] for _o in output]).to('cpu')
        
        # Compute loss with the labels corresponding to the current batch
        loss = self.loss_fn(stacked_output, kwargs["labels"][_current_batch_ids, :])
        
        # Compute gradients and update graph
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Return anything you want using a dictionary
        return {"loss": loss}
```

Once defined, users must give their custom Updater to the Caller. Note that Updater's constructor can take keyword arguments using config files (see next section).

Whenever needed, users can call their Updater with data (i.e. contexts and candidates) along with any additional keyword argument (e.g. labels).
Because multiple LLM servers can be deployed, we also dispatch the Updater's computation. When one calls the Updater with data, contexts and candidates are dispatched over the multiple servers (where each runs the Updater). 
Because *Lamorel* does not know a priori what additional keyword arguments are, these are copied and sent to each LLM. As users may need to know how to associate these arguments to the data handled by the current server,
we provide the `_current_batch_ids` variable giving the indexes of contexts and candidates that are given to the current LLM. You can therefore filter out the data that must be used by the current process:
```python
current_process_labels = [kwargs["labels"][_i] for _i in current_batch_ids["contexts"]]
```

*Lamorel* is in charge of gathering the gradients of all servers such that `self.optimizer.step()` produces the same on each server.
```python
lm_server = Caller(config_args.lamorel_args,
                   custom_module_functions={
                     "main_llm": {
                       'mlp_head': TwoLayersMLPModuleFn()
                     }
                   },
                   custom_updater={"main_llm": TestUpdater()})

result = lm_server.update(
            contexts=["This is an examples prompt, continue it with"],
            candidates=["a sentence"],
            labels=torch.tensor([0, 1], dtype=torch.float32),
        )
losses = [r["loss"] for r in result] # one loss returned per LLM
```

![Training](docs/images/training.gif)

#### Initializers
Additionally, one may also provide an `Initializer` object applying any modification on the model (e.g. freezing some weights) before it is given to the distributed architecture that synchronizes all LLMs.

```python
from lamorel import BaseModelInitializer
class CustomInitializer(BaseModelInitializer):
    def initialize_model(self, model):
        # Do whatevever your want here
        # For instance, freeze all the LLM's weights
        llm_module = model._modules['_LLM_model']
        for param in llm_module.parameters():
                param.requires_grad = False


lm_server = Caller(config_args.lamorel_args,
                   custom_model_initializer={"main_llm": CustomInitializer()})
```

#### Peft adapters
One can use `initializers` to add LoRA adapters to their model:
```python
from lamorel import BaseModelInitializer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING

class PeftInitializer(BaseModelInitializer):
    def __init__(self, use_lora, use_4bit, r, alpha, use_cache=True):
        super().__init__()
        self._use_lora = use_lora
        self._use_4bit = use_4bit
        self._r = r
        self._alpha = alpha
        self._use_cache = use_cache

    def _print_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )
        
    def initialize_model(self, model):
        if self._use_lora:
            llm_module = model._modules['_LLM_model']
            target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[llm_module.config.model_type]

            config = LoraConfig(
                r=self._r,
                lora_alpha=self._alpha,
                target_modules=target_modules,
                lora_dropout=0,
                bias="none"
            )
            
            if not self._use_cache:
                llm_module.gradient_checkpointing_enable()  # reduce number of stored activations

            if self._use_4bit:
                llm_module = prepare_model_for_kbit_training(llm_module)

            peft_model = get_peft_model(llm_module, config)
            peft_model.config.use_cache = self._use_cache

            peft_model.add_adapter("adapter_2", config)  # add a second set of adapters
            
            # Properly set the adapters' device for multi-gpu cases
            parent_module_device = None
            for name, param in peft_model.named_modules():
                if name.split(".")[-1].startswith("lora_"):
                    if hasattr(param, "weight"):
                        param.to(parent_module_device)
                else:
                    if hasattr(param, "weight"):
                        parent_module_device = param.weight.device
                    else:
                        parent_module_device = None

            model._modules['_LLM_model'] = peft_model

        model._modules['_LLM_model'].config.use_cache = self._use_cache
        self._print_trainable_parameters(model)
        return model

    
lm_server = Caller(config_args.lamorel_args,
                   custom_model_initializer={"main_llm": PeftInitializer(
                     True,
                     True,
                     16,
                     32
                   )})
```

When multiple adapters have been added to the model, simply pass `peft_adapter="adapters_name"` to any method of the `Caller` you call to activate the adapters you want.
For instance:
```python
lm_server.generate(contexts=["This is an examples prompt, continue it with"], peft_adapter="adapter_2")
```

*Warning: If no adapters are passed, the last ones activated are used.*

#### Multiple LLMs
When multiple LLMs are used, specify the one to use whenever calling a method from the `Caller`:
```python
lm_server.generate(contexts=["This is an examples prompt, continue it with"], llm_to_call="secondary_llm")
```

#### Unsloth
*Documentation in progres...*

### Setting up the configuration
The configuration of the LLM and the client-server(s) architecture is done using a YAML configuration file following [hydra](https://hydra.cc/)'s API.
Here is what this file should contain:
```yaml
lamorel_args: # Arguments for Lamorel
  log_level: info # (debug, info, error): level of logs returned by Lamorel
  llm_configs:
    main_llm:
      handler: hf_llm # (hf_llm, unsloth): Current handlers implemented
      constructor_kwargs: # any additional argument you would like to pass to the model's constructor
      model_type: seq2seq # (seq2seq, causal): Encoder-Decoder or Decoder-Only model
      model_path: t5-small # name (if downloaded from the Hugging Face's hub) or (absolute) path to the model
      pretrained: true # (true, false): set this to false if you want to keep the LLM's architecture but re-initialize its weights
      minibatch_size: 4 # number of candidates to batch per forward, adapt this number to your GPU memory
      pad_token_id: # to force a token id used for padding
      pre_encode_inputs: false # caching method especially useful when the number of candidates to score is large 
      load_in_4bit: false # quantization
      synchronize_gpus_after_scoring: false # only useful for specific GPU optimizations
      empty_cuda_cache_after_scoring: false # only useful for specific GPU optimizations
  distributed_setup_args:
    backend: gloo  # nccl can also be used but lamorel is not optimized for it (it still uses cpu operations)
    init_timeout: 120 # timeout for the torch distributed rdvz (i.e. time for all process connect with each other)
    timeout: 1800 # timeout for any lamorel call
    multinode: false # set this to true if you are using more than one node
    multinode_args: # if multinode=true, set the following arguments and launch lamorel's launcher on each node
      main_process_ip:
      main_process_port:
      experiment_id: # arbitrary id you want to give (required by torch distributed)
    n_rl_processes: 1
    llm_processes:
      main_llm:
        n_processes: 1 # number of instances in parallel
        devices_per_process: [[0]]  # either "cpu" or list of gpu ids per instance
        ddp_kwargs: # DistributedDataParallel arguments
rl_script_args: # Arguments for Lamorel
  path: ??? # absolute path to your RL script 
  # Provide any additional arguments for your RL script here
```
Examples of configuration files are provided in section below.

### Launch
*Lamorel* comes with a launcher that handles how to launch multiple processes on each machine. To launch your experiment on a machine, you must use the following command:
```
python -m lamorel_launcher.launch --config-path <path> --config-name <name> rl_script_args.path=<path> lamorel_args.accelerate_args.machine_rank=<rank> <additional arguments to override config>
```
*Warning: use absolute paths*

#### Launch examples
Several examples of configurations can be found in [examples](examples).

##### Single machine and no GPU
- Config: [local_cpu_config.yaml](examples/configs/local_cpu_config.yaml)
- Launch command(s):
    - ```shell
        python -m lamorel_launcher.launch --config-path absolute/path/to/project/examples/configs --config-name local_cpu_config rl_script_args.path=absolute/path/to/project/examples/example_script.py
      ```

##### Single machine and GPU(s)
- Config: [local_gpu_config.yaml](examples/configs/local_gpu_config.yaml)
- Launch command(s):
    - ```shell
        python -m lamorel_launcher.launch --config-path absolute/path/to/project/examples/configs --config-name local_gpu_config rl_script_args.path=absolute/path/to/project/examples/example_script.py
      ```
- Set the GPUs visible:
    - We advise you to use the id `0` in the config:
      ```yaml
      main_llm:
      n_processes: 2 # number of instances in parallel
      devices_per_process: [[0],[0]]  # either "cpu" or list of gpu ids per instance
      ```
    - And to set the visible devices at the top of your entry python script:
      ```python
      import os
      visible_device = str(max(0, int(os.environ.get("RANK")) - 1))
      print(f"Setting visible devices to be: {visible_device}")
      os.environ['CUDA_VISIBLE_DEVICES'] = visible_device
      ```


##### SLURM cluster
We here provide an example with a SLURM cluster where our experiment deploys 6 LLM servers on 3 machines. As the schema below shows, the first machine hosts 2 LLM servers and the RL script (which also has access to GPUs).
The two other machines both host only 2 LLM servers.
![Multi-nodes](docs/images/multi-nodes.png)
- Config: [multi-node_slurm_cluster_config.yaml](examples/configs/multi-node_slurm_cluster_config.yaml)
- Launch command(s):
    - ```shell
        sbatch examples/slurm/job.slurm
      ```

## Technical details and contributions
*Lamorel* relies on Pytorch distributed for communications. We advise to use the GLOO backend to allow both CPU and GPU platforms.
*Lamorel* launches the RL script `n_rl_processes` + `n_llm_processes` times. Every time it reaches the `lm_server = Caller(...)` line, *Lamorel* checks whether the current process should be part of the client or the LLM servers.

If it is a client, the script returns the `Caller` object (that the user can use to send requests) and continues.

Otherwise, it creates a [`Server`](lamorel/src/lamorel/server/server.py) that loads the LLM and starts listening.
For the communication between the client and servers, we create a process group between the client and one of the servers which is considered as the master.
This master server listens to requests and dispatches calls (using the [`Dispatcher`](lamorel/src/lamorel/server/dispatcher.py)) to all the servers using another process group (only shared by LLM servers).
Each LLM server performs the asked operations on its received data and sends the results to the master LLM server. The latter gathers results and sends them back to the RL script.

*Lamorel* is still in its very early phase and we are happy to accept any external contribution to it. Please follow the [CONTRIBUTING.md](CONTRIBUTINg.md) file.
