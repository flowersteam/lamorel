## Tutorial to migrate from V0.2 to V0.3
TODO list:
- [ ] Update `lamorel_args` in config
- [ ] Update `Caller` instantiation (everything is now a dictionary with llms as keys)
- [ ] Use `self.llm_config` or `self.model_config` in custom module heads
- [ ] Replace `_current_batch_ids` by `_current_batch_ids["contexts"]` in updaters
- [ ] Remove the instantiation of an Accelerator (from accelerate) and replace all `accelerator.process_index` by `os.environ["RANK"]`
- [ ] Use the new [slurm launcher](../examples/slurm/launcher.sh) if running on slurm clusters
- [ ] When deploying multiple instances of an LLM on the same machine with multiple GPUs, fix the visible devices at the beginning of your entry point:
  ```python
    import os
    visible_device = str(max(0, int(os.environ.get("RANK")) - 1))
    print(f"Setting visible devices to be: {visible_device}")
    os.environ['CUDA_VISIBLE_DEVICES'] = visible_device
  ```