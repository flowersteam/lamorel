## Tutorial to use unsloth models
TODO list:
- [ ] Install unsloth `pip install unsloth`
- [ ] Modify the beginning of your entrypoint:
    ```python
            import os
            import shutil
            os.environ["UNSLOTH_DISABLE_STATISTICS"] = "0"
            working_dir = os.getcwd() + "/rank_" + os.environ.get("RANK")
            if os.path.exists(working_dir):
                shutil.rmtree(working_dir)
            os.makedirs(working_dir)
            os.chdir(working_dir)
            os.environ["TMPDIR"] = working_dir
    ```
- [ ] Handle unsloth models in your initializers, for instance with LoRA:
    ```python
            if self._use_unsloth:
                print("Setting adapters for unsloth model")
                unsloth_peft_config = config.to_dict()
                del unsloth_peft_config["task_type"]
                # Init adapters #
                peft_model = FastLanguageModel.get_peft_model(
                    llm_module,
                    **unsloth_peft_config,
                    use_gradient_checkpointing="unsloth" if not self._use_cache else False
                )
            else:
                print("Setting adapters for transformers model")
                if not self._use_cache and self._gpu_type == "nvidia":
                    llm_module.gradient_checkpointing_enable()  # reduce number of stored activations

                if self._use_4bit:
                    llm_module = prepare_model_for_kbit_training(llm_module)

                peft_model = get_peft_model(llm_module, config)
                peft_model.config.use_cache = self._use_cache
    ```