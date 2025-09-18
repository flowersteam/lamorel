import os
import math
import argparse
import torch



from vllm_llm import VLLM


class Parallelism:
    def __init__(self, sync=False, empty_cache=False):
        self.synchronize_gpus_after_scoring = sync
        self.empty_cuda_cache_after_scoring = empty_cache
        self.model_parallelism_size = 1


class Args:
    def __init__(self, model_path, dtype, seed, minibatch_size, model_type="causal"):
        self.model_path = model_path
        self.dtype = dtype
        self.seed = seed
        self.minibatch_size = minibatch_size
        self.model_type = model_type
        self.pre_encode_inputs = False
        self.parallelism = Parallelism(sync=False, empty_cache=False)


def _register_minimal_modules(m):
    def __score(outputs_like, *_, **__):
        return outputs_like  # tensor (mb, 1)
    if hasattr(m, "register_module_functions"):
        try:
            m.register_module_functions({"__score": __score})
            return
        except NotImplementedError:
            pass
    # fallback si register_module_functions n'est pas implémentée
    setattr(m, "_module_functions", {"__score": __score})


def main():
    parser = argparse.ArgumentParser(description="Test driver for VLLM_test class")
    parser.add_argument("--model", default=os.environ.get("VLLM_TEST_MODEL", "facebook/opt-125m"))
    parser.add_argument("--dtype", default=("float16" if torch.cuda.is_available() else "float32"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minibatch-size", type=int, default=2)
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    # Build Args for your class
    a = Args(
        model_path=args.model,
        dtype=args.dtype,
        seed=args.seed,
        minibatch_size=args.minibatch_size,
        model_type="causal",
    )

    use_cpu = args.cpu or not torch.cuda.is_available()
    devices = [] if use_cpu else list(range(torch.cuda.device_count()))

    print(f"[INFO] Loading model: {a.model_path} | dtype={a.dtype} | use_cpu={use_cpu}")
    m = VLLM(a, devices=devices, use_cpu=use_cpu)

    # Register module functions (at least __score)
    import torch.nn as nn

    class ScoreFunction(nn.Module):
        def score(self, outputs_like, *args, **kwargs):
            return outputs_like



    m.register_module_functions({"_score":ScoreFunction()})

    # --------- CALL generate() ----------
    prompts = [
        "Paris is the capital of",
        "The largest ocean on Earth is the",
    ]
    print("\n[CALL] generate()")

    gens = m.generate(
        prompts,
        return_logprobs=True,
        num_return_sequences=2,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(gens)

    contexts = ["The capital of France is ", "The capital of Italy is ", "How are you ?"]
    candidates = [[" Paris", " Rabat"], [" Rome", "Europe"], ["Fine", "No thanks"]]  # test: candidat vide -> logprob ~ 0



    #--------- CALL forward() ----------
    print("\n[CALL] forward()")
    #fwd = m.forward(["__score"], contexts, candidates=candidates, minibatch_size=args.minibatch_size)
    #print (fwd)



    #----- CALL SCORE() --------
    print("\n[CALL] Score")
    scores = m.score(contexts, candidates)
    print (scores)


    #--- get_model_config
    config = m.get_model_config()
    print(config)


if __name__ == "__main__":
    main()
