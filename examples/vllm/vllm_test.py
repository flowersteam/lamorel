from lamorel import Caller, lamorel_init
lamorel_init()
import torch.nn.functional as F
import torch
import time
import json
import hydra


@hydra.main(config_path='config', config_name='config')
def main(config_args):
    

    lm_server = Caller(config_args.lamorel_args)
    print("[DEBUG] lm_server Ready (Caller Instace ready) !" )
    


    prompt =  "Hello, what's the capital of france ?"
    
    responses = lm_server.generate(contexts=[prompt], max_new_tokens=50, num_return_sequences= 1,return_logprobs=True)

    print (responses)

    contexts = ["The capital of France is ", "The capital of Italy is "]
    candidates = [["Paris", "Rabat"], ["Rome"]]  # test: candidat vide -> logprob ~ 0


    responses2 = lm_server.score(contexts, candidates,return_logprobs=True)


    print(responses2)   
    lm_server.close()

if __name__ == '__main__':
    main()
