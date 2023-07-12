from enum import Enum

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig

class ModelTypesEnum(Enum):
    causal = AutoModelForCausalLM
    seq2seq = AutoModelForSeq2SeqLM


def load_hf_model_and_tokenizer(type, path, pretrained):
    print("Loading model {}".format(path))
    tokenizer = AutoTokenizer.from_pretrained(path)

    # Select class according to type
    config = AutoConfig.from_pretrained(path)

    n_layers_key = 'num_hidden_layers'
    if hasattr(config, "attribute_map") and n_layers_key in config.attribute_map:
        n_layers_key = config.attribute_map[n_layers_key]

    n_layers = getattr(config, n_layers_key)
    model_class = ModelTypesEnum[type].value
    if pretrained:
        model_method = lambda **kwargs: model_class.from_pretrained(path, **kwargs)
    else:
        model_method = lambda **kwargs: model_class.from_config(config, **kwargs)

    return tokenizer, model_method, n_layers
