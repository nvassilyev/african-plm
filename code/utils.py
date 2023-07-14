from transformers import XLMRobertaTokenizer, AutoModelForMaskedLM, AutoTokenizer
from adapter import XmodAdapter

def freeze_shared_layers(model):
    # freeze everything
    for parameter in model.parameters():
        parameter.requires_grad = False

    # unfreeze embeddings and adapters    
    for parameter in model.roberta.embeddings.parameters():
        parameter.requires_grad = True
    for layer in model.roberta.encoder.layer:
        if layer.output.adapter_layer_norm is not None:
            for parameter in layer.output.adapter_layer_norm.parameters():
                parameter.requires_grad = True
        for parameter in layer.output.adapter_modules.parameters():
            parameter.requires_grad = True


def add_adapter(model, init_lang):
    pass


def extend_vocab(model, tokenizer, old_vocab_path = None):
    pass