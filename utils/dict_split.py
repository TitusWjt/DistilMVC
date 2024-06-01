import torch
# Exact weights
def exact_weights(path):
    model_weight = torch.load(path)
    e1_del = []
    for key, _ in model_weight.items():
        if "_decoder" in key:
            e1_del.append(key)
    for key in e1_del:
        del model_weight[key]
    return model_weight

def dict_slice(adict, start, end):
    keys = adict.keys()
    dict_slice = {}
    for k in list(keys)[start:end]:
        dict_slice[k] = adict[k]
    return dict_slice

