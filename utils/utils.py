import math
from torch import nn

def get_activation(activation_type):
    if activation_type == "relu":
        return nn.ReLU()
    elif activation_type == "relu6":
        return nn.ReLU6()
    elif activation_type == "prelu":
        return nn.PReLU()
    elif activation_type == "selu":
        return nn.SELU()
    elif activation_type == "celu":
        return nn.CELU()
    elif activation_type == "gelu":
        return nn.GELU()
    elif activation_type == "sigmoid":
        return nn.Sigmoid()
    elif activation_type == "softplus":
        return nn.Softplus()
    elif activation_type == "softshrink":
        return nn.Softshrink()
    elif activation_type == "softsign":
        return nn.Softsign()
    elif activation_type == "tanh":
        return nn.Tanh()
    elif activation_type == "tanhshrink":
        return nn.Tanhshrink()
    else:
        raise ValueError("Unknown activation type {}".format(activation_type))

def initialize_model(model):
    for name, param in model.named_parameters():
        if 'norm' in name:
            continue
        if name.endswith('bias'):
            param.data.fill_(0)
        else:
            nn.init.kaiming_normal_(param.data)