from torch.nn import *


def init_weights(layer):
    if type(layer) == Linear:
        init.normal_(layer.weight, mean=0, std=0.01)
        layer.bias.data.fill_(0.0)


def gen_model(in_sz, hidden_sz, out_sz):
    model = Sequential(Flatten(), Linear(in_sz, hidden_sz), Sigmoid(), Linear(hidden_sz, out_sz), LogSoftmax(dim=0))
    model.apply(init_weights)
    return model
