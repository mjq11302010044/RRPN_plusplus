import torch
import torch.nn.functional as F


class Mish(torch.nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
        pass
    def forward(self, x):
        print("In mish...")

        return x * torch.tanh(F.softplus(x))


def mish(input):
    '''
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    See additional documentation for mish class.
    '''

    return input * torch.tanh(F.softplus(input))