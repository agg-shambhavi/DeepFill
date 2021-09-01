import torch
import torch.nn as nn
import torch.nn.init as init

import spectralNorm
from layers import *

# Weight Initialization
def weights_init(net, init_type="kaiming", init_gain=0.02):
    """
    Initialize network weights.
    Parameters:
    net (network)  -- network to be initialized
    init_type (str) -- initialization method: normal, xavier & orthogonal
    init_var (float) -- scaling factor
    """

    def init_func(m):
        classname = m.__class__.__name__
        print(classname)
        if hasattr(m, "weight") and classname.find("Conv") != -1:
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    f"Initialization method {init_type} is not implemented"
                )
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif classname.find("Linear") != -1:
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)

    # now apply the initialization function here
    net.apply(init_func)


# Generator
# input: masked image + mask
# output: filled image
class GatedGenerator(nn.Module):
    def __init__(self, in_channels, latent_channels, pad_type, activation, norm):
        super().__init__()
