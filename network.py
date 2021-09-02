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
    def __init__(
        self, in_channels, latent_channels, out_channels, pad_type, activation, norm
    ):
        super().__init__()
        # latent channels = 64
        # in_channels = 4
        # pad_type = zero
        # activation = leaky relu
        # norm = instance norm
        self.coarse = nn.Sequential(
            # encode initial layers
            GatedConv2dLayer(
                in_channels,
                latent_channels,
                7,
                1,
                3,
                pad_type=pad_type,
                activation=activation,
                norm="none",
            ),
            GatedConv2dLayer(
                latent_channels,
                latent_channels * 2,
                4,
                2,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2dLayer(
                latent_channels * 2,
                latent_channels * 4,
                4,
                3,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2dLayer(
                latent_channels * 4,
                latent_channels * 4,
                4,
                2,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            # Bottleneck layer
            GatedConv2dLayer(
                latent_channels * 4,
                latent_channels * 4,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2dLayer(
                latent_channels * 4,
                latent_channels * 4,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2dLayer(
                latent_channels * 4,
                latent_channels * 4,
                3,
                1,
                2,
                dilation=2,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2dLayer(
                latent_channels * 4,
                latent_channels * 4,
                3,
                1,
                4,
                dilation=4,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2dLayer(
                latent_channels * 4,
                latent_channels * 4,
                3,
                1,
                8,
                dilation=8,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2dLayer(
                latent_channels * 4,
                latent_channels * 4,
                3,
                1,
                16,
                dilation=16,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2dLayer(
                latent_channels * 4,
                latent_channels * 4,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2dLayer(
                latent_channels * 4,
                latent_channels * 4,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            # decoder
            TransposeConv2dLayer(
                latent_channels * 4,
                latent_channels * 2,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2dLayer(
                latent_channels * 2,
                latent_channels * 2,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            TransposeConv2dLayer(
                latent_channels * 2,
                latent_channels,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2dLayer(
                latent_channels,
                out_channels,
                7,
                1,
                3,
                pad_type=pad_type,
                activation="tanh",
                norm="norm",
            ),
        )

        self.refinement = nn.Sequential(
            # encoder
            GatedConv2dLayer(
                in_channels,
                latent_channels,
                7,
                1,
                3,
                pad_type=pad_type,
                activation=activation,
                norm="none",
            ),
            GatedConv2dLayer(
                latent_channels,
                latent_channels * 2,
                4,
                2,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2dLayer(
                latent_channels * 2,
                latent_channels * 4,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2dLayer(
                latent_channels * 4,
                latent_channels * 4,
                4,
                2,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            # Bottleneck
            GatedConv2dLayer(
                latent_channels * 4,
                latent_channels * 4,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2dLayer(
                latent_channels * 4,
                latent_channels * 4,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2dLayer(
                latent_channels * 4,
                latent_channels * 4,
                3,
                1,
                2,
                dilation=2,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2dLayer(
                latent_channels * 4,
                latent_channels * 4,
                3,
                1,
                4,
                dilation=4,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2dLayer(
                latent_channels * 4,
                latent_channels * 4,
                3,
                1,
                8,
                dilation=8,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2dLayer(
                latent_channels * 4,
                latent_channels * 4,
                3,
                1,
                16,
                dilation=16,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2dLayer(
                latent_channels * 4,
                latent_channels * 4,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2dLayer(
                latent_channels * 4,
                latent_channels * 4,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            # decoder
            TransposeConv2dLayer(
                latent_channels * 4,
                latent_channels * 2,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2dLayer(
                latent_channels * 2,
                latent_channels * 2,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            TransposeConv2dLayer(
                latent_channels * 2,
                latent_channels,
                3,
                1,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
            ),
            GatedConv2dLayer(
                latent_channels,
                out_channels,
                7,
                1,
                3,
                pad_type=pad_type,
                activation="tanh",
                norm="norm",
            ),
        )

    def forward(self, img, mask):
        first_masked_img = img * (1 - mask) + mask
        coarse_input = torch.cat(
            (first_masked_img, mask), 1
        )  # shape: batch_size, 4, H, W
        coarse_out = self.coarse(coarse_input)
        # refinement network
        refine_masked_img = img * (1 - mask) + coarse_out
        refine_input = torch.cat(
            (refine_masked_img, mask), 1
        )  # shape: batch_size, 4, H, W
        refine_out = self.refinement(refine_input)
        return coarse_out, refine_out


# Discriminator
# Input: generated image + mask or image + mask
# output: patch of size 30 * 30
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels, latent_channels, pad_type, activation, norm):
        super().__init__()
        # Down-sample the input
        self.block1 = (
            Conv2dLayer(
                in_channels,
                latent_channels,
                7,
                1,
                3,
                pad_type=pad_type,
                activation=activation,
                norm="none",
                sn=True,
            ),
        )
        self.block2 = (
            Conv2dLayer(
                latent_channels,
                latent_channels * 2,
                4,
                2,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
                sn=True,
            ),
        )
        self.block3 = (
            Conv2dLayer(
                latent_channels * 2,
                latent_channels * 4,
                4,
                2,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
                sn=True,
            ),
        )
        self.block4 = (
            Conv2dLayer(
                latent_channels * 4,
                latent_channels * 4,
                4,
                2,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
                sn=True,
            ),
        )
        self.block5 = (
            Conv2dLayer(
                latent_channels * 4,
                latent_channels * 4,
                4,
                2,
                1,
                pad_type=pad_type,
                activation=activation,
                norm=norm,
                sn=True,
            ),
        )
        self.block6 = Conv2dLayer(
            latent_channels * 4,
            1,
            4,
            2,
            1,
            pad_type=pad_type,
            activation="none",
            norm="none",
            sn=True,
        )

    def forward(self, img, mask):
        # concat the image and the mask
        x = torch.cat((img, mask), 1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        return x


# Perceptual Network
# VGG-16 conv4_3 features
class PerceptualNet(nn.Module):
    def __init__(self):
        super(PerceptualNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return x
