import torch
from torch.utils.data import DataLoader
from network import *
import dataset


# train function
def trainFn():
    pass


# main function
def main():

    # All constants

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CONSTANTS for generator
    IN_CHANNELS = 3
    LATENT_CHANNELS = 64
    OUT_CHANNELS = 3
    PAD_TYPE = "zero"
    NORM = "in"
    ACTIVATION = "lrelu"

    # Learning rate constants
    LR_G = 1e-4
    LR_D = 4e-4
    BETA1 = 0.5
    BETA2 = 0.999

    # All necessary paths
    ROOTDIR_PATH = r"G:\DeepFill\Data\data256x256"

    # Training constants
    BATCH_SIZE = 32
    NUM_EPOCHS = 1

    # Instantiate all networks

    #  CREATE GENERATOR -> make gen object and initialize weights
    gen = GatedGenerator(
        in_channels=IN_CHANNELS,
        latent_channels=LATENT_CHANNELS,
        out_channels=OUT_CHANNELS,
        pad_type=PAD_TYPE,
        activation=ACTIVATION,
        norm=NORM,
    ).to(DEVICE)

    weights_init(net=gen)

    #  CREATE DISCRIMINATOR -> make disc object and initialize weights
    disc = PatchDiscriminator(
        in_channels=IN_CHANNELS,
        latent_channels=LATENT_CHANNELS,
        pad_type=PAD_TYPE,
        activation=ACTIVATION,
        norm=NORM,
    ).to(DEVICE)

    weights_init(net=disc)

    # Instantiate Perceptual net

    def load_dict(process_net, pretrained_net):
        """
        Function to load pretrained network's state dict to our current network
        """
        # Get the dict from pretrained net
        # idk if state_dict()will be there or not
        pretrained_dict = pretrained_net.state_dict()
        # Get the dict from process_net
        process_dict = process_net.state_dict()
        # Delete the extra keys from pretrained_dict that do not belong to process_dict
        pretrained_dict = {k: v for k, v in pretrained_dict if k in process_dict}
        # Update process dict using pretrained_dict
        process_dict.update(pretrained_dict)
        # Load the updated dict to processing network
        process_net.load_state_dict(process_dict)
        return process_net

    perceptNet = PerceptualNet().to(DEVICE)
    vgg16 = torch.load("./Data/vgg16_pretrained.pth")
    load_dict(perceptNet, vgg16)
    for param in perceptNet.parameters():
        param.requires_grad = False

    # Optimizers
    opt_g = torch.optim.Adam(gen.parameters(), lr=LR_G, betas=(BETA1, BETA2))
    opt_d = torch.optim.Adam(disc.parameters(), lr=LR_D, betas=(BETA1, BETA2))

    # Learning rate scheduler
    def adjust_lr(lr_in, optimizer, epoch, decrease_factor, lr_decrease_epoch):
        """
        Set the lr to (decreased_factor * lr) to every lr_decrease_epoch
        """
        lr = lr_in * (decrease_factor ** (epoch // lr_decrease_epoch))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    # Initialize training data
    train_dataset = dataset.CelebA(ROOTDIR_PATH)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
