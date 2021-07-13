import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator_img(nn.Module):
    """
    Implementation of a Deep Convolutional GAN Generator (for sqare images ).

    Initialisation input :
        - img_size (int) : Lenght of the square images (Ex: 32 for MNIST 32*32)
        - channels (int) : Number of channels of the input images (Ex: 1 for MNIST)
        - latent_dim (int) : Lenght of the input latent space (Ex: 100)
    """
    def __init__(self, img_size, channels, latent_dim):
        super(Generator_img, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator_img(nn.Module):
    """
    Implementation of a Deep Convolutional GAN Discriminator.

    Initialisation input:
        - channels (int) : Number of channels of the input images
        - img_size (int) : Lenght of the square images (Ex: 32 for MNIST 32*32)
    """
    def __init__(self, channels, img_size):
        super(Discriminator_img, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [
                nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


class Generator_(nn.Module):
    """
    Implementation of a simple GAN sicriminator.
    
    Initialisation input:
        - latent_dim (int): Dimention of the latent space (Ex: 100)
        - out_shape (tuple): Shape of the output targeted data (Ex: (10, 10))
    """   
    def __init__(self, latent_dim, out_shape):
        super(Generator_, self).__init__()
        self.out_shape = out_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(out_shape))),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.model(z)
        out = out.view(out.size(0), *self.out_shape)
        return out


class Discriminator_(nn.Module):
    """
    Implementation of a simple GAN sicriminator.
    
    Initialisation input:
        - inp_shape (tuple) : Tuple representing the shape of the inputs (Ex: (10, 10)
    """
    def __init__(self, inp_shape):
        super(Discriminator_, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(inp_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

class Discriminator(nn.Module):
    def __init__(self, input_dim, nb_layers=2, nb_units=256):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList() 
        self.layers += [nn.Linear(input_dim, nb_units), 
                 nn.LeakyReLU()]

        for i in range(1, nb_layers-1):
            self.layers += [nn.Linear(nb_units, nb_units),
                            nn.BatchNorm1d(1),
                            nn.Tanh()]
        self.layers += [nn.Linear(nb_units, 1)]

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        out = self.model(x)
        return out 

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, nb_layers=2, nb_units=256):
        super(Generator, self).__init__()
        self.layers = nn.ModuleList() 
        self.layers += [nn.Linear(input_dim, nb_units), 
                 nn.LeakyReLU()]

        for i in range(1, nb_layers-1):
            self.layers += [nn.Linear(nb_units, nb_units),
                            nn.LeakyReLU()]
        self.layers += [nn.Linear(nb_units, output_dim)]

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        out = self.model(x)
        return out 

class ResNet(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs
