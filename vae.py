import torch.nn as nn

from configparser import ConfigParser

from utils import conv_network, inverse_conv_network, forward

class VAE(nn.Module):
    def __init__(self, enc_size, dec_input_channels):
        """
        Creates a variational autoencoder to be used to encode input images

        enc_size : The size of the means and log standard deviation vectors for
                   the encoder
        dec_input_channels : The number of input channels for the decoder
        """
        nn.Module.__init__(self)

        # Get the window size
        self.window_size = self.read_config()

        # Get the encoder and decoder
        self.encoder = Encoder(enc_size)
        self.decoder = Decoder(dec_input_channels)
        
    def forward(self, inp):
        """
        Propogates the model for the given input

        inp : The given images to run the model on
        """
        # Get the means and log standard deviations
        means, log_stds = self.encode(inp)

        # Sample from the normal distribution
        sample = self.sample(means, log_stds)

        # Decode the distribution sample
        imgs = self.decode(sample)

        return means, log_stds, sample, imgs

    def sample(self, means, log_stds):
        """
        Samples from a normal distribution with the given means and log
        standard deviations

        means : The means of the normal distribution
        log_stds : The standard deviations of the normal distribution
        """
        # Sample from a normal distribution
        dists = torch.randn_like(means)

        # Add the mean and and multiply by e^(-2log_stds)
        return means + (log_stds / 2).exp() * dists

    def encode(self, inp):
        """
        Encodes the input images

        inp : The input images to encode
        """
        return self.encoder(inp)

    def decode(self, sample):
        """
        Decodes the normal distribution sample to an image

        sample : The normal distribution sample to decode
        """
        # Convert the sample to 4x4xdec_input_channels
        img_dist = sample.view(-1, self.dist_shape, self.dist_shape,
                               self.dist_channels).permute(0, 3, 1, 2)

        return self.decoder(img_dist)

    def calculate_loss(self, inp, means, log_stds, dec_imgs):
        """
        Calculates the loss for the input images, the vectors of means, log
        standard deviations, and the generated images
        """
        # Calculate the gaussian loss
        gauss_loss = 0.5 * (means.pow(2) + log_stds.exp() - log_stds - 1).sum(-1)

        # Calculate the MSE loss between images
        img_loss = 0.5 * (dec_imgs - inp).pow(2).sum(-1).sum(-1).sum(-1)

        return (gauss_loss + img_loss).mean()

    def read_config(self):
        """
        Reads the config file to obtain the settings for the recorder, the
        window size for the training data and the game bindings
        """
        config = ConfigParser()
    
        # Read the config file, make sure not to re-name
        config.read("config.ini")

        return config["Window Size"]

class Encoder(nn.Module):
    def __init__(self, enc_size):
        """
        Creates the encoder for the training images

        enc_size : The length of the vectors of means and standard deviations
        """
        nn.Module.__init__(self)

        # Convolution network to 2x2x256
        self.conv = conv_network(3,
                                 [32, 64, 128, 256, 256],
                                 [7, 5, 3, 3, 3],
                                 [3, 2, 2, 2, 1],
                                 [0, 0, 0, 0, 0])

        # The means and log standard deviations of the encoded image
        self.means = nn.Linear(1024, enc_size)
        self.log_stds = nn.Linear(1024, enc_size)

    def forward(self, inp):
        """
        Encodes the given image into two vectors of means and
        standard deviations

        inp : The input images to encode
        """
        # Get the resultant image of the convolutions
        conv = forward(inp, self.conv, [nn.LeakyReLU(), nn.LeakyReLU(),
                                        nn.LeakyReLU(), nn.LeakyReLU(),
                                        nn.LeakyReLU()])

        # Flatten to nx1024
        flattened = conv.view(-1, 1024)

        # Get the means and log standard deviations of the encoded image
        means = self.means(flattened)
        log_stds = self.log_stds(flattened)

        return means, log_stds

class Decoder(nn.Module):
    def __init__(self, input_channels):
        """
        Creates the decoder for the encoded training images

        input_channels : The input channels of the encoded image
        """
        nn.Module.__init__(self)

        # Inverse convolution network to 128x128x1 images
        self.deconv = inverse_conv_network(input_channels,
                                           [256, 128, 64, 32, 3],
                                           [4, 4, 4, 4, 4],
                                           [2, 2, 2, 2, 2],
                                           [1, 1, 1, 1, 1])

    def forward(self, inp):
        """
        Decodes the given distribution sample into an image

        inp : The input distribution sample to decode
        """
        # Get the image from the encoded vectors
        deconv = forward(inp, self.deconv, [nn.ReLU(), nn.ReLU(), nn.ReLU(),
                                            nn.ReLU(), nn.Sigmoid()])

        return deconv.permute(0, 2, 3, 1)
