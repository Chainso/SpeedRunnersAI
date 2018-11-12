import torch
import torch.nn as nn

from configparser import ConfigParser

from utils import linear_network, conv_network, forward

class Model(nn.Module):
    def __init__(self):
        """
        Creates the model used for the bot
        """
        nn.Module.__init__(self)

        self.window_size = self.read_config()

        self.encoder = Encoder((self.window_size["WIDTH"],
                               self.window_size["HEIGHT"]), 100)

    def forward(self, inp):
        """
        Propogates the model for the given input

        inp : The given images to run the model on
        """
        pass

    def get_action(self, inp):
        """
        Returns the action vector for the given input images

        inp : The input images to get the actions for
        """
        pass

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
    def __init__(self, image_shape, enc_size):
        """
        Creates the encoder for the training images

        image_shape : The shape of the input images
        enc_size : The length of the vectors of means and standard deviations
        """
        nn.Module.__init__(self)

    def forward(self, inp):
        """
        Encodes the given image into two vectors of means and
        standard deviations

        inp : The input images to encode
        """
        pass

class Decoder(nn.Module):
    def __init__(self, image_shape, enc_size):
        """
        Creates the decoder for the encoded training images

        image_shape : The shape of the input images
        enc_size : The length of the vectors of means and standard deviations
        """
        nn.Module.__init__(self)

    def forward(self, inp):
        """
        Decodes the given distribution sample into an image

        inp : The input distribution sample to decode
        """
        pass
