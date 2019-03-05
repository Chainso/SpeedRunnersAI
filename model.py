import torch
import torch.nn as nn

from vae import VAE
from distributions import MultiCategoricalDistribution, TanhGaussianMixtureModel
from utils import LSTM

class Model(nn.Module):
    def __init__(self):
        """
        Creates the model to be used by the bot
        """
        nn.Module.__init__(self)

        # Set the encoded size and get the decoder input channels and dist shape
        self.enc_size = 128
        self.dist_shape = 4
        self.dec_input_channels = 128 // (self.dist_shape ** 2)

        # Get the variational autoencoder
        self.vae = VAE(self.enc_size, self.dist_shape, self.dec_input_channels)

        # The LSTM
        self.lstm = LSTM(1, self.enc_size, self.enc_size, 3, dropout = 0.3)

        # Get the multi-categorical distribution
        #self.action = MultiCategoricalDistribution(self.enc_size, 6)
        self.action = TanhGaussianMixtureModel(self.enc_size, 6)

        # The optimizer for the model
        self.optim = torch.optim.Adam(self.parameters(), lr = 1e-3)

    def forward(self, inp):
        """
        Returns the action vector for the given input images

        inp : The input images to get the actions for
        """
        # Get the encoded input sample and resize to input to LSTM
        enc_sample = self.vae.sample(*self.vae.encode(inp))
        enc_sample = enc_sample.view(enc_sample.size()[0], 1,
                                     enc_sample.size()[-1])

        # Get the LSTM output and reshape for action distribution input
        lstm = self.lstm(enc_sample)
        lstm = lstm.view(lstm.size()[0], lstm.size()[-1])

        # Get the actions
        #action = self.action.distribution(lstm)
        action = self.action.mixture(lstm)

        # Round the values
        return action

    def step(self, inp):
        """
        Returns the action vector for a single input image

        inp : The input images to get the actions for
        """
        return self(inp)[0].detach().numpy()
        #return self(inp).detach().numpy()[0]

    def calculate_loss(self, inp, actions):
        """
        Calculates the loss for the given input image and action

        inp : The input images
        action : The target actions
        """
        # Get the means, log standard deviations, distribution sample and
        # generated images
        means, log_stds, sample, gen_imgs = self.vae(inp)

        # Add the next dimension for the LSTM
        sample = sample.view(sample.size()[0], 1, sample.size()[-1])

        # Get the lstm output for the input sample
        lstm = self.lstm(sample.detach())
        lstm = lstm.view(lstm.size()[0], lstm.size()[-1])

        # Get the VAE loss
        vae_loss = self.vae.calculate_loss(inp, means, log_stds, gen_imgs)

        # Get the multi-categorical distributions loss
        action_loss = self.action.calculate_loss(lstm, actions)

        return vae_loss + action_loss

    def train(self, inp, actions):
        """
        Trains the network for the batch of inputs and actions

        inp : The input images
        actions : The target actions
        """
        # Get the loss
        loss = self.calculate_loss(inp, actions)

        # Proprogate the loss backwards
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # Return a numpy array of the losses
        return loss.detach().numpy()

    def reset_hidden_state(self):
        """
        Resets the hidden state for the LSTM
        """
        self.lstm.reset_hidden()

    def save(self, save_path):
        """
        Saves the model at the given save path

        save_path : The path to save the model at
        """
        torch.save(self.state_dict(), save_path)

    def load(self, load_path):
        """
        Loads the model at the given load path

        load_path : The of the model to load
        """
        self.load_state_dict(torch.load(load_path))
