import torch
import torch.nn as nn

from utils import LSTM, GaussianNoise

RANDOM_SEED = 999
torch.manual_seed(RANDOM_SEED)

class Model(nn.Module):
    def __init__(self, state_space, act_n, batch_size, il_weight):
        
        nn.Module.__init__(self)

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