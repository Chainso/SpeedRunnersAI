import torch
import torch.nn as nn
import numpy as np

from utils import LSTM, GaussianNoise

RANDOM_SEED = 999
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class Model(nn.Module):
    """
    Recurrent IQN with R2D3 features
    """
    def __init__(self, state_space, act_n, quantile_dim, hidden_dim,
                 num_hidden):
        nn.Module.__init__(self)

        self.state_space = state_space
        self.act_n = act_n
        self.quantile_dim = quantile_dim

        # If input size is 256 after CNN
        self.quantile_fc = nn.Sequential(
            nn.Linear(quantile_dim, 256),
            nn.ReLU()
        )

        fcs = [self._fc_layer(256, hidden_dim)]
        fcs += [self.fc_layer(hidden_dim, hidden_dim)
                for i in range(num_hidden - 1)]

        self.fc = nn.Sequential(
            *fcs,
            nn.Linear(hidden_dim, act_n),
            nn.Softmax(-1)
        )

    def _fc_layer(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, state, num_quantile_samples):
        """
        Returns the quantile values and quantiles for the given state
        """
        quantiles = torch.rand(num_quantile_samples * len(state),
                               device = state.device)
        quantile_dist = quantiles.repeat(1, self.quantile_dim)

        qrange = torch.range(0, self.quantile_dim, device = state.device)

        quantile_dist = qrange * np.pi * quantile_dist
        quantile_dist = torch.cos(quantile_dist)
        quantile_dist = self.quantile_fc(quantile_dist)

        rep_state = state.repeat(num_quantile_samples, 1)
        quantile_dist = rep_state * quantile_dist

        quantile_values = self.fc(quantile_dist)
        quantile_values = quantile_values.view(num_quantile_samples,
                                               len(state), self.act_n)

        q_vals = quantile_values.mean(0)

        return quantile_values, quantiles, q_vals

    def reset_hidden_state(self):
        """
        Resets the hidden state for the LSTM
        """
        self.lstm.reset_hidden()

    def train_batch(rollouts):
        states, actions, rewards, next_states, terminals = rollouts
        rewards = rewards.repeat

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