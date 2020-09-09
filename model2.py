import torch
import torch.nn as nn
import numpy as np

from copy import deepcopy

from utils import LSTM, NoisyLinear, QuantileLayer
from optim import RAdam

RANDOM_SEED = 999
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class IQN(nn.Module):
    def __init__(self, state_space, act_n, quantile_dim, num_quantiles,
                 hidden_dim, num_hidden, optim_params):
        """
        Rainbow Recurrent IQN

        IQN: https://arxiv.org/pdf/1806.06923.pdf
        R2D2: https://openreview.net/pdf?id=r1lyTjAqYX
        R2D3: https://arxiv.org/abs/1909.01387
        """
        nn.Module.__init__(self)

        self.online = Model(state_space, act_n, quantile_dim, num_quantiles,
                            hidden_dim, num_hidden)
        self.target = deepcopy(self.online)

        self.loss_func = nn.SmoothL1Loss(reduction="mean")
        self.optim = RAdam(self.online.parameters(), **optim_params)

    def forward(self, inp):
        return self.online(inp)

    def step(self, state, greedy=False):
        """
        Takes a step into the environment
        """
        return self.online.step(state, greedy)

    def train_batch(self, rollouts, burn_in_length, sequence_length):
        """
        Trains for a batch of rollouts with the given burn in length and
        training sequence length
        """
        self.optim.zero_grad()

        states, actions, rewards, next_states, terminals, hidden_state = rollouts

        # Add burn in here #######
        
        next_q_vals, next_quantile_vals, next_quantiles, next_hidden = self.target(next_states)
        num_quantiles = next_quantile_vals[1]

        next_actions = next_quantile_vals.argmax(-1, keepdim=1)
        next_actions = next_actions.unsqueeze(1).repeat(1, num_quantiles, 1)
        next_values = next_quantile_vals.gather(-1, next_actions).squeeze(1)

        q_vals, quantile_vals, quantiles = self.online(states)
        action_values = quantile_vals.gather(-1, actions)

        td_error = next_values.unsqueeze(2) - action_values.unsqueeze(1)
        quantile_loss = self.loss_func(next_values.unsqueeze(2),
                                       action_values.unsqueeze(1))

        quantiles = quantiles.unsqueeze(1).repeat(1, self.num_quantiles, 1)
        penalty = torch.abs(quantiles - (td_error < 0).float().detach())

        loss = penalty * quantile_loss # Divide by huber kappa
        loss = loss.sum(2).mean(1)
        meaned_loss = loss.mean(1)

        meaned_loss.backward()
        self.optim.step()

        return meaned_loss, loss

    def train(self, num_batches, batch_size, burn_in_length, sequence_length,
              online_replay_buffer=None, supervised_replay_buffer=None,
              supervised_chance=0.25, writer=None):
        """
        Trains R2D3 style with 2 replay buffers
        """
        assert not online_replay_buffer == supervised_replay_buffer == None

        for batch in range(1, num_batches + 1):
            buff_choice = np.rand()
            if(online_replay_buffer is None or buff_choice < supervised_chance):
                replay_buffer = supervised_replay_buffer
            else:
                replay_buffer = online_replay_buffer

            while(not replay_buffer.ready_to_sample(batch_size)):
                pass

            rollouts, idxs, is_weights = replay_buffer.sample(batch_size)

            loss, new_errors = self.train_batch(rollouts, burn_in_length,
                                                sequence_length)
            replay_buffer.update_priorities(new_errors, idxs)

            if(writer is not None):
                if(buff_choice < supervised_chance):
                    writer.add_summary("Supervised Loss", loss, batch)
                else:
                    writer.add_summary("Online Loss", loss, batch)

                writer.add_summary("Loss", loss, batch)

    def update_target(self):
        """
        Updates the target network
        """
        self.target.load_state_dict(self.online.state_dict())

class Model(nn.Module):
    """
    Recurrent IQN with R2D3 features
    """
    def __init__(self, state_space, act_n, quantile_dim, num_quantiles,
                 hidden_dim, num_hidden): # NUM HIDDEN NOT IN USE CURRENTLY
        nn.Module.__init__(self)

        self.state_space = state_space
        self.act_n = act_n
        self.quantile_dim = quantile_dim
        self.num_quantiles = num_quantiles
        self.hidden_dim = hidden_dim

        self.conv = ConvNet(state_space[-1], 32, lambda x: 2 * x)

        self.lstm = nn.LSTM(256, quantile_dim)

        # If input size is 256 after CNN
        self.quantile_layer = QuantileLayer(quantile_dim, hidden_dim)

        self.advantage = nn.Seqential(
            NoisyLinear(hidden_dim, hidden_dim, 0, 0.05),
            nn.ReLU(-1),
            NoisyLinear(hidden_dim, act_n, 0, 0.05)
        )

        self.advantage = nn.Seqential(
            NoisyLinear(hidden_dim, hidden_dim, 0, 0.05),
            nn.ReLU(-1),
            NoisyLinear(hidden_dim, 1, 0, 0.05)
        )

    def step(self, state, hidden_state=None, greedy=False):
        q_vals, quantile_values, quantiles, hidden_state = self(state,
                                                                hidden_state)

        if(greedy):
            action = q_vals.max(1)
        else:
            action = nn.Categorical(logits = q_vals)
            action = action.sample()

        action = action.item()

        return action, hidden_state

    def forward(self, state, hidden_state=None):
        """
        Returns the quantile values and quantiles for the given state
        """
        if(hidden_state is None):
            hidden_state = tuple([torch.zeros(1, len(state), self.hidden_dim)
                                  for i in range(2)])

        cnn = cnn.view(len(state), -1)
        lstm, hidden_state = self.LSTM(state, hidden_state)

        quantile_values, quantiles = self.quantile_layer(lstm,
                                                         self.num_quantiles)

        advantage = self.advantage(quantile_values)
        value = self.value(quantile_values)

        q_vals = value + advantage - advantage.mean(-1, keepdim=True).mean(1)

        return q_vals, quantile_values, quantiles, hidden_state

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

class ConvNet(nn.Module):
    def __init__(self, inp_channels, init_channels, growth_func):
        nn.Module.__init__(self)

        num_layers = 4
        hidden_channels = [init_channels]
        for i in range(num_layers - 1):
            hidden_channels.append(growth_func(hidden_channels[-1]))

        self.convs = nn.Sequential(
            self._conv_layer(inp_channels, hidden_channels[0], 5, 2, 0),
            self._conv_layer(inp_channels, hidden_channels[0], 3, 2, 0),
            self._conv_layer(inp_channels, hidden_channels[0], 3, 2, 0),
            self._conv_layer(inp_channels, hidden_channels[0], 3, 1, 0),
        )

    def _conv_layer(self, in_channels, out_channels, kernel_size, stride,
                    padding):
        return nn.Sequential(
            nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(-1)
        )

    def forward(self, inp):
        return self.convs(inp)