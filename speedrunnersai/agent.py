import numpy as np

from utils import discount
from speedrunnersai.speedrunners.sr_env import SpeedRunnersEnv

class Agent():
    def __init__(self, model, env):
        self.model = model
        self.env = env

        self.episode = 1
        self.hidden_state = None

    def step(self, state, greedy=False):
        """
        Takes one step in the environment
        """
        action, hidden_state = self.model.step(state, self.hidden_state, greedy)
        return action, hidden_state, *self.env.step(action)

    def train(self, replay_buffer, episodes, burn_in_length, sequence_length,
              decay, n_steps, save_interval, save_path, finish_func=None):
        """
        Adds replays to the buffer using the model for the number of episodes
        given
        """
        finish = False
        self.hidden_state is None

        if(finish_func is not None and finish is False):
            finish = finish_func(self.episode)

        while(not finish and self.episode < episodes + 1):
            terminal = False

            states = []
            actions = []
            rewards = []
            terminals = []
            next_states = []
            hidden_states = []

            while(not terminal and not finish):
                state = self.env.state
                action, new_hidden, next_state, reward, terminal, info = self.step(state, False)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                terminals.append(terminal)
                next_states.append(next_states)
                hidden_states.append(new_hidden)

                self.hidden_state = new_hidden
                if(finish_func is not None and finish is False):
                    finish = finish_func(self.episode)

                if(terminal or (len(states) == (burn_in_length +
                                                sequence_length)) + n_steps):
                    # Trying a new way of discounting
                    disc_rewards = []
                    for i in range(len(states) - n_steps):
                        disc_rewards[i] = discount(rewards[i:i + 1 + n_steps],
                                                   decay)[0]
                        experiences = zip(np.stack(states[burn_in_length + sequence_length]),
                                          np.stack(actions[burn_in_length + sequence_length]),
                                          np.stack(disc_rewards),
                                          np.stack(terminals[burn_in_length + sequence_length]),
                                          np.stack(next_states[burn_in_length + sequence_length]),
                                          np.stack(hidden_states[burn_in_length + sequence_length])
                                         )

                        # Using starting error of 1 for now
                        errors = [1] * (burn_in_length + sequence_length)
                        replay_buffer.add_batch(experiences, errors)

                        if(not terminal):
                            states = states[-burn_in_length:]
                            actions = actions[-burn_in_length]
                            rewards = rewards[-burn_in_length:]
                            terminals = terminals[-burn_in_length:]
                            next_states = next_states[-burn_in_length:]
                            hidden_states = hidden_states[-burn_in_length]

            if(self.episode % save_interval == 0):
                self.model.save(save_path)

            self.episode += 1

    def play(self, episodes, finish_func=None):
        """
        Lets the agent play the environment using the model for the number of
        episodes given
        """
        finish = False
        self.hidden_state = None

        if(finish_func is not None and finish is False):
            finish = finish_func(self.episode)

        while(not finish and self.episode < episodes + 1):
            terminal = False

            while(not terminal):
                state = self.env.state
                _, _, _, reward, terminal, _ = self.step(state, True)

                if(finish is False and finish_func is not None):
                    finish = finish_func(self.episode)

            self.episode += 1
