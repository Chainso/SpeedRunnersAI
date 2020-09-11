import torch
import numpy as np

from threading import Thread
from configparser import ConfigParser
from pykeyboard import PyKeyboardEvent
from time import time

from hdf5_handler import HDF5Handler
from agent import Agent
from utils import discount, normalize


class ModelTrainer(PyKeyboardEvent):
    """
    A keyboard event listener to start and stop the recording.
    """
    def __init__(self, model, replay_buffer, episodes, batch_size,
                 burn_in_length, sequence_length, decay, save_interval,
                 save_path):
        """
        Will play the game using the given model.

        Args:
            model: The model to train.
            replay_buffer: The replay buffer to store experiences in.
            episodes: The number episodes to train for.
            batch_size: The batch size of the training inputs.
            burn_in_length: The length of the burn in sequence of each batch.
            sequence_lengh: The length of the training sequence of each batch.
            decay: The decay of the n_step return.
            save_interal: The number of episodes between each save
            save_path: The path to save the model to
        """
        PyKeyboardEvent.__init__(self)

        # The model to use
        self.model = model
        self.data_handler = data_handler

        self.episodes = episodes
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.decay = decay
        self.save_interval = save_interval
        self.save_path = save_path

        # If the program is currently running
        self.playing = False

        # If the program is listening to the keyboard
        self.listening = False

        # Get the information from the config
        playing_config = self.read_config()

        # Get the start, end and close key and the save interval from the
        # config
        self.start_key = playing_config["START_KEY"].lower()
        self.end_key = playing_config["END_KEY"].lower()
        self.close_key = playing_config["CLOSE_KEY"].lower()

        self.agent = Agent(model)

    def _loop_listening(self):
        """
        Ensures that the program will continue listening until closure
        """
        episode = 1
        collection_length = self.batch_size * self.sequence_length

        while(self.listening):
            def finish_func(episode):
                if(episode >= self.episodes):
                    self.stop()
                    return True
                else:
                    return self.currently_playing()

            # Let the agent train
            agent.train(self.replay_buffer, self.episodes, self.burn_in_length,
                        self.sequence_length, self.decay, self.n_steps,
                        self.save_interval, self.save_path,
                        finish_func)

    def tap(self, keycode, character, press):
        """
        Will handle the key press event to start, stop and end the program.
        """
        character = character.lower()

        if(press):
            # Start recording on the recording key press
            if(character == self.start_key and not self.playing):
                self.resume_playing()
            # Stop recording on the recording key press
            elif(character == self.end_key and self.playing):
                self.pause_playing()
            # Close the program on the close key press
            elif(character == self.close_key):
                self.close_program()

    def resume_playing(self):
        """
        Will cause the program to start recording.
        """
        if(not self.playing):
            print("Playing resumed")
            self.playing = True
            self.sr_game.start()

            if(not self.listening):
                self.listening = True

                loop_listening = Thread(target = self._loop_listening)
                loop_listening.start()

    def pause_playing(self):
        """
        Will cause the program to stop recording
        """
        if(self.playing):
            print("Playing paused")
            self.playing = False
            self.sr_game.pause()

    def close_program(self):
        """
        Will close the program.
        """
        if(self.listening):
            print("Finished playing")
            self.playing = False
            self.listening = False
            self.sr_game.stop()
            self.data_handler.close()

        self.stop()

    def currently_playing(self):
        """
        Returns true if there is an agent actively playing
        """
        return self.playing

    def read_config(self):
        """
        Reads the config file to obtain the settings for the recorder, the
        window size for the training data and the game bindings
        """
        config = ConfigParser()
    
        # Read the config file, make sure not to re-name
        config.read("./config/config.ini")

        return config["AI"]

def train_model(model, data_handler, epochs, batch_size, save_path, cuda):
    model = model.train()

    for epoch in range(1, epochs + 1):
        # Reset the hidden state
        model.reset_hidden_state()

        # The average loss of this epoch
        avg_loss = 0

        for index in range(0, len(data_handler), batch_size):
            # Get the next batch of states and actions
            states = data_handler.get_states(index, index + batch_size, cuda)
            actions = data_handler.get_actions(index, index + batch_size, cuda)

            # Compute the losses
            avg_loss += (model.train_supervised(states, actions)
                         * (batch_size /len(data_handler)))

        # Print the average loss
        print("Epoch ", epoch, ": loss", avg_loss)

        # Save the model
        model.save(save_path + "-" + str(epoch) + ".torch")

def model_config():
    """
    Reads the config file to obtain the settings for the recorder, the
    window size for the training data and the game bindings
    """
    config = ConfigParser()

    # Read the config file, make sure not to re-name
    config.read("./config/config.ini")

    return config["Environment"], config["AI"]

if(__name__ == "__main__"):
    from torch.multiprocessing import Process

    from speedrunnersai.speedrunners.actor import Actor
    from model import IQN
    from per import PERMemory
    # from agent import DQNAgent # Add the agent

    cuda = torch.cuda.is_avaliable()
    device = "cuda" if cuda else "cpu"

    window_size, training_conf = model_config()

    state_space =  (int(window_size["WIDTH"]),
                    int(window_size["HEIGHT"]),
                    int(window_size["DEPTH"]))

    act_n = Actor.NUM_ACTIONS
    quantile_dim = 128
    num_quantiles = 32
    hidden_dim = 128
    num_hidden = 1

    model_args = (state_space, act_n, quantile_dim, num_quantiles, hidden_dim,
                  num_hidden)
    model = IQN(*model_args).to(torch.device(device))

    model_path = "./Trained Models/"
    save_path = model_path + training_conf["SAVE_PATH"]
    load_path = model_path + training_conf["LOAD_PATH"]

    #model.load(load_path)

    capacity = 100000
    alpha = 0.7
    beta = 0.4
    beta_increment = 1e-4
    epsilon = 1e-3
    per_params = (capacity, alpha, beta, beta_increment, epsilon)

    online_replay_buffer = PERMemory(*per_params)

    supervised_replay_buffer = PERMemory(*per_params)

    data_handler = HDF5Handler("r+", 1)
    print("Dataset size:", len(data_handler))
    # ADD EXPERIENCES TO SUPERVISED REPLAY BUFFER HERE
    

    num_batches = 100000
    batch_size = 32
    burn_in_length = 40
    sequence_length = 40
    supervised_chance = 0.25
    model_training_params = (num_batches, batch_size, burn_in_length,
                             sequence_length, online_replay_buffer,
                             supervised_replay_buffer, supervised_chance)
    model_training_proc = Process(target = model.train,
                                  args=model_training_params)

    episodes = 100
    decay = 0.99
    n_steps = 5
    save_interval = 10
    agent_params = (episodes, decay, n_steps, save_interval)

    trainer = ModelTrainer(model, online_replay_buffer, episodes, batch_size,
                           burn_in_length, sequence_length, decay,
                           save_interval, save_path)

    #model_training_proc.start()
    #trainer.run()
