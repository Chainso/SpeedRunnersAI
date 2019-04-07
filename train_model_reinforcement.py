import torch

from threading import Thread
from configparser import ConfigParser
from pykeyboard import PyKeyboardEvent

from model2 import Model2
from hdf5_handler import HDF5Handler
from speedrunners import SpeedRunnersEnv
from utils import discount, normalize


class ModelTrainer(PyKeyboardEvent):
    """
    A keyboard event listener to start and stop the recording.
    """
    def __init__(self, model, data_handler, episodes, batch_size,
                 sequence_length, decay, save_interval, save_path):
        """
        Will play the game using the given model.

        Args:
            model: The model to train.
            data_handler: The data handler for supervised training
            episodes: The number episodes to train for.
            batch_size: The batch size of the training inputs.
            sequence_lengh: The length of each batch
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
        window_size, playing_config, self.speedrunners = self.read_config()

        # Get the start, end and close key and the save interval from the
        # config
        self.start_key = playing_config["START_KEY"].lower()
        self.end_key = playing_config["END_KEY"].lower()
        self.close_key = playing_config["CLOSE_KEY"].lower()

        # Get the game screen size
        game_screen = (int(self.speedrunners["WIDTH"]),
                       int(self.speedrunners["HEIGHT"]))

        # Get the resized screen size from the config
        res_screen_size = (int(window_size["WIDTH"]),
                           int(window_size["HEIGHT"]),
                           int(window_size["DEPTH"]))

        self.sr_game = SpeedRunnersEnv(120, game_screen, res_screen_size)

    def _loop_listening(self):
        """
        Ensures that the program will continue listening until closure
        """
        while(self.listening):
            # Record the keys and game frames while recording is enabled
            while(self.playing):
                episode = 1
                while(episode <= self.episodes):
                    states = []
                    actions = []
                    rewards = []
                    values = []
    
                    state = self.sr_game.reset()
                    terminal = False
    
                    i = 0;
                    while(i < self.batch_size and self.playing and not terminal):
                        tens_state = torch.FloatTensor([state]).to(self.model.device).permute(0, 3, 2, 1)
                        self.sr_game.sv.set_polled()
        
                        action, policy, value, rnd = self.model.step(tens_state)
        
                        next_state, reward, terminal = self.sr_game.step(action)

                        reward = reward + rnd

                        states.append(state)
                        actions.append(action)
                        rewards.append(reward)
                        values.append(value)
    
                        state = next_state
    
                        i += 1
    
                    if(len(states) == self.batch_size * self.sequence_length):
                        returns = discount(rewards, decay)
                        advantages = returns - values
                        advantages = normalize(advantages, 1e-5)
    
                        self.model.train_reinforce([states, actions, rewards,
                                                   advantages])

                        supervised = self.data_handler.sequenced_sample(
                                                           self.batch_size,
                                                           self.sequence_length,
                                                           self.model.device == "cuda"
                                                           )

                        self.model.train_supervised(*supervised)

                    if(episode % self.save_interval == 0):
                        self.model.save(self.save_path)

    def tap(self, keycode, character, press):
        """
        Will handle the key press event to start, stop and end the program.
        """
        print("\nReceived:", character)
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

        # Start the loop if it's the first time starting
        if(not self.listening):
            self.listening = True

            # Create a thread for the main loop
            loop_listening = Thread(target = self._loop_listening)
            loop_listening.start()

            self.sr_game.start()

    def pause_playing(self):
        """
        Will cause the program to stop recording
        """
        if(self.playing):
            print("Playing paused")
            self.playing = False
            self.env.pause()

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

    def read_config(self):
        """
        Reads the config file to obtain the settings for the recorder, the
        window size for the training data and the game bindings
        """
        config = ConfigParser()
    
        # Read the config file, make sure not to re-name
        config.read("config.ini")

        return (config["Window Size"], config["Playing"],
                config["SpeedRunners Config"])

def train_model(model, data_handler, epochs, batch_size, save_path):
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
        model.save(save_path + "/model-" + str(epoch) + ".torch")

if(__name__ == "__main__"):
    """
    TODO

    Use the il_weight to mix reinforcement learning with imitation learning
    """
    cuda = False
    device = "cuda" if cuda else "cpu"

    #model = Model(device)

    state_space = (128, 128, 1)
    act_n = 7
    batch_size = 10
    sequence_length = 15
    il_weight = 1.0
    model_args = (state_space, act_n, batch_size, il_weight, device)
    model = Model2(*model_args).to(torch.device(device))

    data_handler = HDF5Handler("r+", 1)
    print("Data length:", len(data_handler), "samples")
    episodes = 100
    decay = 0.99
    save_interval = 1

    save_path = "./Trained Models/"
    load_path = save_path + "mail-1.torch"
    #model.load(load_path)
    #train_model(model, data_handler, episodes, full_rollout_length, save_path)
    trainer = ModelTrainer(model, data_handler, episodes, batch_size,
                           sequence_length, decay, save_interval, load_path)
    trainer.run()
    #trainer.listening = True
    #trainer.playing = True
    #trainer._loop_listening()
