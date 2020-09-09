import torch

from configparser import ConfigParser
from pykeyboard import PyKeyboardEvent
from threading import Thread

from model import Model
from speedrunners import SpeedRunnersEnv

class ModelRunner(PyKeyboardEvent):
    """
    A keyboard event listener to start and stop the recording.
    """
    def __init__(self, model):
        """
        Will play the game using the given model.

        Args:
            model: The model to use to play with.
        """
        PyKeyboardEvent.__init__(self)

        # The model to use
        self.model = model

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

        self.sr_game = SpeedRunnersEnv(None, game_screen, res_screen_size)

    def _step(self):
        """
        Takes a single action into the environment.
        """
        # Get the state and current act
        state = self.sr_game.state
        state = torch.FloatTensor([state]).permute(0, 3, 1, 2)
        state = state.to(self.model.device)

        action = self.model.step(state)
        self.actor.act(action)

    def _loop_listening(self):
        """
        Ensures that the program will continue listening until closure
        """
        while(self.listening):
            # Record the keys and gamej frames while recording is enabled
            while(self.playing):
                # Get the state and current action
                state = self.sr_game.state
                state = torch.FloatTensor([state]).to(self.model.device).permute(0, 3, 2, 1)

                action, policy, value, rnd_reward = self.model.step(state,
                                                                    False)

                self.sr_game.step(action)

    def tap(self, keycode, character, press):
        """
        Will handle the key press event to start, stop and end the program.
        """
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

        # Start the loop if it's the first time starting
        if(not self.listening):
            self.sr_game.reset()
            self.listening = True

            # Create a thread for the main loop
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

        self.stop()

    def read_config(self):
        """
        Reads the config file to obtain the settings for the recorder, the
        window size for the training data and the game bindings
        """
        config = ConfigParser()
    
        # Read the config file, make sure not to re-name
        config.read("../config/config.ini")

        return (config["Window Size"], config["Playing"],
                config["SpeedRunners Config"])

def model_config():
    """
    Reads the config file to obtain the settings for the recorder, the
    window size for the training data and the game bindings
    """
    config = ConfigParser()

    # Read the config file, make sure not to re-name
    config.read("../config/config.ini")

    return config["Window Size"], config["Playing"]

if(__name__ == "__main__"):
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"

    window_size, playing_config = model_config()

    load_path = "./models/" + playing_config["LOAD_PATH"] + ".torch"

    state_space =  (int(window_size["WIDTH"]),
                    int(window_size["HEIGHT"]),
                    int(window_size["DEPTH"]))

    act_n = 17
    batch_size = 1
    il_weight = 1.0
    model_args = (state_space, act_n, batch_size, il_weight, device)
    model = Model(*model_args).to(torch.device(device))
    model.load(load_path)
    model = model.eval()

    model_runner = ModelRunner(model)
    model_runner.run()
