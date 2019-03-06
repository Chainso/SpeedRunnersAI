import torch

from configparser import ConfigParser
from pykeyboard import PyKeyboardEvent
from threading import Thread

from model import Model
from speedrunners import SpeedRunnersEnv
from time import time

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
        self.actor = Actor()

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

        self.sr_game = SpeedRunnersEnv(300, res_screen_size)
        self.sv = ScreenViewer(game_screen, res_screen_size)

    def _step(self):
        """
        Takes a single action into the environment.
        """
        # Get the state and current act
        state = self.sv.GetNewScreen()
        state = torch.FloatTensor([state]).permute(0, 3, 1, 2)
        state = state.to(self.model.device)

        action = self.model.step(state)
        self.actor.act(action)

    def _loop_listening(self):
        """
        Ensures that the program will continue listening until closure
        """
        while(self.listening):
            # Record the keys and game frames while recording is enabled
            while(self.playing):
                self._step()

    def tap(self, keycode, character, press):
        """
        Will handle the key press event to start, stop and end the program.
        """
        if(press):
            # Start recording on the recording key press
            if(character == self.start_key and not self.recording):
                self.start_recording()
            # Stop recording on the recording key press
            elif(character == self.end_key and self.recording):
                self.stop_recording()   
            # Close the program on the close key press
            elif(character == self.close_key):
                self.close_program()

    def resume_playing(self):
        """
        Will cause the program to start recording.
        """
        if(not self.recording):
            print("Playing resumed")
            self.playing = True

        # Start the loop if it's the first time starting
        if(not self.listening):
            self.listening = True

            # Create a thread for the main loop
            loop_listening = Thread(target = self._loop_listening)
            loop_listening.start()

            # Start recording the screen
            self.sv.Start()

    def pause_playing(self):
        """
        Will cause the program to stop recording
        """
        if(self.recording):
            print("Playing paused")
            self.playing = False
            self.sv.Stop()

    def close_program(self):
        """
        Will close the program.
        """
        if(self.listening):
            print("Finished playing")
            self.playing = False
            self.listening = False
            self.sv.Stop()

        self.stop()

def run_model(model, screen_viewer, actor, load_path, cuda):
    if(cuda):
        model = model.cuda()

    model.load(load_path)

    screen_viewer.Start()

    last = None

    while(True):
        a = time()
        screen = screen_viewer.GetNewScreen(last)
        print("Time 1:", time() - a)
        if(cuda):
            screen = torch.cuda.FloatTensor([screen]).permute(0, 3, 1, 2)
        else:
            screen = torch.FloatTensor([screen]).permute(0, 3, 1, 2)

        action = model(screen)[0]

        #actor.act(action)
        last = screen
        print("Time 2:", time() - a)
    screen_viewer.Stop()

    def read_config(self):
        """
        Reads in the window sizes for the screen viewer from the config file
        """
        config = ConfigParser()
    
        # Read the config file, make sure not to re-name
        config.read("config.ini")
    
        return (config["Window Size"], config["Playing"],
                config["speedrunners Config"])

if(__name__ == "__main__"):
    load_path = "./Trained Models/model-15.torch"

    model = Model()
    model.load(load_path)

    model_runner = ModelRunner()
    model_runner.run()

"""
if(__name__ == "__main__"):
    model = Model()

    window_size, playing_config, speedrunners_config = read_config()

    window_size = (int(window_size["WIDTH"]), int(window_size["HEIGHT"]),
                   int(window_size["DEPTH"]))

    speedrunners_size = (int(speedrunners_config["WIDTH"]),
                         int(speedrunners_config["HEIGHT"]))
    screen_viewer = ScreenViewer(speedrunners_size, window_size)

    actor = Actor()
    load_path = "./Trained Models/model-15.torch"
    cuda = False

    run_model(model, screen_viewer, actor, load_path, cuda)
"""