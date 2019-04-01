import numpy as np

from hdf5_handler2 import HDF5Handler
from speedrunners import SpeedRunnersEnv

from collections import OrderedDict
from threading import Thread
from configparser import ConfigParser
from pykeyboard import PyKeyboardEvent

class Recorder(PyKeyboardEvent):
    """
    A keyboard event listener to start and stop the recording.
    """
    def __init__(self):
        """
        Will create a keyboard event listener to handle the recording and
        closing of the program using the settings in the config file.
        """
        PyKeyboardEvent.__init__(self)

        # If the program is currently running
        self.recording = False

        # If the program is listening to the keyboard
        self.listening = False

        # Get the information from the config
        recording_config, window_size, self.speedrunners = self.read_config()

        # Get the start, end and close key and the save interval from the
        # config
        self.start_key = recording_config["START_KEY"].lower()
        self.end_key = recording_config["END_KEY"].lower()
        self.close_key = recording_config["CLOSE_KEY"].lower()
        self.save_interval = int(recording_config["SAVE_INTERVAL"])

        # Get the game screen size
        game_screen = (int(self.speedrunners["WIDTH"]),
                       int(self.speedrunners["HEIGHT"]))

        # Get the resized screen size from the config
        res_screen_size = (int(window_size["WIDTH"]),
                           int(window_size["HEIGHT"]),
                           int(window_size["DEPTH"]))

        self.sr_game = SpeedRunnersEnv(None, game_screen, res_screen_size)

        # The active actions and direction, right direction by default
        # Values -1 for off, 1 for on
        self.actions = [(self.speedrunners["JUMP"], 0),
                        (self.speedrunners["GRAPPLE"], 0),
                        (self.speedrunners["ITEM"], 0),
                        (self.speedrunners["BOOST"], 0),
                        (self.speedrunners["SLIDE"], 0),
                        (self.speedrunners["LEFT"], 0),
                        (self.speedrunners["RIGHT"], 0)]

        self.actions = OrderedDict(self.actions) 

    def _loop_listening(self):
        """
        Ensures that the program will continue listening until closure
        """
        # Create a list of states and actions
        states = []
        actions = []

        while(self.listening):
            # The save counter
            save_counter = 0

            # Record the keys and game frames while recording is enabled
            while(self.recording):
                # Get the state and current action
                state = self.sr_game.sv.GetNewScreen()
                self.sr_game.sv.set_polled()
                action = [self.actions[button] for button in self.actions]

                # Append to the list of states and actions
                states.append(state)
                actions.append(action)

                save_counter += 1

                # If it is time to save then convert to numpy and save to the
                # Numpy file
                if(save_counter == self.save_interval):
                    states = np.stack(states)
                    actions = np.stack(actions)

                    # Create a thread to save the datajjp
                    saver = Thread(target = self.data_handler.add,
                                   args = (states, actions))
                    saver.start()

                    states = []
                    actions = []
                    save_counter = 0

    def run(self):
        """
        Will run the recorder, using a queue to place data if provided

        data_queue : The queue to place the data in
        """
        # Create the data handler to use
        self.data_handler = HDF5Handler("a", self.save_interval)

        PyKeyboardEvent.run(self)

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
            # Otherwise map the key to the action
            elif(character in self.actions):
                self.actions[character] = 1
        # Map the key to the action
        elif(not press and character in self.actions):
            self.actions[character] = 0

    def start_recording(self):
        """
        Will cause the program to start recording.
        """
        if(not self.recording):
            print("Recording started")
            self.recording = True

        # Start the loop if it's the first time starting
        if(not self.listening):
            self.listening = True

            # Create a thread for the main loop
            loop_listening = Thread(target = self._loop_listening)
            loop_listening.start()

            # Start recording the screen
            self.sr_game.start()

    def stop_recording(self):
        """
        Will cause the program to stop recording
        """
        if(self.recording):
            print("Recording paused")
            self.recording = False
            self.sr_game.pause()

    def close_program(self):
        """
        Will close the program.
        """
        if(self.listening):
            print("Recording stopped")
            self.recording = False
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

        return (config["Recording"], config["Window Size"],
                config["SpeedRunners Config"])

if(__name__ == "__main__"):
    # Make the keyboard listener
    recorder = Recorder()
    recorder.run()
