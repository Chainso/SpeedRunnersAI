import numpy as np

from data_handler import DataHandler
#from screen_viewer import ScreenViewer

from collections import OrderedDict
from threading import Thread
from configparser import ConfigParser
from pykeyboard import PyKeyboardEvent

class Recorder(PyKeyboardEvent):
    """
    A keyboard event listener to start and stop the recording.
    """
    def __init__(self, save_interval):
        """
        Will create a keyboard event listener to handle the recording and
        closing of the program using the settings in the config file.
        """
        PyKeyboardEvent.__init__(self)

        # If the program is currently running
        self.recording = False

        # If the program is listening to the keyboard
        self.listening = True

        # Get the information from the config
        recording_config, window_size, self.bindings = self.read_config()

        # Get the start, end and close key and the save interval from the
        # config
        self.start_key = recording_config["START_KEY"].lower()
        self.end_key = recording_config["END_KEY"].lower()
        self.close_key = recording_config["CLOSE_KEY"].lower()
        self.save_interval = int(recording_config["SAVE_INTERVAL"])

        # Get the screen size from the config
        screen_size = (int(window_size["WIDTH"]), int(window_size["HEIGHT"]),
                       int(window_size["DEPTH"]))

        #self.sv = ScreenViewer(screen_size)
        #self.sv.GetHandle("SpeedRunners")

        # The data handler for the training data, using read and write mode    
        self.data_handler = DataHandler("a")

        # The active actions and direction, right direction by default
        self.actions = [(self.bindings["JUMP"], 0),
                        (self.bindings["GRAPPLE"], 0),
                        (self.bindings["ITEM"], 0),
                        (self.bindings["BOOST"], 0),
                        (self.bindings["SLIDE"], 0),
                        ("direction", 1)]
        self.actions = OrderedDict(self.actions) 

        # Make sure that it is listening
        listening_loop = Thread(target = self._loop_listening)
        listening_loop.start()

    def _loop_listening(self):
        """
        Ensures that the program will continue listening until closure
        """
        while(self.listening):
            # Create a list of states and actions
            states = []
            actions = []

            # The save counter
            save_counter = 0

            # Record the keys and game frames while recording is enabled
            while(self.recording):
                # Get the state and current action
                state = self.sv.GetScreen()
                action = [self.actions[press] for press in self.actions]

                # Append to the list of states and actions
                states.append(state)
                actions.append(action)

                save_counter += 1

                # If it is time to save then convert to numpy and save to the
                # HDF5 file
                if(save_counter == self.save_interval):
                    states = np.stack(states)
                    actions = np.stack(actions)

                    self.data_handler.add(states, actions)

    def tap(self, keycode, character, press):
        """
        Will handle the key press event to start, stop and end the program.
        """
        print(character, press)
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
            # Check for left direction
            elif(character == self.bindings["LEFT"]):
                self.actions["direction"] = 0
            # Check for right direction
            elif(character == self.bindings["RIGHT"]):
                self.actions["direction"] = 1
            # Otherwise map the key to the action
            elif(character in self.actions):
                self.actions[character] = 0
        # Map the key to the action
        elif(not press and character in self.actions):
            self.actions[character] = 1

    def start_recording(self):
        """
        Will cause the program to start recording.
        """
        self.recording = True

    def stop_recording(self):
        """
        Will cause the program to stop recording
        """
        self.recording = False

    def close_program(self):
        """
        Will close the program.
        """
        self.recording = False
        self.listening = False

    def read_config(self):
        """
        Reads the config file to obtain the settings for the recorder, the
        window size for the training data and the game bindings
        """
        config = ConfigParser()
    
        # Read the config file, make sure not to re-name
        config.read("config.ini")
    
        return (config["Recording"], config["Window Size"],
                config["SpeedRunners Bindings"])
         
if(__name__ == "__main__"):
    # Make the keyboard listener
    recorder = Recorder()
    recorder.start()
