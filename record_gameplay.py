import numpy as np

from numpy_handler import NumpyHandler
from screen_viewer import ScreenViewer

from collections import OrderedDict
from threading import Thread
from multiprocessing import Process, Queue
from configparser import ConfigParser
from pykeyboard import PyKeyboardEvent

class Recorder(PyKeyboardEvent):
    """
    A keyboard event listener to start and stop the recording.
    """
    def __init__(self, data_queue = None):
        """
        Will create a keyboard event listener to handle the recording and
        closing of the program using the settings in the config file.

        data_queue : The queue to put the data in if given
        """
        PyKeyboardEvent.__init__(self)

        # The data queue
        self.data_queue = data_queue

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

        self.sv = ScreenViewer(game_screen, res_screen_size)

        # The active actions and direction, right direction by default
        self.actions = [(self.speedrunners["JUMP"], 0),
                        (self.speedrunners["GRAPPLE"], 0),
                        (self.speedrunners["ITEM"], 0),
                        (self.speedrunners["BOOST"], 0),
                        (self.speedrunners["SLIDE"], 0),
                        ("direction", 1)]
        self.actions = OrderedDict(self.actions) 

    def _loop_listening(self):
        """
        Ensures that the program will continue listening until closure
        """
        # Create a list of states and actions
        states = []
        actions = []

        while(self.listening):
            # The last state
            last_state = None

            # The save counter
            save_counter = 0

            # Record the keys and game frames while recording is enabled
            while(self.recording):
                # Get the state and current action
                state = self.sv.GetNewScreen(last_state)
                action = [self.actions[button] for button in self.actions]

                # Append to the list of states and actions
                states.append(state)
                actions.append(action)

                last_state = state

                save_counter += 1

                # If it is time to save then convert to numpy and save to the
                # Numpy file
                if(save_counter == self.save_interval):
                    states = np.stack(states)
                    actions = np.stack(actions)

                    # Create a thread to save the data
                    saver = Thread(target = self.data_handler.add,
                                   args = (states, actions))
                    saver.start()

                    states = []
                    actions = []

    def _listen_with_queue(self):
        """
        Ensures that the program will continue listening until closure and put
        the data into a queue
        """
        while(self.listening):
            # The last state
            last_state = None

            # Record the keys and game frames while recording is enabled
            while(self.recording):
                # Get the state and current action
                state = self.sv.GetNewScreen(last_state)
                action = [self.actions[button] for button in self.actions]

                # Put the state and action in a queue
                self.data_queue.put([state, action]);

                last_state = state

    def run(self, data_queue = None):
        """
        Will run the recorder, using a queue to place data if provided

        data_queue : The queue to place the data in
        """
        self.data_queue = data_queue

        # The data handler for the training data, using read and write mode
        # Only open the file if not using queues
        if(self.data_queue is None):   
            self.data_handler = NumpyHandler("ab+", self.save_interval)

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
            # Check for left direction
            elif(character == self.speedrunners["LEFT"]):
                self.actions["direction"] = 0
            # Check for right direction
            elif(character == self.speedrunners["RIGHT"]):
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
        if(not self.recording):
            print("Recording started")
            self.recording = True

        # Start the loop if it's the first time starting
        if(not self.listening):
            self.listening = True

            # Make sure the keys are being recorded
            # Use the queue listening loop if a queue was provided
            if(self.data_queue is not None):
                loop_listening = Thread(target = self._listen_with_queue)
            else:
                loop_listening = Thread(target = self._loop_listening)
            loop_listening.start()

            # Start recording the screen
            self.sv.Start()

    def stop_recording(self):
        """
        Will cause the program to stop recording
        """
        if(self.recording):
            print("Recording paused")
            self.recording = False
            self.sv.Stop()

    def close_program(self):
        """
        Will close the program.
        """
        if(self.listening):
            print("Recording stopped")
            self.recording = False
            self.listening = False
            self.sv.Stop()

        if(self.data_queue is not None):
            self.data_queue.put("END")

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
    # Get the save interval from the configuration
    config = ConfigParser()

    # Read the config file, make sure not to re-name
    config.read("config.ini")

    save_interval = int(config["Recording"]["SAVE_INTERVAL"])
    """
    # The data queue
    data_queue = Queue()

    # Make the keyboard listener
    recorder = Recorder()
    
    # The recorder process
    record_proc = Process(target = recorder.run, args = (data_queue,))

    # The data handler
    data_handler = NumpyHandler("ab+", save_interval)

    # The data process
    data_proc = Process(target = data_handler.save_from_queue,
                        args = (data_queue,))
    data_proc.start()
    record_proc.start()
    """
    recorder = Recorder()
    recorder.run()