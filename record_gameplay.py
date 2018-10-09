from data_handler import DataHandler
from screen_viewer import ScreenViewer

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
        self.listening = True

        recording_config, window_size = self.read_config()

        # Get the start, end and close key from the config
        self.start_key = recording_config["START_KEY"].lower()
        self.end_key = recording_config["END_KEY"].lower()
        self.close_key = recording_config["CLOSE_KEY"].lower()

        # Get the screen size from the config
        screen_size = (int(window_size["WIDTH"]), int(window_size["HEIGHT"]),
                       int(window_size["DEPTH"]))

        self.sv = ScreenViewer(screen_size)
        #self.sv.GetHandle("SpeedRunners")

        # The data handler for the training data, using read and write mode
        self.data_handler = DataHandler("a")

        # The last action that was taken
        self.last_action = ""

        # Make sure that it is listening
        listening_loop = Thread(target = self._loop_listening)
        listening_loop.start()

    def _loop_listening(self):
        """
        Ensures that the program will continue listening until closure
        """
        while(self.listening):
            # Create a list of states, actions, and directions
            states = []
            actions = []

            # Record the keys and game frames while recording is enabled
            while(self.recording):
                state = self.sv.GetScreen()

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
            # Otherwise map the key to the direction and action
            else:
        elif(not press and character == self.bindings["boost"])
                
            
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
        Reads the config file to obtain the settings for the recorder and the
        window size for the training data
        """
        config = ConfigParser()
    
        # Read the config file, make sure not to re-name
        config.read("config.ini")
    
        return config["Recording"], config["Window Size"]
         
if(__name__ == "__main__"):
    # Make the keyboard listener
    recorder = Recorder()
    recorder.start()
