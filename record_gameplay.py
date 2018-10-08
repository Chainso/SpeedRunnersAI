import h5py

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

        recording_config, window_size = self.read_config()

        # Get the start, end and close key from the config
        self.start_key = recording_config["START_KEY"]
        self.end_key = recording_config["END_KEY"]
        self.close_key = recording_config["CLOSE_KEY"]

        # Get the screen size from the config
        self.screen_size = (window_size["WIDTH"], window_size["HEIGHT"])

    def _loop_record(self):
        """
        The main loop to record the gameplay while the program is running.
        """
        while(self.recording):
            pass

    def tap(self, keycode, character, press):
        """
        Will handle the key press event to start, stop and end the program.
        """
        print("seen")

    def start_recording(self, func):
        """
        Will cause the program to start recording.

        func : The function to execute while the program is recording
        """
        self.recording = True
        #self.sv = ScreenViewer(screen_size)
        print("started")
        # Start a new thread for the loop to run
        run_loop = Thread(target = self._loop_record, args = (func,))
        run_loop.start()

    def stop_recording(self):
        """
        Will cause the program to stop recording
        """
        self.recording = False
        print("ended")

    def close_program(self):
        """
        Will close the program.
        """
        self.recording = False
        print("finito")

    def read_config(self):
        config = ConfigParser()
    
        # Read the config file, make sure not to re-name
        config.read("config.ini")
    
        return config["Recording"], config["Window Size"]

if(__name__ == "__main__"):
    # Make the keyboard listener
    recorder = Recorder()
    print("prog")