import h5py

from configparser import ConfigParser

class HDF5Handler():
    """
    A class to handle how the training data is stored and opened
    """
    def __init__(self, mode, chunk_size):
        """
        Will create a data handler to easily control the storage and accessing
        of training data

        mode : The mode to open the file with (HDF5 modes)
        chunk_size : The chunk size to use if creating the datasets
        """
        self.chunk_size = chunk_size

        # Read the config file
        training_config, window_size = self.read_config()

        # Get the screen size from the config
        screen_size = (int(window_size["WIDTH"]), int(window_size["HEIGHT"]),
                       int(window_size["DEPTH"]))

        # The file path for the HDF5 file
        self.file_path = ("./Training Data/" + training_config["FILE_NAME"] +
                          ".hdf5")

        # The HDF5 file to work with
        self.file = h5py.File(self.file_path, mode)

        # Create the datasets if it is a new file
        # Start off at size 0, but they are resizable
        if(len(self.file.keys()) == 0):
            self.states = self.file.create_dataset("states", (0, *screen_size),
                                                   chunks = (chunk_size,
                                                             *screen_size),
                                                   maxshape = (None,
                                                               *screen_size))

            self.actions = self.file.create_dataset("actions", (0, 6),
                                                   chunks = (chunk_size, 6),
                                                    maxshape = (None, 6))
        else:
            self.states = self.file["states"]
            self.actions = self.file["actions"]

    def resize(self, size):
        """ 
        Resizes the datasets in order to accomodate the size given
        """
        # Add the given size to the current size of the datasets
        #self.states.resize(len(self.states) + size, axis = 0)
        self.actions.resize(len(self.actions) + size, axis = 0)
        
    def add(self, states, actions):
        """
        Adds the given states, actions and directions to the dataset

        states : The states to add to the dataset
        actions : The actions to add to the dataset
        """
        # Make sure the lengths of the data is the same and greater than 0
        assert len(states) == len(actions)
        assert len(states) > 0

        # Resize the datasets to add the size of the new data
        self.resize(len(states))

        # Add the new data to the end of the dataset
        #self.states[len(self.states) - len(states):, :, :, :] = states
        self.actions[len(self.actions) - len(actions):, :] = actions

        self.file.flush()

    def close(self):
        self.file.close()

    def read_config(self):
        """
        Reads the config file to obtain the settings for training data and the
        window size
        """
        config = ConfigParser()
    
        # Read the config file, make sure not to re-name
        config.read("config.ini")
    
        return config["Training"], config["Window Size"]

    def get_states(self):
        """
        Returns the states in the HDF5 file
        """
        return self.states

    def get_actions(self):
        """
        Returns the actions in the HDF5 File
        """
        return self.actions
