import h5py
import torch
import numpy as np

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
                                                               *screen_size),
                                                   dtype = "i8")

            self.actions = self.file.create_dataset("actions", (0, 6),
                                                    chunks = (chunk_size, 6),
                                                    maxshape = (None, 6))
        else:
            self.states = self.file["states"]
            self.actions = self.file["actions"]

    def __len__(self):
        """
        The size of the dataset
        """
        return len(self.states)

    def resize(self, size):
        """
        Resizes the datasets in order to accomodate the size given
        """
        # Add the given size to the current size of the datasets
        self.states.resize(len(self.states) + size, axis = 0)
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
        self.states[len(self.states) - len(states):, :, :, :] = states
        self.actions[len(self.actions) - len(actions):, :] = actions

        self.file.flush()

    def close(self):
        """
        Closes the file
        """
        print("Dataset size:", str(len(self)))
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

    def sample(self, num_samples, cuda = False):
        """
        Retrieves the given number of samples from the training data

        num_samples : The number of samples to retrieve
        cuda : If the sample should return a cuda tensor
        """
        # Get the random indices to get data from
        randIndices = np.random.randint(0, len(self), num_samples)

        # Get the state and action at that index
        states = self.states[randIndices, :, :, :]
        actions = self.actions[randIndices, :]

        # Stack the array
        states = (np.stack(states) / 255.0)
        actions = np.stack(actions)

        # Convert to PyTorch tensor
        if(cuda):
            states = torch.cuda.FloatTensor(states).permute(0, 3, 1, 2)
            actions = torch.cuda.FloatTensor(actions)
        else:
            states = torch.FloatTensor(states).permute(0, 3, 1, 2)
            actions = torch.FloatTensor(actions)

        return states, actions

    def sequenced_sample(self, num_samples, sample_size, cuda = False):
        """
        Retreives a sequence of the given sample size at random from the
        training data

        num_samples : The number of sample to get
        sample_size : The size of each sample
        cuda : If the sample should be a cuda tensor
        """
        # The states and actions
        states = []
        actions = []

        # Get the random indices to get data from
        randIndices = np.random.randint(0, len(self) - sample_size, num_samples)

        for rand_int in randIndices:
            # Get the state and action at that index
            states.append(self.states[rand_int:rand_int + sample_size, :, :, :])
            actions.append(self.actions[rand_int:rand_int + sample_size, :])
  
        # Stack the arrays
        states = (np.stack(states) / 255.0)
        actions = np.stack(actions)

        # Convert to PyTorch tensor
        if(cuda):
            states = torch.cuda.FloatTensor(states).permute(0, 1, 4, 2, 3)
            actions = torch.cuda.FloatTensor(actions)
        else:
            states = torch.FloatTensor(states).permute(0, 1, 4, 2, 3)
            actions = torch.FloatTensor(actions)

        return states, actions

    def get_shuffled(self, cuda = False):
        """
        Returns the training data shuffled

        cuda : If the training data should be a cuda tensor
        """
        return self.sample(len(self), cuda)

    def get_states(self, start_index = 0, end_index = None, cuda = False):
        """
        Returns the states in the HDF5 file as a PyTorch Tensor

        start_index : The index to start retrieving from
        end_index : The index to stop at
        cuda : If the states should be a cuda tensor
        """
        # Convert the end index being none to the end of the array
        if(end_index is None):
            end_index = len(self)

        # Bound values from 0-1
        np_states = np.stack(self.states[start_index:end_index] / 255.0)

        # Get the proper tensor and permute
        if(cuda):
            tens_states = torch.cuda.FloatTensor(np_states).permute(0, 3, 1, 2)
        else:
            tens_states = torch.FloatTensor(np_states).permute(0, 3, 1, 2)

        return tens_states

    def get_actions(self, start_index = 0, end_index = None, cuda = False):
        """
        Returns the actions in the HDF5 File

        start_index : The index to start retrieving from
        end_index : The index to stop at
        cuda : If the states should be a cuda tensor
        """
        # Convert the end index being none to the end of the array
        if(end_index is None):
            end_index = len(self)

        # Bound values from 0-1
        np_actions = np.stack(self.actions[start_index:end_index])

        # Get the proper tensor
        if(cuda):
            tens_actions = torch.cuda.FloatTensor(np_actions)
        else:
            tens_actions = torch.FloatTensor(np_actions)

        return tens_actions
