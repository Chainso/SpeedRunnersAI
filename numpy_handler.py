import numpy as np

from configparser import ConfigParser

class NumpyHandler():
    """
    A class to handle how the training data is stored and opened
    """
    def __init__(self, mode, save_interval):
        """
        Will create a data handler to easily control the storage and accessing
        of training data
        
        mode : The mode to open the file with
        save_interval : The number of data points to save at one time
        """
        # The save interval
        self.save_interval = save_interval

        # Read the config file
        training_config = self.read_config()

        # The file path for the HDF5 file
        self.file_path = ("./Training Data/" + training_config["FILE_NAME"] +
                          ".npz")

        # The HDF5 file to work with
        self.file = open(self.file_path, mode)
        
    def add(self, states, actions):
        """
        Adds the given states, actions and directions to the dataset

        states : The states to add to the dataset
        actions : The actions to add to the dataset
        """
        # Make sure the lengths of the data is the same and greater than 0
        assert len(states) == len(actions)
        assert len(states) > 0

        # Add the new data to the end of the dataset
        np.savez_compressed(self.file, states = states, actions = actions)
        print("saved")
        self.file.flush()

    def save_from_queue(self, queue):
        """
        Obtain data from a queue and save it (used in multiprocessing)

        queue : The queue to get the states and actions from
        """
        # The process just started
        end_proc = False

        while(not end_proc):
            # The states and actions to save
            states = []
            actions = []

            # Get from the queue until it is time to save
            while(len(states) < self.save_interval and not end_proc):
                if(queue.empty()):
                    print("loop")
                    # Get the message from the queue
                    msg = queue.get()

                    # Check if it the end message, break and save if it is
                    if(msg == "END"):
                        end_proc = True
                        self.close()
                    else:
                        # Then it is a state and action
                        state, action = msg
    
                        # Append them to the list of states
                        states.append(state)
                        actions.append(action)

            if(len(states) > 0):
                # Convert to a numpy array
                
                states = np.stack(states)
                actions = np.stack(actions)

                # Save the states and actions
                self.add(states, actions)

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
    
        return config["Training"]

    def load_data(self):
        """
        Loads and returns the npz data
        """
        return np.load(self.file_path)

