import numpy as np
import h5py

from speedrunners.actor import Actor
from hdf5_handler import HDF5Handler

actor = Actor()
handler = HDF5Handler("r", 1)
states, acts = handler.states[()], handler.actions[()]
print(type(states), len(acts))

actions = np.zeros((len(acts), actor.num_actions()))
for i in range(len(handler)):
    action = actor.continuous_to_discrete(acts[i])
    actions[i] = action

training_config, window_size = handler.read_config()

# Get the screen size from the config
screen_size = (int(window_size["WIDTH"]), int(window_size["HEIGHT"]),
               int(window_size["DEPTH"]))

# The file path for the HDF5 file
file_path = ("./Training Data/discrete-" + training_config["FILE_NAME"] +
             ".hdf5")

f = h5py.File(file_path, "w")

fstates = f.create_dataset("states", (0, *screen_size),
                                                   chunks = (300,
                                                             *screen_size),
                                                   maxshape = (None,
                                                               *screen_size),
                                                   dtype = "i8")

factions = f.create_dataset("actions", (0, actor.num_actions()),
                            chunks = (300, actor.num_actions()),
                            maxshape = (None, actor.num_actions()))

fstates.resize(len(fstates) + len(states), axis = 0)
factions.resize(len(factions) + len(actions), axis = 0)

        # Add the new data to the end of the dataset
fstates[len(fstates) - len(states):, :, :, :] = states
factions[len(factions) - len(actions):, :] = actions

f.flush()
print(len(factions))
f.close()
