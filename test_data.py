import h5py

file = h5py.File("./Training Data/training_data.hdf5", "r+")

states = file["states"]
actions = file["actions"]

#print("States\n\n", states[:])
print("\n\nActions\n\n", len(actions), "\n\n", actions[:])