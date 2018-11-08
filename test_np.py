import numpy as np

file_path = "./Training Data/training_data.npz"

file = open(file_path, "rb+")

data = np.load(file)
print(data)
states = data["states"]
actions = data["actions"]

#print("States\n\n", states[:])
print("\n\nActions\n\n", len(actions), "\n\n", actions[:])