import h5py

file = h5py.File("./Training Data/training_data.hdf5", "r+")

states = file["states"]
actions = file["actions"]

print("States\n\n", states[:])
print("\n\nActions\n\n", actions[:], "\n\n", len(actions))

num = 0
for action in actions:
    sub = 0
    for a in action:
        if(a == 1):
            sub += 1

    if(sub > 1):
        num += 1

print(num)