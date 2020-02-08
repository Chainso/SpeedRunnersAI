import h5py

def conv():
    # The file path for the HDF5 file
    file_path = "./Training Data/training_data.hdf5"
    
    # The HDF5 file to work with
    file = h5py.File(file_path, "a")
    
    actions = file["actions"]
    #actions[...] = (actions[...] * 2) - 1
    
if(__name__ == "__main__"):
    import multiprocessing as mp