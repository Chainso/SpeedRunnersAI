import torch

from configparser import ConfigParser

from model import Model
from hdf5_handler import HDF5Handler

def train_model(model, data_handler, epochs, batch_size, save_path, cuda):
    model = model.train()

    for epoch in range(1, epochs + 1):
        # Reset the hidden state
        model.reset_hidden_state()

        # The average loss of this epoch
        avg_loss = 0

        for index in range(0, len(data_handler), batch_size):
            # Get the next batch of states and actions
            states = data_handler.get_states(index, index + batch_size, cuda)
            actions = data_handler.get_actions(index, index + batch_size, cuda)

            # Compute the losses
            avg_loss += (model.train_supervised(states, actions)
                         * (batch_size /len(data_handler)))

        # Print the average loss
        print("Epoch ", epoch, ": loss", avg_loss)

        # Save the model
        model.save(save_path + "/model-" + str(epoch) + ".torch")

def model_config():
    """
    Reads the config file to obtain the settings for the recorder, the
    window size for the training data and the game bindings
    """
    config = ConfigParser()

    # Read the config file, make sure not to re-name
    config.read("config.ini")

    return config["Window Size"]

if(__name__ == "__main__"):
    cuda = torch.cuda.is_available()
    device = "cuda" if cuda else "cpu"

    window_size = model_config()

    state_space =  (int(window_size["WIDTH"]),
                    int(window_size["HEIGHT"]),
                    int(window_size["DEPTH"]))

    act_n = 7
    batch_size = 1
    il_weight = 1.0
    model_args = (state_space, act_n, batch_size, il_weight, device)
    model = Model(*model_args).to(torch.device(device))

    data_handler = HDF5Handler("r+", 1)
    print("Dataset size:", len(data_handler))

    epochs = 100
    full_rollout_length = batch_size * 15

    save_path = "./Trained Models/"
    load_path = save_path + "model-1.torch"
    #model.load(load_path)
    train_model(model, data_handler, epochs, full_rollout_length, save_path,
                cuda)
