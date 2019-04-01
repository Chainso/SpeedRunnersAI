import torch

from model2 import Model2
from hdf5_handler import HDF5Handler

def train_model(model, data_handler, epochs, batch_size, save_path):
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

if(__name__ == "__main__"):
    cuda = False
    device = "cuda" if cuda else "cpu"

    #model = Model(device)

    state_space = (128, 128, 1)
    act_n = 7
    il_weight = 1.0
    model_args = (state_space, act_n, il_weight, device)
    model = Model2(*model_args).to(torch.device(device))

    data_handler = HDF5Handler("r+", 1)
    print(len(data_handler))    
    epochs = 100
    batch_size = 15 * 10

    save_path = "./Trained Models/"
    load_path = save_path + "model2-1.torch"
    #model.load(load_path)
    train_model(model, data_handler, epochs, batch_size, save_path)
