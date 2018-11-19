from model import Model
from hdf5_handler import HDF5Handler

def train_model(model, data_handler, epochs, batch_size, cuda, save_path):
    if(cuda):
        model = model.cuda()

    states = data_handler.get_states()
    for epoch in range(epochs):
        # The average loss of this epoch
        avg_loss = 0

        for index in range(0, len(data_handler), batch_size):
            # Get the next batch of states and actions
            states = data_handler.get_states(index, index + batch_size, cuda)
            actions = data_handler.get_actions(index, index + batch_size, cuda)

            # Compute the losses
            avg_loss += model.train(states, actions) * (batch_size /
                                                        len(data_handler))

        # Print the average loss
        print(avg_loss)

        # Save the model
        model.save(save_path + "/model-" + str(epoch + 1) + ".torch")

if(__name__ == "__main__"):
    model = Model()
    data_handler = HDF5Handler("r+", 1)

    epochs = 15
    batch_size = 69
    cuda = False
    save_path = "./Trained Models"
    model.load(save_path + "/model-15.torch")
    train_model(model, data_handler, epochs, batch_size, cuda, save_path)