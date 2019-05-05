# Design any kind of network using any library to train with the given data
# Validation data is not given until the end of the assignment and will be given as an image
# The goal is to fight over-fitting and make the best prediction possible

import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from model import Model

# tf.enable_eager_execution()

train_data = scipy.io.loadmat('../Xtrain.mat')['Xtrain']


def plot_train_data():
    plt.plot(train_data)
    plt.title("Training data")
    plt.show()


# Prepare the data for the network
def prepare_data(batch_size=50):
    steps_per_epoch = len(train_data) - batch_size

    x_train = np.zeros((steps_per_epoch, batch_size, 1))
    y_train = np.zeros((steps_per_epoch, 1))
    for i in range(steps_per_epoch):
        x_train[i] = train_data[i: i + batch_size]
        y_train[i] = train_data[i + batch_size]

    print("Shape of x_train", x_train.shape)
    print("Shape of y_train", y_train.shape)

    return x_train, y_train


def study_batches_and_net_dim(dropouts, batch_sizes, n_nodes_hidden_layers, epochs=100):
    for dropout in dropouts:
        for batch_size in batch_sizes:
            x_train, y_train = prepare_data(batch_size)

            for nodes in n_nodes_hidden_layers:
                model = Model(x_train, y_train, batch_size, nodes, dropout=dropout,
                              extra_name="_nodes_" + str(nodes) + "_batch_" + str(batch_size))
                output_train = model.train_model(epochs)
                predictions = model.predict()

                print("Len of prediction", len(predictions))
                print("Len of train data", len(train_data))

                plt.plot(train_data, label="Real data")
                plt.plot([1000 + i for i in range(len(predictions))], predictions, label="Prediction")
                title = "Batch size = " + str(batch_size) + ". Nodes = " + str(nodes) + ". Epochs = " + str(
                    epochs) + ". Dropout = " + str(dropout)
                plt.title(title)
                plt.legend()
                plt.show()

                plt.plot(output_train.history['loss'])
                plt.title("Loss history in training. " + title)
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.show()


# study_batches_and_net_dim([0, 0.3, 0.5, 0.7], [10, 50, 100, 200, 400, 500], [2, 5, 10, 50, 100, 200, 400])
batch_size = 100
nodes = 10
epochs = 1

x_train, y_train = prepare_data(batch_size)
model = Model(x_train, y_train, batch_size, nodes, dropout=0,
              extra_name="_nodes_" + str(nodes) + "_batch_" + str(batch_size))
model.train_model(epochs=epochs)
# model.load_model_weights()
# predict_plot(epochs, 500)
model.predict(original_data=train_data)

# model = Model(x_train, y_train, batch_size, nodes, dropout=0,
#               extra_name="_nodes_" + str(nodes) + "_batch_" + str(batch_size))
# model.train_model(epochs=epochs)
#
# predict_plot(epochs, 500)
