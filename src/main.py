# Design any kind of network using any library to train with the given data
# Validation data is not given until the end of the assignment and will be given as an image
# The goal is to fight over-fitting and make the best prediction possible

import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from model import Model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

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
            losses = {}
            for nodes in n_nodes_hidden_layers:
                model = Model(x_train, y_train, batch_size, nodes, dropout=dropout,
                              extra_name="_nodes_" + str(nodes) + "_batch_" + str(batch_size))
                output_train = model.train_model(epochs)
                predictions = model.predict(1000, original_data=train_data)

                losses[str(nodes)] = output_train['loss']

            for key, value in losses.items():
                plt.plot(value, label=str(key) + " nodes")
            plt.title("Loss history in training")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()


def print_errors(test_data, predictions):
    print("MSE =", mean_squared_error(test_data, predictions))
    print("MAE =", mean_absolute_error(test_data, predictions))
    print("r_2 =", r2_score(test_data, predictions))
    print("RMSE =", math.sqrt(mean_squared_error(test_data, predictions)))


batch_size = 450
nodes = 100
epochs = 100

x_train, y_train = prepare_data(batch_size)
model = Model(x_train, y_train, batch_size, nodes, dropout=0,
              extra_name="_nodes_" + str(nodes) + "_batch_" + str(batch_size))
model.train_model(epochs=epochs, train_batch_size=16)
# model.load_model_weights()
model.predict(200, original_data=train_data)
