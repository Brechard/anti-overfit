# Design any kind of network using any library to train with the given data
# Validation data is not given until the end of the assignment and will be given as an image
# The goal is to fight over-fitting and make the best prediction possible

import scipy.io
import matplotlib.pyplot as plt

train_data = scipy.io.loadmat('../Xtrain.mat')['Xtrain']


def plot_train_data():
    plt.plot(train_data)
    plt.title("Training data")
    plt.show()
