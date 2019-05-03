# Design any kind of network using any library to train with the given data
# Validation data is not given until the end of the assignment and will be given as an image
# The goal is to fight over-fitting and make the best prediction possible

import scipy.io
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# tf.enable_eager_execution()

train_data = scipy.io.loadmat('../Xtrain.mat')['Xtrain']


def plot_train_data():
    plt.plot(train_data)
    plt.title("Training data")
    plt.show()


# Prepare the data for the network
batch_size = 50
steps_per_epoch = len(train_data) - batch_size

x_train = np.zeros((steps_per_epoch, batch_size, 1))
y_train = np.zeros((steps_per_epoch, 1))
for i in range(steps_per_epoch):
    x_train[i] = train_data[i: i + batch_size]
    y_train[i] = train_data[i + batch_size]

print("Shape of x_train", x_train.shape)
print("Shape of y_train", y_train.shape)

# Model initialization
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(units=30, input_shape=(batch_size, 1)))
model.add(tf.keras.layers.LSTM(10))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, epochs=100, batch_size=1)
