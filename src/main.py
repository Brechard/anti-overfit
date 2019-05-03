# Design any kind of network using any library to train with the given data
# Validation data is not given until the end of the assignment and will be given as an image
# The goal is to fight over-fitting and make the best prediction possible

import scipy.io
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

train_data = scipy.io.loadmat('../Xtrain.mat')['Xtrain']
train_data = [i[0] for i in train_data]


def plot_train_data():
    plt.plot(train_data)
    plt.title("Training data")
    plt.show()


#
batch_size = 50
steps_per_epoch = len(train_data) - batch_size

# Model initialization
model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Embedding(input_dim=batch_size, output_dim=batch_size))
model.add(tf.keras.layers.Dense(units=batch_size, input_shape=(batch_size,)))
model.add(tf.keras.layers.GRU(1, return_sequences=True))
model.compile(optimizer='adam', loss='mean_squared_error')

# Training
x_train = np.zeros((steps_per_epoch, batch_size))
y_train = np.zeros((steps_per_epoch, 1))
for i in range(steps_per_epoch):
    x_train[i] = train_data[i: i + batch_size]
    y_train[i] = train_data[i + batch_size]

model.fit(x_train, y_train, epochs=1)
model.evaluate(x_train, y_train)

# plot_train_data()
