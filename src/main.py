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
    y_train[i] = train_data[i + batch_size] \
        - train_data[i + batch_size - 1]

print("Shape of x_train", x_train.shape)
print("Shape of y_train", y_train.shape)


def train_model(epochs=50, json_str='model', weight_str='model', extra_name=""):
    # Model initialization
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(units=20, input_shape=(batch_size, 1)))
    model.add(tf.keras.layers.LSTM(1, dropout=0.4))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(x_train, y_train, epochs=epochs, batch_size=1, verbose=2)
    # model.evaluate(x_train, y_train)

    # serialize model to JSON
    model_json = model.to_json()
    with open(json_str + extra_name + '.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weight_str + extra_name + '.h5')
    print("Saved model to disk")


def load_model(json_str='model.json', weight_str='model.h5'):
    json_file = open(json_str, 'r')
    loaded_nnet = json_file.read()
    json_file.close()

    serve_model = tf.keras.models.model_from_json(loaded_nnet)
    serve_model.load_weights(weight_str)
    serve_model.compile(optimizer='adam', loss='mean_squared_error')

    print("Model loaded")

    return serve_model


def predict(model, last_window, predict_size=200):
    input_data = last_window
    predictions = []
    for i in range(predict_size):
        new_value = model.predict(input_data)
        # new_value += input_data[:, -1, :]
        predictions.append(new_value[0][0])

        new_input_data = np.zeros(input_data.shape)
        new_input_data[0, :-1, :] = input_data[0, 1:, :]
        new_input_data[0, -1, :] = new_value

        input_data = new_input_data

    return predictions


def correlation(model, real_data, window=100, plot=True):
    data_to_predict = real_data[-window:, 0]
    data_for_prediction_to_add = real_data[-window - 1:, -1]

    data_for_prediction = x_train[- window - batch_size: -batch_size]

    diff_predictions = model.predict(data_for_prediction)

    predictions = [data_for_prediction_to_add[i] + diff_predictions[i][0] for i in range(window)]

    correlation = np.correlate(data_to_predict, predictions, "same")

    if plot:
        plt.subplot(211)
        plt.plot(data_to_predict, label="Data to predict")
        plt.plot(predictions, label="Prediction")
        plt.legend()
        plt.subplot(212)
        plt.plot([i - int(len(data_to_predict) / 2) for i in range(len(data_to_predict))], correlation)
        plt.title("Correlation")
        plt.show()

    return correlation


train_model(30)
# predictions = predict(load_model(), x_train[-1:])

# pred = model.predict(x_train)
# prediction = np.zeros((len(train_data), 1))
# prediction[batch_size:] = pred

# print("Len of prediction", len(predictions))
# print("Len of train data", len(train_data))

# plt.plot(train_data, label="Real data")
# plt.plot([1000 + i for i in range(len(predictions))], predictions, label="Prediction")
# plt.legend()
# plt.show()

# print(prediction)
# plot_train_data()
correlation(load_model(), train_data)
# plt.show()
