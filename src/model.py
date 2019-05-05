import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Model:
    def __init__(self, x_train, y_train, batch_size, n_nodes_hidden_layer, dropout,
                 json_str='model', weight_str='model', extra_name=""):
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.nodes = n_nodes_hidden_layer
        self.dropout = dropout
        self.json_str = '../models/' + json_str + extra_name + '_dropout_' + str(dropout) + '.json'
        self.weight_str = '../models/' + weight_str + extra_name + '_dropout_' + str(dropout) + '.h5'

        # Model initialization
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(self.nodes, dropout=dropout))
        model.add(tf.keras.layers.Dense(1))
        model.compile(optimizer='adam', loss='mse')
        self.model = model

    def train_model(self, epochs=50, save_model_weights=True, train_batch_size=1):
        """
        Train the network and save the weights if indicated as input
        :param epochs: Number of epochs to train the network
        :param save_model_weights: Flag to save the weights when training is done
        :param train_batch_size: Batch size used for the fit function
        :return: information from the training
        """

        train_output = self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=train_batch_size, verbose=2)

        if save_model_weights:
            self.save_model_weights()

        return train_output

    def save_model_weights(self):
        """ Save the weights of the model """

        # serialize model to JSON
        model_json = self.model.to_json()
        with open(self.json_str, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(self.weight_str)
        print("Saved model to disk")

    def load_model_weights(self):
        """ Load the weights of the model """

        json_file = open(self.json_str, 'r')
        loaded_nnet = json_file.read()
        json_file.close()

        serve_model = tf.keras.models.model_from_json(loaded_nnet)
        serve_model.load_weights(self.weight_str)
        serve_model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = serve_model
        print("Model loaded")

    def predict(self, predict_size=200, plot=True, original_data=None, last_window=None):
        """
        Predict the next values from the data given. It first uses the last_window values to predict the next value
        and then slides the window with the new value to create the next prediction.
        :param predict_size: Number of predictions to do
        :param plot: flag to plot the predictions
        :param original_data: data used to create the training data
        :param last_window: Values that will be used to create the first prediction.
        :return: Values of the predictions as an array
        """
        if plot and original_data is None:
            raise Exception("To plot the original data is needed")

        last_window = self.x_train[-1:] if last_window is None else last_window
        input_data = last_window
        predictions = []
        for i in range(predict_size):
            new_value = self.model.predict(input_data)
            # new_value += input_data[:, -1, :]
            predictions.append(new_value[0][0])

            new_input_data = np.zeros(input_data.shape)
            new_input_data[0, :-1, :] = input_data[0, 1:, :]
            new_input_data[0, -1, :] = new_value

            input_data = new_input_data

        if plot:
            plt.plot(original_data, label="Real data")
            plt.plot([1000 + i for i in range(len(predictions))], predictions, label="Prediction")
            title = "Batch size = " + str(self.batch_size) + ". Nodes = " + str(self.nodes)
            title += ". Dropout = " + str(self.dropout) if self.dropout else ""
            plt.title(title)
            plt.legend()
            plt.show()

        return predictions

    def correlation(self, real_data, window=100, plot=True):
        """
        Calculates the cross-correlation of the values predicted and the real values. Used to check that we have
        not created a persistence model.
        :param real_data: Data that we will try to predict
        :param window: Number of values from that data we will predict
        :param plot: Flag to plot the results
        :return: correlation values
        """
        data_to_predict = real_data[-window:, 0]
        data_for_prediction = self.x_train[- window - self.batch_size: -self.batch_size]

        predictions = self.model.predict(data_for_prediction)[:, 0]

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
