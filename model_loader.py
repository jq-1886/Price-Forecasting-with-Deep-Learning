"""Module containing classes to implement trained and untrained models"""
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


class Trained_Model:
    """Class to implement trained models"""

    def __init__(self, model_str, security_str='Al'):
        """
        Initialiser for Trained_Model class to load a model saved from Google
        Colab implementations

        Parameters
        ----------
            model_str (str):
                The general model to be implemented, must be chosen from the
                following:

                    Convolutional Neural Network (CNN)
                    ----------------------------------
                    'CNN' - vanilla CNN
                    'CNN_GRU' - CNN with GRU units
                    'CNN_LSTM' - CNN with LSTM units

                    Gated Recurrent Unit (GRU)
                    ---------------------------
                    'GRU' - stacked GRU
                    'GRU_AE' - stacked GRU autoencoder
                    'GRU_LSTM' - stacked GRU LSTM hybrid

                    Long Short-Term Memory Network (LSTM)
                    -------------------------------------
                    'LSTM' - stacked LSTM
                    'LSTM_AE' - stacked LSTM autoencoder
                    'LSTM_GRU' - stacked LSTM GRU hybrid

                    Multi-Layer Perceptron (MLP)
                    ----------------------------
                    'MLP' - Multi-layer perceptron
                    'MLP_AE' - Multi-layer perceptron autoencoder

            security_str (str):
                The security on which the model was trained, must be chosen
                from the following:

                    Commodities
                    -----------
                    'Al' - LME 3 month aluminium futures
                    'Cu' - LME 3 month copper futures
                    'Corn' - CME rolling active month corn futures

                    Currencies
                    ----------
                    'EURCHF' - spot Euro / Swiss Franc currency pair
                    'EURUSD' - spot Euro / US dollar currency pair
                    'GBPUSD' - spot British pound / US dollar currency pair

                    Fixed Income
                    ------------
                    'Bund10y' - rolling 10y German Bund yield
                    'Gilt10y' - rolling 10y British Gilt yield
                    'Treasury10y' - rolling 10y US Treasury yield

                    Equities
                    --------
                    'Amazon' - NASDQ listed Amazon.com Inc. common stock
                    'Google' - NASDAQ listed Alphabet Inc. class A common stock
                    'Nvidia' - NASDAQ listed NVIDIA Corporation common stock

        Example
        -------
            trained_model = Trained_Model('CNN', 'Al')
        """

        # assinging class attributes
        self.model_str = model_str
        self.security_str = security_str

        # loading model from 'saved_models' directory
        self.model = keras.models.load_model('./saved_models/'
                                             + self.model_str + '_'
                                             + self.security_str)

    def train(self, X_train, y_train, epochs=1, batch_size=32):
        """
        Method to train the model for a specified number of epochs

        Parameters
        ----------
            X_train (array or array like):
                An array of X values on which the model will be trained for
                'epochs' number of epochs, predicting y values from these X
                values

            y_train (array or array like):
                An array of correct y values against which the predictions
                will be assessed by the loss function

            epochs (int):
                The number of epochs, or complete cycles through the X values,
                the model performs, default is 1

            batch_size (int):
                The number of X batches the optimiser assesses when optimising
                the model, default is 32

        Example
        -------
            trained_model = Trained_Model('CNN', 'Al')
            trained_model.train(X_train, y_train, 1, 32)

        Returns
        -------
            None
            .model class attribute will have updated weights and biases
            following a successful call
        """

        # training the model via a call to keras fit() method
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def evaluate(self, X_test, y_test):
        """
        Method to evaluate model predictions through calls to sci-kit learn

        Parameters
        ----------
            X_test (array or array like):
                An array of X values from which the model will predict y

            y_test (array or array like):
                An array of correct y values against which the predictions
                will be assessed by the loss function

        Example
        -------
            trained_model = Trained_Model('CNN', 'Al')
            trained_model.train(X_train, y_train, 1, 32)
            trained_model.evaluate(X_test, y_test)

        Returns
        -------
            mse (flt):
                the mean squared error of the predictions

            rmse (flt):
                the root mean squared error of the predictions

            mae (flt):
                the mean absolute error of the predictions
        """

        # predicting y values with .model attribute
        y_pred = self.model.predict(X_test)
        # calculating mean squared error
        mse = mean_squared_error(y_test, y_pred)
        # calculating root mean squared error
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        # calculating mean absolute error
        mae = mean_absolute_error(y_test, y_pred)

        return mse, rmse, mae


class Untrained_Model(Trained_Model):
    """Class to implement untrained models, a subclass of Trained_Model"""

    def __init__(self, model_str, window_len=30):
        """
        Initialiser for Untrained_Model class to generate models with randomly
        initilaised weights, where the model configurations are the defaults
        of those implemented in the associated Colab notebooks

        Parameters
        ----------
            model_str (str):
                The general model to be implemented, must be chosen from the
                following:

                    Convolutional Neural Network (CNN)
                    ----------------------------------
                        'CNN' - vanilla CNN
                        'CNN_GRU' - CNN with GRU units
                        'CNN_LSTM' - CNN with LSTM units

                    Gated Recurrent Unit (GRU)
                    ---------------------------
                        'GRU' - stacked GRU
                        'GRU_AE' - stacked GRU autoencoder
                        'GRU_LSTM' - stacked GRU LSTM hybrid

                    Long Short-Term Memory Network (LSTM)
                    -------------------------------------
                        'LSTM' - stacked LSTM
                        'LSTM_AE' - stacked LSTM autoencoder
                        'LSTM_GRU' - stacked LSTM GRU hybrid

                    Multi-Layer Perceptron
                    ----------------------
                        'MLP' - Multi-layer perceptron
                        'MLP_AE' - Multi-layer perceptron autoencoder

        Example
        -------
            trained_model = Trained_Model('CNN', 'Al')
        """

        # assigning class attributes
        self.model_str = model_str
        self.window_len = window_len

        # setting the keras model to sequential mode
        self.model = Sequential()

        if self.model_str == 'CNN':
            # layer 1
            self.model.add(Conv1D(filters=128,
                                  kernel_size=2,
                                  activation='tanh',
                                  input_shape=(self.window_len, 1)))
            # layer 2
            self.model.add(Conv1D(filters=64,
                                  kernel_size=2,
                                  activation='tanh'))
            # layer 3
            self.model.add(MaxPooling1D(pool_size=2))
            # layer 4
            self.model.add(Flatten())
            # layer 5
            self.model.add(Dense(1))

        elif self.model_str == 'CNN_GRU':
            # layer 1
            self.model.add(TimeDistributed(Conv1D(filters=128,
                                                  kernel_size=2,
                                                  activation='tanh'),
                           input_shape=(None, self.window_len, 1)))
            # layer 2
            self.model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
            # layer 3
            self.model.add(TimeDistributed(Flatten()))
            # layer 4
            self.model.add(GRU(units=128,
                               activation='tanh',
                               return_sequences=False))
            # layer 5
            self.model.add(Dropout(0.1))
            # layer 6
            self.model.add(Dense(1))

        elif self.model_str == 'CNN_LSTM':
            # layer 1
            self.model.add(TimeDistributed(Conv1D(filters=128,
                                                  kernel_size=2,
                                                  activation='tanh'),
                           input_shape=(None, self.window_len, 1)))
            # layer 2
            self.model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
            # layer 3
            self.model.add(TimeDistributed(Flatten()))
            # layer 4
            self.model.add(LSTM(units=128,
                                activation='tanh',
                                return_sequences=False))
            # layer 5
            self.model.add(Dropout(0.1))
            # layer 6
            self.model.add(Dense(1))

        elif self.model_str == 'GRU':
            # input layer
            self.model.add(GRU(50,
                               activation='tanh',
                               return_sequences=True,
                               input_shape=(self.window_len, 1)))
            self.model.add(Dropout(0.2))
            # hidden layer
            self.model.add(GRU(50))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(1))

        elif self.model_str == 'GRU_AE':
            # layer 1
            self.model.add(GRU(units=128,
                               return_sequences=True,
                               input_shape=(self.window_len, 1),
                               activation='tanh'))
            self.model.add(Dropout(0.2))
            # layer 2
            self.model.add(GRU(units=64,
                               return_sequences=False,
                               activation='tanh'))
            self.model.add(Dropout(0.2))
            # layer 3
            self.model.add(RepeatVector(self.window_len))
            # layer 4
            self.model.add(GRU(units=64,
                               return_sequences=True,
                               activation='tanh'))
            self.model.add(Dropout(0.2))
            # layer 5
            self.model.add(GRU(units=128,
                               return_sequences=False,
                               activation='tanh'))
            self.model.add(Dropout(0.1))
            # layer 6
            self.model.add((Dense(1)))

        elif self.model_str == 'GRU_LSTM':
            # layer 1
            self.model.add(GRU(units=64,
                               activation='tanh',
                               return_sequences=True,
                               input_shape=(self.window_len, 1)))
            self.model.add(Dropout(0.2))
            # layer 2
            self.model.add(LSTM(units=64, activation='tanh'))
            self.model.add(Dropout(0.1))
            # layer 3
            self.model.add(Dense(1))

        elif self.model_str == 'LSTM':
            # input layer
            self.model.add(LSTM(units=50,
                                return_sequences=True,
                                input_shape=(self.window_len, 1),
                                activation='tanh'))
            self.model.add(Dropout(0.2))
            # second model layer with dropout
            self.model.add(LSTM(units=50,
                                return_sequences=True,
                                activation='tanh'))
            self.model.add(Dropout(0.2))
            # third model layer with dropout
            self.model.add(LSTM(units=50,
                                return_sequences=True,
                                activation='tanh'))
            self.model.add(Dropout(0.2))
            # fourth model layer with dropout
            self.model.add(LSTM(units=50, activation='tanh'))
            self.model.add(Dropout(0.2))
            # final layer with one output
            self.model.add(Dense(units=1))

        elif self.model_str == 'LSTM_AE':
            # layer 1
            self.model.add(LSTM(units=128,
                                return_sequences=True,
                                input_shape=(self.window_len, 1),
                                activation='tanh'))
            self.model.add(Dropout(0.2))
            # layer 2
            self.model.add(LSTM(units=64,
                                return_sequences=False,
                                activation='tanh'))
            self.model.add(Dropout(0.2))
            # layer 3
            self.model.add(RepeatVector(self.window_len))
            # layer 4
            self.model.add(LSTM(units=64,
                                return_sequences=True,
                                activation='tanh'))
            self.model.add(Dropout(0.2))
            # layer 5
            self.model.add(LSTM(units=128,
                                return_sequences=False,
                                activation='tanh'))
            self.model.add(Dropout(0.1))
            # layer 6
            self.model.add((Dense(1)))

        elif self.model_str == 'LSTM_GRU':
            # layer 1
            self.model.add(LSTM(units=64,
                                activation='tanh',
                                return_sequences=True,
                                input_shape=(self.window_len, 1)))
            self.model.add(Dropout(0.2))
            # layer 2
            self.model.add(GRU(units=64, activation='tanh'))
            self.model.add(Dropout(0.1))
            # layer 3
            self.model.add(Dense(1))

        elif self.model_str == 'MLP':
            # layer 1
            self.model.add(Dense(256,
                                 activation='tanh',
                                 input_shape=(self.window_len, 1)))
            # layer 2
            self.model.add(Dense(128, activation='tanh'))
            # layer 3
            self.model.add(Dense(64, activation='tanh'))
            # layer 4
            self.model.add(Dense(32, activation='tanh'))
            # layer 5
            self.model.add(Flatten())
            # layer 6
            self.model.add(Dense(1))

        elif self.model_str == 'MLP_AE':
            # layer 1
            self.model.add(Dense(256,
                                 activation='tanh',
                                 input_shape=(self.window_len, 1)))
            # layer 2
            self.model.add(Dense(128, activation='tanh'))
            # layer 3
            self.model.add(Dense(64, activation='tanh'))
            # layer 4
            self.model.add(Dense(32, activation='tanh'))
            # layer 5
            self.model.add(Flatten())
            # layer 6
            self.model.add(RepeatVector(self.window_len))
            # layer 7
            self.model.add(Dense(32, activation='tanh'))
            # layer 8
            self.model.add(Dense(64, activation='tanh'))
            # layer 9
            self.model.add(Dense(128, activation='tanh'))
            # layer 10
            self.model.add(Flatten())
            # layer 11
            self.model.add(Dense(1))

        # defining the optimizer for all model choices
        self.optimizer = Adam(learning_rate=0.005,
                              beta_1=0.9, beta_2=0.999, clipnorm=1.0)
        # compiling the model
        self.model.compile(self.optimizer, loss='mean_squared_error')

    def train(self, X_train, y_train, epochs=1, batch_size=32):
        """
        Method to train the model for a specified number of epochs, inherited
        from Trained_Model parent class

        Parameters
        ----------
            X_train (array or array like):
                An array of X values on which the model will be trained for
                'epochs' number of epochs, predicting y values from these X
                values

            y_train (array or array like):
                An array of correct y values against which the predictions
                will be assessed by the loss function

            epochs (int):
                The number of epochs, or complete cycles through the X values,
                the model performs, default is 1

            batch_size (int):
                The number of X batches the optimiser assesses when optimising
                the model, default is 32

        Example
        -------
            trained_model = Trained_Model('CNN', 'Al')
            trained_model.train(X_train, y_train, 1, 32)

        Returns
        -------
            None
            .model class attribute will have updated weights and biases
            following a successful call
        """

        # training the model using inherited method
        super().train(X_train, y_train, epochs, batch_size)

    def evaluate(self, X_test, y_test):
        """
        Method to evaluate model predictions through calls to sci-kit learn,
        inherited from Trained_Model parent class

        Parameters
        ----------
            X_test (array or array like):
                An array of X values from which the model will predict y

            y_test (array or array like):
                An array of correct y values against which the predictions
                will be assessed by the loss function

        Example
        -------
            trained_model = Trained_Model('CNN', 'Al')
            trained_model.train(X_train, y_train, 1, 32)
            trained_model.evaluate(X_test, y_test)

        Returns
        -------
            mse (flt):
                the mean squared error of the predictions

            rmse (flt):
                the root mean squared error of the predictions

            mae (flt):
                the mean absolute error of the predictions
        """

        # returning values from inherited method
        return super().evaluate(X_test, y_test)
