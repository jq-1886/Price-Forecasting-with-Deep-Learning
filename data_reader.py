"""Module containing a class to handle data pre-processing"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class Data_Reader:
    """Class to handle data pre-processing"""

    def __init__(self, file_str, test_year):
        """
        Initialiser for Data_Reader class to read data in .csv format

        Parameters
        ----------
            filename (str):
                The file to be read, must be chosen from the following:

                Commodities
                -----------
                'Al' - LME 3 month aluminium futures
                'Cu' - LME 3 month copper futures
                'Corn' - CME rolling active month corn futures

                Currencies
                ----------
                'EURCHF' - spot Euro / Swiss currency pair
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

            Note, all quotes are daily closing prices or yield

        Example
        -------
            d_reader = Data_Reader('Al', 2019)
        """

        # creating variables for directory traversing and file reading
        directory_str = './data/'
        suffix_str = '.csv'

        # creating file_str to read relevant file
        self.file_str = file_str
        self.test_year = test_year

        # converting ['date'] column to datetime values
        self.data = pd.read_csv(directory_str+file_str+suffix_str)
        self.data['date'] = pd.to_datetime(self.data['date'], dayfirst=True)

        # inserting ['year'] column to allow for mask filtering
        self.data['year'] = self.data['date'].dt.year

        # creating a dictionary of all financial securities to be examined
        self.security_dict = {
                        'Al': 'LME Aluminium 3m futures price',
                        'Cu': 'LME Copper 3m futures price',
                        'Corn': 'CME rolling active month corn futures price',
                        'EURCHF': 'Spot Euro/Swiss Franc exchange rate',
                        'EURUSD': 'Spot Euro/US dollar exchange rate',
                        'GBPUSD': 'Spot British Pound/US dollar exchange rate',
                        'Bund10y': '10y German Bund yield',
                        'Gilt10y': '10y British Gilt yield',
                        'Treasury10y': '10y US Treasury yield',
                        'Amazon': 'Amazon.com Inc. common stock price',
                        'Google': 'Alphabet Inc. class A common stock price',
                        'Nvidia': 'Nvidia Corporation common stock price'
                            }

        # creating a bool to keep track of method calls
        self.tt_bool = False

    def extract_train_test(self):
        """
        Method to extract training, test and validation datasets

        Creates Data_reader.train_data, .train_data_norm, .test_data,
        .test_data_norm, .val_train_data, .val_train_data_norm, .val_test_data
        and .val_test_data_norm attributes, where '_norm' suffixed attributes
        contain normalised prices of relevant attribute without '_norm' suffix

        Parameters
        ----------
            None

        Example
        -------
            d_reader.extract_train_test()
            d_reader.train_data
            d_reader.train_data_norm
            d_reader.test_data
            d_reader.test_data_norm
            d_reader.val_train_data
            d_reader.val_train_data_norm
            d_reader.val_test_data
            d_reader.val_test_data_norm

        Returns
        -------
            None
        """

        # recordring max and min year values
        max_year = max(self.data['year'])
        min_year = min(self.data['year'])

        # raising an exception if test_year is not equal to max_year
        if not max_year == self.test_year:
            raise ValueError('Test year is not last year of dataset')

        # raising an exception if test_year is not in dataset
        if min_year > self.test_year:
            raise ValueError('Test year is not included in dataset')

        # creating a mask for filtering the test set
        test_mask = self.data['date'].dt.year == int(self.test_year)
        # creating a mask for filtering the validation set
        val_mask = self.data['date'].dt.year == int(self.test_year - 1)

        # creating .train_data and .test_data attributes based on year
        self.train_data = self.data.price[~test_mask]
        self.test_data = self.data.price[test_mask]

        # creating .val_train_data and .val_test_data attributes based on year
        self.val_train_data = self.train_data[~val_mask]
        self.val_test_data = self.train_data[val_mask]

        # creating an instance of our scaler for normalisation
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler_val = MinMaxScaler(feature_range=(0, 1))

        # getting the length of training and test datasets
        self.train_len = len(self.train_data)
        self.test_len = len(self.test_data)

        # getting the length of validation training and test datasets
        self.val_train_len = len(self.val_train_data)
        self.val_test_len = len(self.val_test_data)

        # converting to a numpy array
        self.train_data = np.array(self.train_data, ndmin=2)
        # transposing the array to have leading axis as 1
        self.train_data = self.train_data.T
        # normalising the data
        self.train_data_norm = self.scaler.fit_transform(self.train_data)

        # converting to a numpy array
        self.test_data = np.array(self.test_data, ndmin=2)
        # transposing the array to have leading axis as 1
        self.test_data = self.test_data.T
        # normalising the data
        self.test_data_norm = self.scaler.fit_transform(self.test_data)

        # converting to a numpy array
        self.val_train_data = np.array(self.val_train_data, ndmin=2)
        # transposing the array to have leading axis as 1
        self.val_train_data = self.val_train_data.T
        # normalising the data
        self.val_train_data_norm = self.scaler_val.fit_transform(
            self.val_train_data)

        # converting to a numpy array
        self.val_test_data = np.array(self.val_test_data, ndmin=2)
        # transposing the array to have leading axis as 1
        self.val_test_data = self.val_test_data.T
        # normalising the data
        self.val_test_data_norm = self.scaler_val.fit_transform(
            self.val_test_data)

        # setting tt_bool to true
        self.tt_bool = True

    def extract_xy(self, window_len, time_distributed=False):
        """
        Method to extract X and y values from training, test and
        validation datasets

        Creates Data_reader.X_train, .y_train, .X_test, .y_test,
        .X_val_train, .y_val_train, .X_val_test and .y_val_test attributes

        Parameters
        ----------
            window_len (int):
                length of prediction horizon in days

            time_distributed (bool):
                bool to instigate reshaping of X array class attributes to
                ensure compatability with TimeDistributed Keras layers

                shape is changed from:
                    [samples, window_len, features]

                to:
                    [samples, subsequences, window_len, features]

                where:
                    - samples = number of window_len timesteps from which
                        predictions are made
                    - subsequences = 1
                    - window_len = length of timesteps from which predictions
                        are made
                    - features = 1

                default value is False

        Example
        -------
            d_reader.extract_xy()
            d_reader.X_train
            d_reader.y_train
            d_reader.X_test
            d_reader.y_test
            d_reader.X_val_train
            d_reader.X_val_test
            d_reader.y_val_train
            d_reader.y_val_test

        Returns
        -------
            None
        """

        # logic to ensure .extract_train_test() has been called
        if not self.tt_bool:
            raise Exception('call .extract_train_test() first')

        # defining the prediction horizon
        self.window_len = window_len

        # creating lists to store X and y values for training and testing
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

        # creating lists to store X and y values for validation
        self.X_val_train = []
        self.y_val_train = []
        self.X_val_test = []
        self.y_val_test = []

        # a loop to iterate through the training dataset
        for i in range(self.window_len, self.train_len):
            # appending window_len values to X_train
            self.X_train.append(self.train_data_norm[i - self.window_len:i])
            # appending single y value to y_train
            self.y_train.append(self.train_data_norm[i])

        # converting X_train and y_train to numpy arrays
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)

        # reshaping to enforce shape requirements
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0],
                                                 self.X_train.shape[1], 1))

        # a loop to iterate through the test dataset
        for i in range(self.window_len, self.test_len):
            # appending window_len values to X_test
            self.X_test.append(self.test_data_norm[i - self.window_len:i])
            # appending single y value to y_test
            self.y_test.append(self.test_data_norm[i])

        # converting X_test and y_test to a numpy arrays
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)

        # reshaping to enforce shape requirements
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0],
                                               self.X_test.shape[1], 1))

        # a loop to iterate through the validation training dataset
        for i in range(self.window_len, self.val_train_len):
            # appending window_len values to X_val_train
            self.X_val_train.append(self.val_train_data_norm[i - self.
                                    window_len:i])
            # appending single y value to y_val_train
            self.y_val_train.append(self.val_train_data_norm[i])

        # converting X_train and y_train to numpy arrays
        self.X_val_train = np.array(self.X_val_train)
        self.y_val_train = np.array(self.y_val_train)

        # reshaping to enforce shape requirements
        self.X_val_train = np.reshape(self.X_val_train,
                                      (self.X_val_train.shape[0],
                                       self.X_val_train.shape[1], 1))

        # a loop to iterate through the validation test dataset
        for i in range(self.window_len, self.val_test_len):
            # appending window_len values to X_test
            self.X_val_test.append(self.val_test_data_norm[i -
                                   self.window_len:i])
            # appending single y value to y_test
            self.y_val_test.append(self.val_test_data_norm[i])

        # converting X_test and y_test to a numpy arrays
        self.X_val_test = np.array(self.X_val_test)
        self.y_val_test = np.array(self.y_val_test)

        # reshaping to enforce shape requirements
        self.X_val_test = np.reshape(self.X_val_test,
                                     (self.X_val_test.shape[0],
                                      self.X_val_test.shape[1], 1))

        # assertions to ensure datasets are of correct sizes
        assert self.X_train.shape[0] == (self.train_len - self.window_len)
        assert self.X_train.shape[1] == self.window_len

        assert self.X_test.shape[0] == (self.test_len - self.window_len)
        assert self.X_test.shape[1] == self.window_len

        assert self.X_val_train.shape[0] == (self.val_train_len -
                                             self.window_len)
        assert self.X_val_train.shape[1] == self.window_len

        assert self.X_val_test.shape[0] == (self.val_test_len -
                                            self.window_len)
        assert self.X_val_test.shape[1] == self.window_len

        # if time_distributed bool is True, X arrays are reshaped
        if time_distributed:
            self.X_val_train = self.X_val_train.reshape(
                (self.X_val_train.shape[0], 1, self.window_len, 1))
            self.X_val_test = self.X_val_test.reshape(
                (self.X_val_test.shape[0], 1, self.window_len, 1))
            self.X_train = self.X_train.reshape(
                (self.X_train.shape[0], 1, self.window_len, 1))
            self.X_test = self.X_test.reshape(
                (self.X_test.shape[0], 1, self.window_len, 1))

    def extract_xy_extended(self, window_len, time_distributed=False,
                            output_len=15):
        """
        Method to extract X and y values from training and test datasets
        when using an extended prediction horizon

        Creates Data_reader.X_train, .y_train, .X_test and .y_test attributes

        Parameters
        ----------
            window_len (int):
                length of prediction horizon in days

            time_distributed (bool):
                bool to instigate reshaping of X array class attributes to
                ensure compatability with TimeDistributed Keras layers

                shape is changed from:
                    [samples, window_len, features]

                to:
                    [samples, subsequences, window_len, features]

                where:
                    - samples = number of window_len timesteps from which
                        predictions are made
                    - subsequences = 1
                    - window_len = length of timesteps from which predictions
                        are made
                    - features = 1

                default value is False

        Example
        -------
            d_reader.extract_xy()
            d_reader.X_train
            d_reader.y_train
            d_reader.X_test
            d_reader.y_test
            d_reader.X_val_train
            d_reader.X_val_test
            d_reader.y_val_train
            d_reader.y_val_test

        Returns
        -------
            None
        """

        # defining the prediction horizon
        self.window_len = window_len
        # defining the length of prediction
        self.output_len = output_len

        # creating lists to store X and y values for training and testing
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

        # creating lists to store X and y values for validation
        self.X_val_train = []
        self.y_val_train = []
        self.X_val_test = []
        self.y_val_test = []

        # a loop to iterate through the training dataset
        for i in range(self.window_len, self.train_len):
            # appending window_len values to X_train
            self.X_train.append(self.train_data_norm[i - self.window_len:i])
            # appending single y value to y_train
            self.y_train.append(self.train_data_norm[i - self.output_len:i])

        # converting X_train and y_train to numpy arrays
        self.X_train = np.array(self.X_train)
        self.y_train = np.array(self.y_train)

        # reshaping to enforce shape requirements
        self.X_train = np.reshape(self.X_train, (self.X_train.shape[0],
                                                 self.X_train.shape[1], 1))

        # a loop to iterate through the test dataset
        for i in range(self.window_len, self.test_len):
            # appending window_len values to X_test
            self.X_test.append(self.test_data_norm[i - self.window_len:i])
            # appending single y value to y_test
            self.y_test.append(self.test_data_norm[i - self.output_len:i])

        # converting X_test and y_test to a numpy arrays
        self.X_test = np.array(self.X_test)
        self.y_test = np.array(self.y_test)

        # reshaping to enforce shape requirements
        self.X_test = np.reshape(self.X_test, (self.X_test.shape[0],
                                               self.X_test.shape[1], 1))

    def extract_real_prices(self, y_pred, y_dummy):
        """
        Method to extract actual, predicted and dummy unnormalised prices

        Creates Data_reader.actual_price, .predicted_price and .dummy_price
        attributes

        Parameters
        ----------
            y_pred (array):
                y values predicted by a model from X

            y_dummy (array):
                y avlues predicted by Base_Model class from X

        Example
        -------
            d_reader.extract_real_prices()
            d_reader.actual_price
            d_reader.predicted_price
            d_reader.dummy_price

        Returns
        -------
            None
        """

        # reversing normalisation
        self.predicted_price = self.scaler.inverse_transform(y_pred)

        # retrieving real price from self.data
        self.y_true = self.data.price[self.train_len + self.window_len:]
        # converting to a numpy array
        self.y_true = np.array(self.y_true, ndmin=2)
        # transposing the array to have leading axis as 1
        self.y_true = self.y_true.T
        # normalising the acutal price
        self.y_true = self.scaler.fit_transform(self.y_true)
        # converting to a numpy array
        self.actual_price = np.array(self.y_true)
        # reversing normalisation
        self.actual_price = self.scaler.inverse_transform(self.actual_price)

        # converting y_dummy to real prices
        self.dummy_price = self.scaler.inverse_transform(y_dummy)

        # assertions to ensure datasets are of correct sizes
        assert self.actual_price.shape[0] == (self.test_len - self.window_len)
        assert y_dummy.shape[0] == (self.test_len - self.window_len)
        assert y_dummy.shape[1] == 1
