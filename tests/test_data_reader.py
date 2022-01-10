import pytest
import numpy as np

# import data_reader as dr
from data_reader import Data_Reader

in_data = Data_Reader('Al', 2019)
max_year = max(in_data.data['year'])
min_year = min(in_data.data['year'])
window_len = 30

# probably don't need this
model_list = ['CNN', 'CNN_GRU', 'CNN_LSTM', 'GRU', 'GRU_AE', 'GRU_LSTM',
              'LSTM', 'LSTM_AE', 'LSTM_GRU', 'MLP', 'MLP_AE']

security_list = ['Al', 'Cu', 'Corn', 'EURCHF', 'EURUSD', 'GBPUSD', 'Gilt10y',
                 'Bund10y', 'Treasury10y', 'Amazon', 'Google', 'Nvidia']


def test_date():
    d_reader = Data_Reader('Al', max_year + 1)

    with pytest.raises(Exception):
        d_reader.extract_train_test()

    d_reader = Data_Reader('Al', min_year - 1)

    with pytest.raises(Exception):
        d_reader.extract_train_test()


def test_extract_train_test():
    for s in security_list:
        d_reader = Data_Reader(s, 2019)
        d_reader.extract_train_test()

        d_len = d_reader.data.shape[0]
        train_len = d_reader.train_data.shape[0]
        test_len = d_reader.test_data.shape[0]

        assert d_len == train_len + test_len

        max_train = max(d_reader.train_data_norm)
        max_test = max(d_reader.test_data_norm)
        max_val_train = max(d_reader.val_train_data_norm)
        max_val_test = max(d_reader.val_test_data_norm)
        assert np.allclose(max_train, max_test, max_val_train, max_val_test,
                           1.0)

        min_train = min(d_reader.train_data_norm)
        min_test = min(d_reader.test_data_norm)
        min_val_train = min(d_reader.val_train_data_norm)
        min_val_test = min(d_reader.val_test_data_norm)
        assert np.allclose(min_train, min_test, min_val_train, min_val_test,
                           0.0)

        assert np.argmax(d_reader.train_data) == np.argmax(d_reader.
                                                           train_data_norm)
        assert np.argmax(d_reader.test_data) == np.argmax(d_reader.
                                                          test_data_norm)
        assert np.argmax(d_reader.val_train_data) == np.argmax(
                d_reader.val_train_data_norm)
        assert np.argmax(d_reader.val_test_data) == np.argmax(
                d_reader.val_test_data_norm)


def test_extract_xy():
    for s in security_list:
        d_reader = Data_Reader(s, 2019)
        d_reader.extract_train_test()
        d_reader.extract_xy(window_len)

        X_train_shape = d_reader.X_train.shape
        X_test_shape = d_reader.X_test.shape
        X_val_train_shape = d_reader.X_val_train.shape
        X_val_test_shape = d_reader.X_val_test.shape

        y_train_shape = d_reader.y_train.shape
        y_test_shape = d_reader.y_test.shape
        y_val_train_shape = d_reader.y_val_train.shape
        y_val_test_shape = d_reader.y_val_test.shape

        train_shape = d_reader.train_data.shape
        test_shape = d_reader.test_data.shape

        X_val_shape = X_val_train_shape[0] + X_val_test_shape[0]
        y_val_shape = y_val_train_shape[0] + y_val_test_shape[0]

        assert X_train_shape[1] == window_len
        assert X_test_shape[1] == window_len
        assert X_train_shape[2] == 1
        assert X_test_shape[2] == 1

        assert X_val_train_shape[1] == window_len
        assert X_val_test_shape[1] == window_len
        assert X_val_train_shape[2] == 1
        assert X_val_test_shape[2] == 1

        assert y_train_shape[1] == 1
        assert y_test_shape[1] == 1

        assert train_shape[0] - window_len == X_train_shape[0]
        assert train_shape[0] - window_len == y_train_shape[0]
        assert test_shape[0] - window_len == X_test_shape[0]
        assert test_shape[0] - window_len == y_test_shape[0]

        assert X_train_shape[0] - window_len == X_val_shape
        assert y_train_shape[0] - window_len == y_val_shape


def test_method_sequence():
    for s in security_list:
        d_reader = Data_Reader(s, 2019)

        with pytest.raises(Exception):
            d_reader.extract_xy(window_len)
