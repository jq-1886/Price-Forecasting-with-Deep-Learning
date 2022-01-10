"""Module containing a base model"""
import numpy as np


class Base_Model:
    """Class to perform base/dummy modelling"""

    def __init__(self, input_data, window_horizon):
        """
        Initialiser for Base_Model class

        Parameters
        ----------
            input_data (array):
                array or array like X values to generate predictions

            window_horizon (int):
                X value time step length, predicting 1 time step ahead

        Example
        -------
            Base_Model(X_test, 30)
        """

        self.input_data = input_data
        self.window_horizon = window_horizon

    def predict_y(self, test_len):
        """
        Method to predict y from X, where y is the mean of window_horizon time
        steps of X

        Parameters
        ----------
            test_len (int):
                length of test dataset, equivalent to number of predictions to
                be made

        Example
        -------
            Base_Model.predict_y(150)

        Returns
        -------
            y_pred as class attribute
        """

        # initial definition of y_pred attribute as empty list
        self.y_pred = []

        # converting .input_data attribute to an array with 2 dimensions
        X_data = np.array(self.input_data, ndmin=2)

        # iterating through X_data and appending predicted values to y_pred
        for i in range(0, test_len):
            self.y_pred.append(np.mean(X_data[i]))

        # converting y_pred to an array with 2 dimensions
        self.y_pred = np.array(self.y_pred, ndmin=2)
        # reshaping to enforce shape constraints
        self.y_pred = np.reshape(self.y_pred, (test_len, 1))
