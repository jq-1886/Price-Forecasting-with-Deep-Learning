"""Module containing a class to handle plotting"""
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.lines as lines
import matplotlib.pyplot as plt


class Security_Plotter:
    """Class to perform plotting"""

    def __init__(self, date_series, y_actual, y_pred, y_dummy, train_len,
                 window_len, security_str, model_str):
        """
        Initialiser for Security_Plotter class

        Parameters
        ----------
            date_series (array):
                array or array like date values of dates

            y_actual (array):
                array or array like actual price data

            y_pred (array):
                array or array like predicted price data

            y_dummy (array):
                array or array like dummy price data

            train_len (int):
                an integer recording the length of the training dataset

            window_len (int):
                an integer recording the length of the prediction window

            security_str (str):
                a string of the relevant security

            model_str (str):
                a string of the relevant model

        Example
        -------
            Security_Plotter(dates, actual_prices, pred_prices, dummy_prices,
                             250, 30, security_dict, security_str)
        """

        self.date_series = date_series
        self.y_actual = y_actual
        self.y_pred = y_pred
        self.y_dummy = y_dummy
        self.train_len = train_len
        self.window_len = window_len
        self.security_str = security_str
        self.model_str = model_str
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

    def single_plot(self):
        """
        Method to plot y_acutal, y_pred and y_dummy

        Parameters
        ----------
            None

        Example
        -------
            Security_Plotter.plot(250, 30)

        Returns
        -------
            plot of prices, to console or notebook
        """

        # converting to datetime date format and slicing
        date_time = self.date_series.data.date[self.train_len +
                                               self.window_len:]
        # converting the series to datetime using pandas
        series_dates = pd.to_datetime(date_time).dt.date
        # resetting index to 0 based
        series_dates = series_dates.reset_index(drop=True)
        # converting to matplotlib format
        series_dates = mdates.date2num(series_dates)

        # setting YearLocator
        years = mdates.YearLocator()
        # setting MonthLocator
        months = mdates.MonthLocator()
        # setting format to give year and verbose month '2019-Jan'
        d_format = mdates.DateFormatter('%Y-%b')

        # creating our figure and axes
        self.fig, ax = plt.subplots()
        # setting image size inches
        self.fig.set_size_inches(12, 6)
        # plotting the various y values
        ax.plot(series_dates, self.y_actual, label='Acutal Price')
        ax.plot(series_dates, self.y_pred, label='Predicted Price')
        ax.plot(series_dates, self.y_dummy, label='Dummy Price')
        # setting x axis label
        ax.set_xlabel('Date')
        # getting y axis label from .security_dict attribute
        ax.set_ylabel(self.security_dict[self.security_str])
        # setting title
        ax.set_title(self.model_str + ' ' + self.security_str
                     + ' Acutal, Predicted and Dummy Prices')
        # defining legend
        ax.legend()
        # informing matplotlib that x axis contains dates
        ax.xaxis_date()
        # implementing the formatter
        self.fig.autofmt_xdate()

        # setting minor and major locators and format
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(d_format)
        ax.xaxis.set_minor_locator(years)

    def single_plot_extended(self, output_len):
        """
        Method to plot y_acutal, y_pred and y_dummy from an extended
        prediction horizon

        Parameters
        ----------
            output_len (int):
                the length of the prediction window in days

        Example
        -------
            Security_Plotter.plot(250, 30)

        Returns
        -------
            plot of prices, to console or notebook
        """

        # converting to datetime date format and slicing
        date_time = self.date_series.data.date[self.train_len +
                                               self.window_len:]
        # converting the series to datetime using pandas
        series_dates = pd.to_datetime(date_time).dt.date
        # resetting index to 0 based
        series_dates = series_dates.reset_index(drop=True)
        # converting to matplotlib format
        series_dates = mdates.date2num(series_dates)

        # setting YearLocator
        years = mdates.YearLocator()
        # setting MonthLocator
        months = mdates.MonthLocator()
        # setting format to give year and verbose month '2019-Jan'
        d_format = mdates.DateFormatter('%Y-%b')

        # defining y_pred_plot to hold each prediction, initialised to 0s
        self.y_pred_plot = np.zeros((self.y_pred.shape[0],
                                     self.y_pred.shape[0]))

        # a loop to populate y_pred_plot with the relevant element from y_pred
        start_index = 0
        # moving through each prediction stopping before the final prediction
        for i in range(0, self.y_pred.shape[0] - output_len):
            for j in range(0, output_len):
                self.y_pred_plot[i][start_index + j] = self.y_pred[i][j]
            # incrementing start_index to move the first non-zero of element
            #  of y_pred_plot one column to the right for each new row
            start_index += 1

        # converting 0s to nans to allow for easy plotting
        self.y_pred_plot[self.y_pred_plot == 0] = np.nan

        # creating our figure and axes
        self.fig, ax = plt.subplots()
        # setting image size inches
        self.fig.set_size_inches(12, 6)
        # plotting the various y values
        ax.plot(series_dates, self.y_actual, label='Acutal Price')
        ax.plot(series_dates, self.y_dummy, label='Dummy Price')
        # a loop to plot each prediction from the model
        for i in range(0, self.y_pred.shape[0] - output_len):
            ax.plot(series_dates, self.y_pred_plot[i], linestyle='dotted')
        # setting x axis label
        ax.set_xlabel('Date')
        # getting y axis label from .security_dict attribute
        ax.set_ylabel(self.security_dict[self.security_str])
        # setting title
        ax.set_title(self.model_str + ' ' + self.security_str
                     + ' Extended Acutal, Predicted and Dummy Prices')
        # creating a manually entered legend line to denote predictions
        legend_line = lines.Line2D([0], [0], label='Predicted Prices',
                                   linestyle='dotted', color='k')
        # assigning handles and labels
        handles, labels = ax.get_legend_handles_labels()
        # including legend_line
        handles.extend([legend_line])
        # defining legend
        ax.legend(handles=handles)
        # informing matplotlib that x axis contains dates
        ax.xaxis_date()
        # implementing the formatter
        self.fig.autofmt_xdate()

        # setting minor and major locators and format
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(d_format)
        ax.xaxis.set_minor_locator(years)
