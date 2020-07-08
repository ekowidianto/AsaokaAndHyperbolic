"""
Created on Wed July 08 11:18:00 2020

@author: EOWO
"""


import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


class AsaokaHyperbolic:
    """
    An observational settlement prediction method based on field monitoring data using Asaoka method (Asaoka, 1978)
     and Hyperbolic method (S.A. Tan, 1994).

    Attributes:
        date (list - timestamp)     : input dates of observed settlement
        settlement (list - float)   : observed settlement values (+ve upward, -ve downward)
        asaoka_result (dict)        : asaoka analysis result in dictionary form which contains
         predicted ultimate settlement, degree of consolidation, slope & intercept of asaoka analysis and a datframe
         containing details of calculated values for asaoka analysis.
        hyperbolic_result (dict)    : hyperbolic analysis result in dictionary form which contains
         predicted ultimate settlement, degree of consolidation, slope & intercept of hyperbolic analysis and a datframe
         containing details of calculated values for asaoka analysis.
    """

    def __init__(self, date_list, settlement_list):
        """
        The constructor for AsaokaHyperbolic class.

        :param date_list            : a list of dates (dd-mm-yyyy; e.g. 23-01-2008) of in string type
        :type                       : list
        :param settlement_list      : a list of monitored settlement taken on the dates
        :type                       : list
        """

        # Initialize Variables
        self.date = pd.to_datetime(date_list, dayfirst=True)
        self.settlement = settlement_list
        assert (len(self.date) == len(self.settlement)), "Date and settlement inputs are not of the same length."

        self.asaoka_surcharge_date = None
        self.hyperbolic_surcharge_date = None
        self.hyperbolic_regression_date = None

        self.asaoka_is_calculated = False
        self.hyperbolic_is_calculated = True

        self.asaoka_result = {}
        self.hyperbolic_result = {}

        # Create a data-frame to contain data
        self.df = pd.DataFrame(data=self.settlement, columns=['Settlement'], index=self.date)
        self.df.index.name = 'Date'
        elapsed_days = list(self.df.index - self.df.index[0])
        elapsed_days = [x.days for x in elapsed_days]
        self.df['Elapsed Days'] = elapsed_days

    def asaoka(self, day_interval, surcharge_date):
        """
        The function to predict ultimate settlement and current degree of consolidation based on Asaoka (Asaoka, 1978)
        method.

        :param day_interval             : number of interval (in days) used in the analysis
        :type day_interval              : int
        :param surcharge_date           : Date (dd/mm/yyyy) of maximum surcharge attained in string format
        :type surcharge_date            : string
        :return ultimate_settlement     : predicted ultimate settlement value
        :return degree_of_consolidation : current degree-of-consolidation of soil
        """

        # Convert input string date (dd-mm-yyyy) to timestamp type
        surcharge_date = pd.to_datetime(surcharge_date, dayfirst=True)
        self.asaoka_surcharge_date = surcharge_date

        # Re-sample data to daily frequency
        df_asaoka = self.df.copy()
        df_asaoka = df_asaoka[['Elapsed Days', 'Settlement']].resample('D').interpolate(method='linear')
        df_asaoka['$S_{n-1}$'] = df_asaoka['Settlement']
        df_asaoka['$S_n$'] = np.append(df_asaoka['Settlement'][1:].values, np.nan)

        # Re-sample data back to day_frequency
        df_asaoka = df_asaoka.loc[surcharge_date:]
        df_asaoka = df_asaoka[['Elapsed Days', 'Settlement', '$S_{n-1}$']].resample(
            str(int(day_interval)) + 'D').interpolate(method='linear')
        df_asaoka['$S_n$'] = np.append(df_asaoka['$S_{n-1}$'][1:].values, np.nan)
        df_asaoka = df_asaoka.dropna()

        # Calculate ultimate settlement and degree of consolidation
        slope, intercept, r_value, p_value, std_err = stats.linregress(-df_asaoka['$S_{n-1}$'], -df_asaoka['$S_n$'])
        ultimate_settlement = intercept / (1 - slope)
        degree_of_consolidation = np.clip(-np.round(self.df['Settlement'][-1] / ultimate_settlement * 100, 2), 0, 100)

        self.asaoka_result['Ultimate Settlement'] = ultimate_settlement
        self.asaoka_result['DOC'] = degree_of_consolidation
        self.asaoka_result['slope'] = slope
        self.asaoka_result['intercept'] = intercept
        self.asaoka_result['df_asaoka'] = df_asaoka.copy()

        self.asaoka_is_calculated = True
        return ultimate_settlement, degree_of_consolidation

    def hyperbolic(self, surcharge_date, regression_date, alpha=0.76):
        """
        The function to predict ultimate settlement and current degree of consolidation based on Asaoka (Asaoka, 1978)
        method.

        :param surcharge_date           : date (dd/mm/yyyy) of maximum surcharge attained in string format
        :type surcharge_date            : string
        :param regression_date          : start of regression Date (dd/mm/yyyy) for the analysis in string format
        :type regression_date           : string
        :param alpha                    : assumed theoretical slope of initial linear segment in Tv/U vs Tv graph
        :type alpha                     : float
        :return ultimate_settlement     : predicted ultimate settlement value
        :return degree_of_consolidation : current degree-of-consolidation of soil
        """

        # Convert input string dates (dd-mm-yyyy) to timestamp type
        surcharge_date = pd.to_datetime(surcharge_date, dayfirst=True)
        regression_date = pd.to_datetime(regression_date, dayfirst=True)
        self.hyperbolic_surcharge_date = surcharge_date
        self.hyperbolic_regression_date = regression_date

        # Resample data to daily frequency
        df_hyperbolic = self.df.copy()
        df_hyperbolic = df_hyperbolic.loc[surcharge_date:]
        df_hyperbolic['Elapsed Days'] = df_hyperbolic['Elapsed Days'] - df_hyperbolic['Elapsed Days'][0]
        df_hyperbolic['$t/S(t)$'] = -(df_hyperbolic['Elapsed Days'].iloc[1:]) / \
                                    (df_hyperbolic['Settlement'].iloc[1:] - df_hyperbolic['Settlement'].iloc[0])

        # Calculate ultimate settlement and degree of consolidation
        slope, intercept, r_value, p_value, std_err = \
            stats.linregress(df_hyperbolic.loc[regression_date:]['Elapsed Days'],
                             df_hyperbolic.loc[regression_date:]['$t/S(t)$'])
        ultimate_settlement = 1 / slope * alpha - df_hyperbolic['Settlement'][0]
        degree_of_consolidation = np.clip(-np.round(self.df['Settlement'][-1] / ultimate_settlement * 100, 2), 0, 100)

        self.hyperbolic_result['Ultimate Settlement'] = ultimate_settlement
        self.hyperbolic_result['DOC'] = degree_of_consolidation
        self.hyperbolic_result['slope'] = slope
        self.hyperbolic_result['intercept'] = intercept
        self.hyperbolic_result['df_hyperbolic'] = df_hyperbolic.copy()

        self.hyperbolic_is_calculated = True
        return ultimate_settlement, degree_of_consolidation

    def plot_asaoka(self, filepath_to_save=None):
        """
        The function plots the results of hyperbolic analysis.
        :param filepath_to_save : filepath to save figure
        :return                 : asaoka matplotlib figure
        """

        assert (self.asaoka_is_calculated is True), "Asaoka analysis has not been run."

        temp = self.asaoka_result['df_asaoka']
        slope = self.asaoka_result['slope']
        intercept = self.asaoka_result['intercept']
        ultimate_settlement = self.asaoka_result['Ultimate Settlement']

        fig = plt.figure(figsize=(8, 16))

        # Asaoka Plot
        ax = plt.subplot(2, 1, 1)
        plt.plot(-temp['$S_{n-1}$'], -temp['$S_n$'], 'bo--', label="Actual")
        plt.plot([0, 5000], [intercept, slope * 5000 + intercept], 'r--', label='$S_{t}$ = ' + str(np.round(slope, 2)) + '$S_{t-1}$ + ' + str(
            np.round(intercept, 2)) + '; Ultimate Settlement = ' + str(np.round(ultimate_settlement, 2)) + ' mm')
        plt.plot([0, 5000], [0, 5000], 'k', label='45° Line')

        ax.annotate(r"   $S_{t}=A\cdot S_{t-1}+B$", xy=(0.01, 0.8), xycoords='axes fraction', fontsize=12)
        ax.annotate(
            'A=' + str(np.round(slope, 2)) + '  B=' + str(np.round(intercept, 2)) + r"   Ult. Settlement$ =\frac{B}{(1-A)}$",
            xy=(0.035, 0.75), xycoords='axes fraction', fontsize=12)

        plt.title('Asaoka Method', fontsize=12)
        plt.xlabel('$S_{t-1}$ [mm]', fontsize=12)
        plt.ylabel('$S_{t}$ [mm]', fontsize=12)
        plt.grid()
        plt.xlim(-temp['$S_{n-1}$'].max() - 100, -temp['$S_{n-1}$'].min() + 100)
        plt.ylim(-temp['$S_{n-1}$'].max() - 100, -temp['$S_{n-1}$'].min() + 100)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(fontsize=11, loc=2)

        # Settlement Plot
        ax = plt.subplot(2, 1, 2)
        plt.plot(self.df.index, -self.df['Settlement'], 'bo-', label='Settlement')
        plt.axvline(self.asaoka_surcharge_date, color='k', Ls='--', label='Surcharge Date')

        plt.title('Soil Settlement', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Settlement [mm]', fontsize=12)
        plt.grid()
        ax.invert_yaxis()
        plt.xticks(fontsize=10, rotation=45)
        plt.yticks(fontsize=10)
        plt.legend(fontsize=11, loc=1)

        if filepath_to_save is None:
            plt.savefig('Output_Asaoka Plot.png')
        else:
            plt.savefig(filepath_to_save+'\\Output_Asaoka Plot.png')

    def plot_hyperbolic(self, filepath_to_save = None):
        """
        The function plots the results of hyperbolic analysis.
        :param filepath_to_save : filepath to save figure
        :return                 : hyperbolic matplotlib figure
        """

        assert (self.asaoka_is_calculated is True), "Asaoka analysis has not been run."

        temp = self.hyperbolic_result['df_hyperbolic']
        slope = self.hyperbolic_result['slope']
        intercept = self.hyperbolic_result['intercept']
        ultimate_settlement = self.asaoka_result['Ultimate Settlement']

        fig = plt.figure(figsize=(8, 16))
        # Hyperbolic plot
        ax = plt.subplot(2, 1, 1)
        plt.plot(temp['Elapsed Days'], temp['$t/S(t)$'], 'bo', label="Actual")
        plt.plot([temp['Elapsed Days'].loc[self.hyperbolic_regression_date:][0], temp['Elapsed Days'][-1]],
                 [slope * temp['Elapsed Days'].loc[self.hyperbolic_regression_date:][0] + intercept,
                  slope * temp['Elapsed Days'][-1] + intercept], 'r--',
                 label='$t/S(t)$ = ' + str(np.round(slope, 6)) + '$t$ + ' + str(np.round(intercept, 6))
                       + '; Ultimate Settlement = ' + str(np.round(ultimate_settlement, 2)) + ' mm')

        ax.annotate("Linear Regression: " + str(temp['Elapsed Days'].loc[self.hyperbolic_regression_date:][0])
                    + " days after surcharge achieved was achieved", xy=(0.01, 0.85),xycoords='axes fraction',
                    fontsize=12)

        plt.title('Hyperbolic Method', fontsize=12)
        plt.xlabel('$t$ [days]', fontsize=12)
        plt.ylabel('$t/S(t)$ [days/mm]', fontsize=12)
        plt.grid()
        plt.ylim(temp['$t/S(t)$'].min(), temp['$t/S(t)$'].max() * 1.25)
        plt.xlim(temp['Elapsed Days'].min(), temp['Elapsed Days'].max() * 1.25)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(fontsize=11, loc=2)

        # Settlement Plot
        ax = plt.subplot(2, 1, 2)
        plt.plot(self.df.index, -self.df['Settlement'], 'bo-', label='Settlement')
        plt.axvline(self.hyperbolic_surcharge_date, color='k', Ls='--', label='Surcharge Date')
        plt.axvline(self.hyperbolic_regression_date, color='r', Ls='--', label='Start of Regression Date')

        plt.title('Soil Settlement', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Settlement [mm]', fontsize=12)
        plt.grid()
        ax.invert_yaxis()
        plt.xticks(fontsize=10, rotation=45)
        plt.yticks(fontsize=10)
        plt.legend(fontsize=11, loc=1)

        if filepath_to_save is None:
            plt.savefig('Output_Hyperbolic Plot.png')
        else:
            plt.savefig(filepath_to_save+'\\Output_Hyperbolic Plot.png')