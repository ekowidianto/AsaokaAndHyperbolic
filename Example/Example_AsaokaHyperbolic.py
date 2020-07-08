"""
Created on Wed July 08 11:18:00 2020

@author: EOWO
"""
import pandas as pd
from sys import path
from AsaokaHyperbolic import *

df_data = pd.read_csv('Example_monitoring_data.csv')

lst_date = list(df_data['Date'])
lst_settlement = list(df_data['Settlement (mm)'])

prediction = AsaokaHyperbolic(lst_date, lst_settlement)

asaoka_ulst_settl, asaoka_doc = prediction.asaoka(day_interval=4, surcharge_date='05-01-2020')

hyperbolic_ulst_settl, hyperbolic_doc = prediction.hyperbolic(surcharge_date='05-01-2020', regression_date='09-01-2020')

prediction.plot_asaoka()

prediction.plot_hyperbolic()


