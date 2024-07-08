"""Importing all the libraries"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta


"""Loading data"""

data = pd.read_csv('Nat_Gas.csv', parse_dates = ['Dates'])
data['Prices'] = data['Prices'].astype(float)



"""Initializing predictor and dependent values for Linear Regression"""

X = data['Dates'].map(datetime.toordinal).values.reshape(-1, 1)
y = data['Prices'].values

last_date = data['Dates'].max()
future_dates = []

