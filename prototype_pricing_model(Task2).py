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

for i in range(1, 13):
    future_date = last_date + timedelta(days=30*i)
    future_dates.append(future_date)


future_dates_ordinal = np.array([date.tooridnal() for date in future_dates]).reshape(-1, 1)

future_prices = LinearRegression().fit(X, y).predict(future_dates_ordinal)

def future_day_price(day_str):
    day = datetime.strptime(day_str, '%Y-%m-%d')
    day_ordinal = day.toordinal()

    return future_prices[day_ordinal]
    

