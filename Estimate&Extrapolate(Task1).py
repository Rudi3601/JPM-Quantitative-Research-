"""importing libraries"""

import pandas as pd                                                     #for data manipulaton & analysis
import numpy as np                                                      #for numerical operations
import matplotlib.pyplot as plt                                         #for data visualization
from sklearn.linear_model import LinearRegression                       #for Linear Regression model
from datetime import datetime, timedelta                                #for handling date and time operations


"""Loading the data"""

data = pd.read_csv('Nat_Gas.csv', parse_dates = ['Dates'])
data['Prices'] = data['Prices'].astype(float)

#print(data)


"""Visualizing Historical Data"""

plt.figure(figsize=(12,6))
plt.plot(data['Dates'], data['Prices'], marker='o', linestyle='-', color='b')
plt.title('Historical Natural Gas Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.show()



"""Initializing the predictor vand dependent values for Linear Regression"""


X = data['Dates'].map(datetime.toordinal).values.reshape(-1, 1)     #Converst dates into ordinal form and reshapes the array into a column vector
y = data['Prices'].values                                           #Extracts Prices column as the target variable




"""Modeling the price for any given date for historical data"""

def estimate_price(date_str):                                   #function that takes in date string as an input
    date = datetime.strptime(date_str, '%Y-%m-%d')              #converts the date string into datetime object
    date_ordinal = np.array([[date.toordinal()]])               #converts the date into its ordinal form and puts in a 2D array for regression model


    """Fitting the linear regression model"""

    model = LinearRegression().fit(X, y)                                #Fits linear regression model and takes in input of dates in ordinal form and prices as the target variable

    """Getting the estimated price for the provided date"""
    
    estimated_price = model.predict(date_ordinal)
    #print(estimated_price[0])

    return estimated_price[0]



"""Extrapolating the data for the next year"""


last_date = data['Dates'].max()                             #Gets the latest date from our data
future_dates = []                                           #Array for future dates

for i in range(1, 13):                                      #this loop generates future dates for the next 12 months with the assumption that a month has 30 days on average
    future_date = last_date + timedelta(days=30*i)
    future_dates.append(future_date)


future_dates_ordinals = np.array([date.toordinal() for date in future_dates]).reshape(-1, 1)             #converts the dates into its ordinal form and puts in a 2D array for regression model


future_prices = LinearRegression().fit(X, y).predict(future_dates_ordinals)                              #Using LinearRegression model to predict prices for the future



plt.figure(figsize=(12,6))
plt.plot(future_dates, future_prices, marker='o', linestyle='-', color='r', label='Extrapolated Prices')
plt.title('Extrapolated Natural Gas Prices')
plt.xlabel('Future Dates')
plt.ylabel('Future Prices')
plt.grid(True)
plt.show()





#input_date = '2023-06-03'
#estimate_price(input_date)