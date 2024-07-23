"""Importing all the libraries"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from matplotlib import pyplot as plt


"""Loading data"""

data = pd.read_csv('Nat_Gas.csv', parse_dates = ['Dates'])
prices = data['Prices'].values                                          #extracting 'prices' as a numpy array
dates = data['Dates'].values                                            #extracting 'dates' as a numpy array



"""Defining Simple Linear Regression Function"""

def simple_regression(x, y):
    xbar = np.mean(x)
    ybar = np.mean(y)
    slope = np.sum((x -  xbar) * (y - ybar)) / np.sum((x - xbar) ** 2)
    intercept = ybar - slope * xbar
    return slope, intercept


"""Converting Dates to Days from Start Date"""

start_date = date(2020, 10, 31)
days_from_start = [(day - pd.Timestamp(start_date)).days for day in dates]              #Converts each date to the number of days since the start date


"""Calculting Linear Regression Slope and Intercept"""

time = np.array(days_from_start)                                                        #Conversts the list of days from start date to a numpy array
slope, intercept = simple_regression(time, prices)


"""Calculate Sin and Cosine values for time and prices"""

sin_prices = prices - (time * slope + intercept)
sin_time = np.sin(time * 2 * np.pi / 365)
cos_time = np.cos(time * 2 * np.pi / 365)


"""Bilinear Regression"""

def bilinear_regression(y, x1, x2):
    slope1 = np.sum(y * x1) / np.sum(x1 ** 2)
    slope2 = np.sum(y * x2) / np.sum(x2 ** 2)
    return slope1, slope2


"""Calculating Amplitude and Phase Shift"""

slope1, slope2 = bilinear_regression(sin_prices, sin_time, cos_time)
amplitude = np.sqrt(slope1 ** 2 + slope2 ** 2)
shift = np.arctan2(slope2, slope1)



"""Interpolation Function"""

def interpolate(date):
    days = (date - pd.Timestamp(start_date)).days
    return amplitude * np.sin(days * 2 * np.p1 / 365 + shift) + days * slope + intercept



"""Plotting the estimate for the entire dataset"""

continuous_dates = pd.date_range(start = pd.Timestamp(start_date), end = pd.Timestamp('2025-09-30'), freq = 'D')
plt.plot(continuous_dates, [interpolate(date) for date in continuous_dates], label = 'Smooted Estimate')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Natural Gas Prices')
plt.legend()
plt.show()



"""Defining the Pricing Model"""

def price_contract(injection_dates, withdrawal_dates, injection_volumes, withdrawal_volumes, storage_cost_per_month, injection_cost_per_mmbtu, withdrawal_cost_per_mmbtu, max_storage_volume):
    total_value = 0
    storage_volume = 0

    for inject_date, volume in zip(injection_dates, injection_volumes):
        price_at_injection = interpolate(pd.Timestamp(inject_date))                     #Calculates the price at the injection date by interpolating
        injection_cost = injection_cost_per_mmbtu * volume                              #Calculates the total injection cost
        total_value -= (price_at_injection * volume) + injection_cost                   #Subtracts the total injection cost from the total pricing value
        storage_volume += volume                                                        #Increase storage_volume by injected volume


    for withdraw_date, volume in zip(withdrawal_dates, withdrawal_volumes):
        price_at_withdrawal = interpolate(pd.Timestamp(withdraw_date))                  #Calculates the price at the withdrawal date by interpolating
        withdrawal_cost = withdrawal_cost_per_mmbtu * volume                            #Calculates the total withdrawal cost
        total_value += (price_at_withdrawal * volume) + withdrawal_cost                 #Adds the total withdrawal cost from the total pricing value
        storage_volume -= volume                                                        #Decrease storage_volume by witdrawal volume


    months_in_storage = (pd.Timestamp(withdrawal_dates[-1]) - pd.Timestamp(injection_dates[0])).days / 30
    storage_cost = storage_cost_per_month * months_in_storage

    total_value -= storage_cost

    return total_value



"""Testing the Pricing Model"""

injection_dates = ['2023-06-01', '2023-07-01']
withdrawal_dates = ['2023-12-01', '2024-01-01']
injection_volumes = [500000, 500000]
withdrawal_volumes = [500000, 500000]
storage_cost_per_month = 100000
injection_cost_per_mmbtu = 10000
withdrawal_cost_per_mmbtu = 10000
max_storage_volume = 1000000

contract_value = price_contract(injection_dates, withdrawal_dates, injection_volumes, withdrawal_volumes, storage_cost_per_month, injection_cost_per_mmbtu, withdrawal_cost_per_mmbtu, max_storage_volume)
print(f"Contract'svalue is ${contract_value:.2f}")


