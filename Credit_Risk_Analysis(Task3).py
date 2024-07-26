"""Importing necessary libraries"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report


"""Reading CSV file"""

data = pd.read_csv('Task 3 and 4_Loan_Data.csv')

"""Dropping customer id as I believe it will be irrelevant to our prediction"""

data.drop(columns = ['customer_id'], inplace = True)


"""Filling missing values (if any) with suitable mean or median"""

data['credit_lines_outstanding'].fillna(data['credit_lines_outstanding'].median()[0], inplace = True)

data['loan_amt_outstanding'].fillna(data['loan_amt_outstanding'].mean()[0], inplace = True)

data['total_debt_outstanding'].fillna(data['total_debt_outstanding'].mean()[0], inplace = True)

data['income'].fillna(data['income'].mean()[0], inplace = True)

data['years_employed'].fillna(data['years_employed'].mean()[0], inplace = True)

data['fico_score'].fillna(data['fico_score'].mean()[0], inplace = True)

data['default'].fillna(data['default'].mode()[0], inplace = True)


"""Splitting the features from the data"""

features = data["credit_lines_outstanding", "loan_amt_outstanding", 'total_debt_outstanding', 'income', 'years_employed', 'fico_score']


"""Splitting target variable from the data"""

target = data['default']



"""Scaling the features"""

scaler = StandardScaler()
features = scaler.fit_transform(features)



