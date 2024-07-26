"""Importing Libraries"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.linear_model import LogisticRegression


"""Reading Data CSV File"""

data = pd.read_csv('Task 3 and 4_Loan_Data.csv')

"""Dropping customer id as I believe it will be irrelevant to our prediction"""

data.drop(columns = ['customer_id'], inplace = True)


"""Filling missing values (if any) with suitable mean or median"""

"""data['credit_lines_outstanding'].fillna(data['credit_lines_outstanding'].median(), inplace = True)

data['loan_amt_outstanding'].fillna(data['loan_amt_outstanding'].mean(), inplace = True)

data['total_debt_outstanding'].fillna(data['total_debt_outstanding'].mean(), inplace = True)

data['income'].fillna(data['income'].mean(), inplace = True)

data['years_employed'].fillna(data['years_employed'].mean(), inplace = True)"""

data['fico_score'].fillna(data['fico_score'].mean(), inplace = True)

data['default'].fillna(data['default'].mode()[0], inplace = True)


"""Extracting feature and target variables from the data"""

fico_scores = data['fico_score']                #feature 
defaults = data['default']                      #target


"""Splitting data into train and test sets"""

X_train, X_test, y_train, y_test = train_test_split(fico_scores, defaults,test_size = 0.2, random_state = 42)
X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)

n_buckets = 10


"""Function that evaluates model performance"""

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilitites = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, probabilitites)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return roc_auc, accuracy, report



"""Model setup and evaluation on equal-width buckets"""

equal_width = KBinsDiscretizer(n_bins = n_buckets, encode = 'ordinal', strategy = 'uniform')
X_train_equal_width = equal_width.fit_transform(X_train)
X_test_equal_width = equal_width.transform(X_test)

log_reg = LogisticRegression(random_state=0, solver='liblinear', tol=1e-5, max_iter=10000)
roc_auc, accuracy, report = evaluate_model(log_reg, X_train_equal_width, X_test_equal_width, y_train, y_test)

print(f'ROC AUC for equal-width buckets: {roc_auc}')
print(f'Accuracy for equal-width buckets: {accuracy}')
print(f'Classification Report for equal-width buckets: \n{report}')




"""Model setup and evaluation on equal-frequency buckets"""

equal_freq = KBinsDiscretizer(n_bins = n_buckets, encode = 'ordinal', strategy = 'quantile')
X_train_equal_freq = equal_freq.fit_transform(X_train)
X_test_equal_freq = equal_freq.transform(X_test)

log_reg = LogisticRegression(random_state=0, solver='liblinear', tol=1e-5, max_iter=10000)
roc_auc, accuracy, report = evaluate_model(log_reg, X_train_equal_freq, X_test_equal_freq, y_train, y_test)

print(f'ROC AUC for equal-frequency buckets: {roc_auc}')
print(f'Accuracy for equal-frequency buckets: {accuracy}')
print(f'Classification Report for equal-frequency buckets: \n{report}')




"""Model setup and evaluation on k-means buckets"""

k_means = KBinsDiscretizer(n_bins = n_buckets, encode = 'ordinal', strategy = 'kmeans')
X_train_k_means = k_means.fit_transform(X_train)
X_test_k_means = k_means.transform(X_test)

log_reg = LogisticRegression(random_state=0, solver='liblinear', tol=1e-5, max_iter=10000)
roc_auc, accuracy, report = evaluate_model(log_reg, X_train_k_means, X_test_k_means, y_train, y_test)

print(f'ROC AUC for k-means buckets: {roc_auc}')
print(f'Accuracy for k-means buckets: {accuracy}')
print(f'Classification Report for k-means buckets: \n{report}')




