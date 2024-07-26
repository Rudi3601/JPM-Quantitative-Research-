"""Importing necessary libraries"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


"""Reading CSV file"""

data = pd.read_csv('Task 3 and 4_Loan_Data.csv')

"""Dropping customer id as I believe it will be irrelevant to our prediction"""

data.drop(columns = ['customer_id'], inplace = True)


"""Filling missing values (if any) with suitable mean or median"""

data['credit_lines_outstanding'].fillna(data['credit_lines_outstanding'].median(), inplace = True)

data['loan_amt_outstanding'].fillna(data['loan_amt_outstanding'].mean(), inplace = True)

data['total_debt_outstanding'].fillna(data['total_debt_outstanding'].mean(), inplace = True)

data['income'].fillna(data['income'].mean(), inplace = True)

data['years_employed'].fillna(data['years_employed'].mean(), inplace = True)

data['fico_score'].fillna(data['fico_score'].mean(), inplace = True)

data['default'].fillna(data['default'].mode()[0], inplace = True)


"""Splitting the features from the data"""

features = data[["credit_lines_outstanding", "loan_amt_outstanding", 'total_debt_outstanding', 'income', 'years_employed', 'fico_score']]


"""Splitting target variable from the data"""

target = data['default']



"""Scaling the features"""

scaler = StandardScaler()
features = scaler.fit_transform(features)

"""Splitting data into train and test sets"""
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state = 42)


"""Initializing three models: SVM, MLP, and RF"""

param_grid_svm = {
    "kernel": ["rbf", "linear"],
    "C": [0.1, 1, 10],
    "gamma": [0.01, 0.1, 1]
}

svm_model = GridSearchCV(SVC(probability = True), param_grid_svm, cv = 5, scoring = 'accuracy', n_jobs =- 1)
svm_model.fit(X_train, y_train)



mlp_model = MLPClassifier(hidden_layer_sizes = (32, 20, 2), activation = 'relu', solver = 'adam', random_state = 42)
mlp_model.fit(X_train, y_train)



param_grid_rf = {
    "n_estimators": [10, 50, 100],
    "max_depth": [3, 5, 7],
    "criterion": ["gini", "entropy"]
}

rf_model = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv = 5, scoring = 'accuracy', n_jobs =- 1)
rf_model.fit(X_train, y_train)




"""Getting models' accuracies and best parameters"""


svm_accuracy = accuracy_score(y_test, svm_model.predict(X_test))
print('SVM Accuracy:', svm_accuracy)

best_params_svm = svm_model.best_params_
print("Best Parameters for SVM:", best_params_svm)


mlp_accuracy = accuracy_score(y_test, mlp_model.predict(X_test))
print("MLP Accuracy:", mlp_accuracy)



rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
print('RF Accuracy:', rf_accuracy)

best_params_rf = rf_model.best_params_
print("Best Parameters for RF:", best_params_rf)


"""Function to calculate predicted expected loss"""

def calculate_expected_loss(model, features):
    features_scaled = scaler.transform([features])      #Scaling the features
    
    pd = model.predict_proba(features_scaled)[0][1]         #Predicting the probability of default

    print("Probability of defaulting is:", pd)

    loan_amount = features[1]
    recovery_rate = 0.10
    expected_loss = loan_amount * pd * (1 - recovery_rate)

    return expected_loss



"""Test Example"""

example_features = [1, 10000, 150000, 60000, 3, 650]


expected_loss_svm = calculate_expected_loss(svm_model, example_features)
print(f'Expected Loss using SVM Model: ${expected_loss_svm:.2f}')

expected_loss_mlp = calculate_expected_loss(mlp_model, example_features)
print(f'Expected Loss using MLP Model: ${expected_loss_mlp:.2f}')

expected_loss_rf = calculate_expected_loss(rf_model, example_features)
print(f'Expected Loss using RF Model: ${expected_loss_rf:.2f}')




