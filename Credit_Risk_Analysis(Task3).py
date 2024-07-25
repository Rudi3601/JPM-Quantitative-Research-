"""Importing necessary libraries"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report


"""Reading CSV file"""

data = pd.read_csv('Task 3 and 4_Loan_Data.csv')

print(data)