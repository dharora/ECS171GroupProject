from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split as split
from sklearn import preprocessing as pre
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import numpy as np
import pickle

# Load dataset
df = pd.read_csv('./data/mushrooms.csv')

# Drop unimportant features
df.drop(['gill-attachment'], axis=1, inplace=True)
df.drop(['stalk-shape'], axis=1, inplace=True)
df.drop(['stalk-root'], axis=1, inplace=True)
df.drop(['stalk-surface-below-ring'], axis=1, inplace=True)
df.drop(['stalk-color-below-ring'], axis=1, inplace=True)
df.drop(['veil-type'], axis=1, inplace=True)
df.drop(['veil-color'], axis=1, inplace=True)
df.drop(['ring-number'], axis=1, inplace=True)

# Splitting data
X, y = df.iloc[:, 1:], df.iloc[:, 0] 
X_train, X_test, y_train, y_test = split(X, y, random_state=42)

# Label encoding
label   = pre.LabelEncoder()
y_train = label.fit_transform(y_train)
y_test  = label.fit_transform(y_test)

# One-hot encoding
hot = pre.OneHotEncoder()
hot.fit(X_train)
X_train = hot.transform(X_train).toarray()
X_test  = hot.transform(X_test).toarray()

model = CategoricalNB()

# Fit model on train set 
model.fit(X_train, y_train)

# Generate predictions on train/test sets
y_pred_train, y_pred_test = model.predict(X_train), model.predict(X_test)

pickle.dump(model, open('model.pkl','wb'))

print(X_train)