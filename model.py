# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
import pickle

dataset = pd.read_csv('C:\\Users\\sushi\\OneDrive\\Documents\\Data Science Exercise Data.csv')

train_dat, test_dat = train_test_split(dataset, test_size=0.2)

X_train = train_dat.iloc[:, 2:8]
X_test = test_dat.iloc[:, 2:8]


y_train = train_dat.iloc[:, -1]
y_test = test_dat.iloc[:, -1]



cvfit = LassoCV(alphas=[0.00465], cv=3).fit(X_train,y_train)
preds = cvfit.predict(X_test)
sum([x**2 for x in y_test-preds])/len(preds)


# Saving model to disk
pickle.dump(cvfit, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[26, 48,15,7,7,33]]))

import os
os.getcwd()