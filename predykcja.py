# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 15:00:43 2021

@author: niezn
"""

import csv
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import mkl
import pycaret.clustering as py
dataset=pd.read_csv('danepyth22.csv')
dataset1 = py.get_clusters(dataset, num_clusters=3, ignore_features=['country'])
#kmeans=create_model('kmeans',num_clusters=5)



X=dataset[['population']].astype(float)
Y=dataset['suicides_no'].astype("float")

X.fillna(X.mean(), inplace=True)
Y.fillna(Y.mean(), inplace=True)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=101)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
predictions


plt.scatter(X_test, y_test,  color='black')
plt.plot(X_test, predictions, color='blue', linewidth=3)
plt.legend()
plt.title('Predykcja')

plt.xticks(())
plt.yticks(())
plt.xlabel('population')
plt.ylabel('sugerowana liczba samobójstw')
plt.grid()




plt.xticks(())
plt.yticks(())

plt.show()



pred=pd.DataFrame(predictions)
print(pred)
pred.to_csv("pred.csv")



mse = mean_squared_error(y_test, predictions, squared=False)