# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 19:26:31 2021

@author: niezn
"""

import pandas as pd
import csv
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

dataset=pd.read_csv('danepyth22.csv')
plt.scatter(dataset['year'],dataset['suicides_no'])
km=KMeans(n_clusters=4)
y_predicted=km.fit_predict(dataset[['year','suicides_no']])
dataset['cluster']=y_predicted
dataset1=dataset[dataset.cluster==0]
dataset2=dataset[dataset.cluster==1]
dataset3=dataset[dataset.cluster==2]
dataset4=dataset[dataset.cluster==3]

plt.scatter(dataset1['year'],dataset1['suicides_no'],color='green')
plt.scatter(dataset2['year'],dataset2['suicides_no'],color='red')
plt.scatter(dataset3['year'],dataset3['suicides_no'],color='black')
plt.scatter(dataset4['year'],dataset4['suicides_no'],color='blue')

plt.xlabel('year')
plt.ylabel('suicides_no')
plt.legend()

pred=pd.DataFrame(dataset)
print(pred)
pred.to_csv("cluster.csv")
plt.title('Klastrowanie')