# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 01:33:18 2017

@author: SAI
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')

X = dataset.iloc[:, [3,4]].values

# using elbow method to find the optimum number of clusters

from sklearn.cluster import KMeans

wcss=[]
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Applying Kmeans

kmeans = KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(X)

#Visualisation

plt.scatter (X[y_kmeans== 0,0] ,X[y_kmeans== 0,1], s=100,c='red',     label='cluster1')
plt.scatter (X[y_kmeans== 1,0] ,X[y_kmeans== 1,1], s=100,c='blue',    label='cluster2')
plt.scatter (X[y_kmeans== 2,0] ,X[y_kmeans== 2,1], s=100,c='green',   label='cluster3')
plt.scatter (X[y_kmeans== 3,0] ,X[y_kmeans== 3,1], s=100,c='cyan',    label='cluster4')
plt.scatter (X[y_kmeans== 4,0] ,X[y_kmeans== 4,1], s=100,c='magenta', label='cluster5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow', label='Centroid' )
plt.title('Cluster of clients')
plt.xlabel('Annual income in ($)')
plt.ylabel('Spending score in (1-100)')
plt.legend()
plt.show()