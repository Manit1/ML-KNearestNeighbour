# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 09:25:04 2018

@author: Manit
"""
'''k-nearest neighbour
    ALGORITHM
        step1:choose the number of k numbers
        step2:take the k nearest of the new data point according to the euclidian distance
        step3:among these k neighbours count the number of data points in each category
        step4:assign the new data point to the category where it encountered the most neighbours'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('Social_Network_Ads.csv')

x=data.iloc[:,2:4].values
y=data.iloc[:,4].values


from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


#fitting knn to the training dataset
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(x_train,y_train)





#predicting the test set result
y_pred=classifier.predict(x_test)
#the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)


from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()