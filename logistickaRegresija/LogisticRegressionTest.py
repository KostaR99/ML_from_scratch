from operator import imod
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from LogisticRegression import LogisticRegression

bc = datasets.load_breast_cancer()

X, y = bc.data, bc.target

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1234)

def accuracy(y_true,y_pred):
    return np.sum(y_true==y_pred)/len(y_true)

regresor = LogisticRegression(lr=0.01,n_iters=10000)
regresor.fit(X_train,y_train)
predikcije = regresor.predict(X_test)

print("Preciznost: ",accuracy(y_test,predikcije))