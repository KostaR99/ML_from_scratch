import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X,y = datasets.make_regression(n_samples = 100,n_features=1,noise=20,random_state=4)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1234)

from LinearRegression import LinearRegression

regresor = LinearRegression(lr = 0.01)
regresor.fit(X_train,y_train)

predikcije = regresor.predict(X_test)
#mean squared error
def MSE(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)

mse=MSE(y_test,predikcije)

print(mse)  

y_pred_linija = regresor.predict(X)
fig = plt.figure(figsize=(8,6))
m1=plt.scatter(X_train,y_train,s=10)
m2=plt.scatter(X_test,y_test,s=10)
plt.plot(X,y_pred_linija,color="black",linewidth=2,label="predikcija")
plt.show()