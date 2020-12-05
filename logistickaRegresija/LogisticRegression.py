import numpy as np
class LogisticRegression:
    def __init__(self,lr=0.001,n_iters=1000):
        self.lr = lr
        self.n_iters=n_iters
        self.weights = None
        self.bias = None


    def fit(self,X,y):
        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        #gradijentni spust

        for _ in range(self.n_iters):
            linearni_model = np.dot(X,self.weights)+self.bias
            y_prediktovano = self._sigmoid(linearni_model)

            #weight update
            dw = (1/n_samples) * np.dot(X.T,(y_prediktovano-y))
            db = (1/n_samples) * np.sum(y_prediktovano-y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linearni_model = np.dot(X,self.weights)+self.bias
        y_prediktovano = self._sigmoid(linearni_model)
        y_prediktovane_klase = [1 if i>0.5 else 0 for i in y_prediktovano]
        return y_prediktovane_klase

    def _sigmoid(self,X):
        return 1/(1+np.exp(-X))