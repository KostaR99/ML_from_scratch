import numpy as np
from collections import Counter
def eucledian(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    
    def __init__(self,k=3):
        self.k=k
    
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y


    def predict(self,X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
    

    def _predict(self,x):
        distance = [eucledian(x,x_train) for x_train in self.X_train]
        k_indeksi = np.argsort(distance)[:self.k]
        k_najbljize_klase = [self.y_train[i] for i in k_indeksi]

        najcesci = Counter(k_najbljize_klase).most_common(1);

        return najcesci[0][0]
