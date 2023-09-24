import numpy as np
from collections import Counter
def euclidean_distance(x1,x2):
    distance=np.sqrt(np.sum((x1-x2)**2))
    return distance
class KNN:
    def __init__(self,k=3):
        self.k=k
    def fit(self,X,y):
        self.X_train=X
        self.y_train=y
    def predict(self,X):
        predictions=[self.predict_(x) for x in X]
        return predictions
    def predict_(self,x):
        distances=[euclidean_distance(x,x_train) for x_train in self.X_train]

        #get the closest k
        k_indices=np.argsort(distances)[:self.k]
        k_nearest_labels=[self.y_train[i] for i in k_indices]

        #majority vote
        most_common=Counter(k_nearest_labels).most_common()
        return most_common[0][0]

from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
iris=datasets.load_iris()
x,y=iris.data,iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
clf=KNN(k=5)
clf.fit(x_train,y_train)
predictions=clf.predict(x_test)
print(predictions)
acc=np.sum(predictions==y_test)/len(y_test)
print(acc)