import numpy as np
import pandas as pd
import warnings;warnings.simplefilter('ignore')
import csv

class RegularizedLogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False,reg=10):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.reg = reg

    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    def add_intercept(self,X):
        intercept=np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def loss(self,h,y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()    
   
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * (gradient + self.reg*self.theta/y.size)
            
        
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.add_intercept(X)
    
        return self.sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold):
        return self.predict_prob(X) >= threshold


##PUTTING LOGISTIC REGRESSION iNTO PLAY
##IMPORT DATA 

X_train=pd.read_csv("data/X_train.csv")
y_train=np.array(pd.read_csv("data/y_train.csv")).flatten()
X_test=pd.read_csv("data/X_test.csv")

model=RegularizedLogisticRegression()
model.fit(X_train,y_train)

###PREDICT THE VALUES
y_pred= [int(round(x)) for x in model.predict(X_test,.10)]
print(y_pred)

###GET THE ACCURACY OF THE MODEL
accuracy=((y_pred.count(0)/31)+(y_pred.count(1)/31))/2
print("Accuracy of the model is:",accuracy*100)

####EXPORT DATA INTO CSV FORMAT
submission=pd.DataFrame({     
    "Prediction":y_pred 
})
submission.to_csv("data/Name_StudentID.csv",index=False)