import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 


class LogisticRegressionScratch:
    def __init__(self , learning_rate = 0.01 , n_iter = 1000 , fit_intercept = True , verbose = False):
        self.lr = learning_rate
        self.epoch = n_iter 
        self.fit_intercept = fit_intercept
        self.ver = verbose 


    def _add_intercept(self , X):
        if self.fit_intercept:
            return np.hstack([np.ones((X.shape[0] , 1)), X]) 
        return X 
    

    def _sigmoid(self , z):
        return 1 / (1+np.exp(-z)) 
    

    def fit(self,X,y):

        X = self._add_intercept(X) 
        m,n = X.shape 
        self.w = np.zeros(n) 

        for i in range(self.epoch): 
            linear = X.dot(self.w) 
            y_pred = self._sigmoid(linear) 
            grad = (1/m) * X.T.dot(y_pred-y) 
            self.w -= self.lr * grad 


            if self.ver and i%(self.epoch // 10 or 1) == 0: 
               loss = -(1/m) * np.sum(y*np.log(y_pred+1e-9) + (1-y)*np.log(1-y_pred+1e-9))
               print(f"Iter {i:4d} | Loss: {loss:.8f}") 

        return self 
    
    def predict(self, X): 
        X = self._add_intercept(X) 
        return np.where(self._sigmoid(X.dot(self.w)) >= 0.5 , 1 , 0)
    

np.random.seed(42)

X = np.random.randn(200 , 2)

y = (X[: , 0] + X[: , 1] > 0).astype(int) 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

model = LogisticRegressionScratch(learning_rate=0.1, n_iter=1000, verbose=True)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

plt.scatter(X[:,0], X[:,1], cmap='bwr')

plt.show()