import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 


class LogisticRegressionScratch:
    def __init__(self , learning_rate = 0.01 , n_iter = 1000 , fit_intercept = True , verbose = False):

        """
        Parameters
        ----------
        self : self refers to the current instance of the class, without self, variables inside __init__ would disappear once it ends 
        learning_rate : float, optional
            Step size parameter, defaults to 0.01
        n_iter : int, optional
            Number of iterations, defaults to 1000
        fit_intercept : bool, optional
            Whether the intercept should be estimated or not. If set to
            False, no intercept will be used in calculations (e.g. data is
            already centered). Defaults to True
        verbose : bool, optional
            Controls the verbosity of the object, defaults to False
        """

        self.lr = learning_rate
        self.epoch = n_iter 
        self.fit_intercept = fit_intercept
        self.ver = verbose 


    def _add_intercept(self , X):

        """
        If fit_intercept is True, this adds a column of ones to the
        beginning of the design matrix X. This is used to represent the
        intercept term in the linear model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data

        Returns
        -------
        X : array-like of shape (n_samples, n_features + 1)
            Design matrix with intercept term
        """

        if self.fit_intercept:
            return np.hstack([np.ones((X.shape[0] , 1)), X]) 
        return X 
    

    def _sigmoid(self , z):

        """
        The sigmoid function maps real numbers to the interval [0,1].

        Parameters
        ----------
        z : array-like of shape (n_samples, n_features)
            Input array

        Returns
        -------
        y : array-like of shape (n_samples, n_features)
            Output array
        """

        return 1 / (1+np.exp(-z)) 
    

    def fit(self,X,y):


        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values
        Returns
        -------
        self : object
            Returns the instance of the class
        """


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

        """
        Predict using the logistic model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Returns predicted values.
        """


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