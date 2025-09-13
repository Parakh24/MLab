import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 


# this class implements a simple logistic regression using gradient descent 
class LogisticRegressionScratch:
    def __init__(self , learning_rate = 0.01 , n_iter = 1000 , fit_intercept = True , verbose = False):

        """ 
        Parameters  
        ----------
        self : self refers to the current instance of the class
        learning_rate : float, optional 
            Step size parameter, defaults to 0.01
        n_iter : int, optional
            Number of iterations, defaults to 1000
        fit_intercept : bool, optional
            Whether the intercept should be estimated or not.
            If set to False, no intercept will be used in calculations.
            Defaults to True
        verbose : bool, optional
            Controls the verbosity of the object, defaults to False
        """

        self.lr = learning_rate          # learning rate
        self.epoch = n_iter              # number of iterations
        self.fit_intercept = fit_intercept
        self.ver = verbose 


    def _add_intercept(self , X):
        """ 
        If fit_intercept is True, this adds a column of ones to the 
        beginning of the design matrix X. This represents the bias term.
        """
        if self.fit_intercept:
            return np.hstack([np.ones((X.shape[0] , 1)), X])  # add bias term
        return X 
    

    def _sigmoid(self , z):
        """ 
        The sigmoid function squashes input into the range (0,1),
        representing probability of belonging to class 1.
        """
        return 1 / (1+np.exp(-z)) 
    

    def fit(self,X,y):
        """
        Fits the logistic regression model on training data using gradient descent.
        
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

        X = self._add_intercept(X)      # add intercept term if needed
        m,n = X.shape                   # number of samples (m) and features (n)
        self.w = np.zeros(n)            # initialize weights with zeros

        # gradient descent loop
        for i in range(self.epoch): 
            linear = X.dot(self.w)                  # linear combination
            y_pred = self._sigmoid(linear)          # apply sigmoid to get probability
            grad = (1/m) * X.T.dot(y_pred - y)      # gradient of loss function
            self.w -= self.lr * grad                # update weights

            # print loss occasionally if verbose=True
            if self.ver and i%(self.epoch // 10 or 1) == 0: 
               loss = -(1/m) * np.sum(y*np.log(y_pred+1e-9) + (1-y)*np.log(1-y_pred+1e-9))
               print(f"Iter {i:4d} | Loss: {loss:.8f}") 

        return self 
    

    def predict(self, X): 
        """
        Predict class labels for given samples.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.
        
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels (0 or 1)
        """
        X = self._add_intercept(X)
        return np.where(self._sigmoid(X.dot(self.w)) >= 0.5 , 1 , 0)
    


# ----- Dataset -----
# Feature (X): study hours
# Target (y): pass(1) / fail(0)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])  
y = np.array([0, 0, 0, 0, 1, 1, 1, 1])                  

# ----- Train-Test Split -----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----- Feature Scaling -----
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ----- Model Training -----
model = LogisticRegressionScratch(learning_rate=0.1, n_iter=1000, verbose=True)
model.fit(X_train, y_train)

# ----- Predictions -----
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# ----- Visualization -----
plt.scatter(X, y, c=y, cmap='bwr', edgecolor='k') 
plt.xlabel("Hours studied") 
plt.ylabel("Pass/Fail") 
plt.title("Logistic Regression Dataset") 
plt.show() 
