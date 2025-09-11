import numpy as np                                              #used for vector/dot operations 
from sklearn.datasets import make_regression                    #a helper that generates a synthetic regression dataset
from sklearn.preprocessing import StandardScaler                #imports Standardscaler, used to standardize features(0 mean , unit variance)
from sklearn.metrics import mean_squared_error, r2_score        #imports two evaluation metrices MSE(mean squared error) and R^2 
from sklearn.model_selection import train_test_split 

# this class implements a simple linear regression using gradient descent
class LinearRegressionScratch:  

     def __init__(self , learning_rate = 0.01 , n_iter = 1000 , fit_intercept = True , verbose = False):
          
          """ 
          Parameters  
          ----------
          self : self refers to the current instance of the class,  without self, variables inside __init__ would disappear once it ends 
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
          self.fit_incpt = fit_intercept 
          self.ver = verbose 
            
     #_add_intercept is a convention that it is a private file and users outside the class are not allowed to acces this function
     def _add_intercept(self,X):
              
          """ 
          If fit_intercept is True, this adds a column of ones to the 
          beginning of the design matrix X. This is used to represent the 
          intercept term in the linear model. 
          """

     # self.fit_intercept checks whether bias is present in the equation 
          if self.fit_intercept:
               return np.hstack([np.ones((X.shape[0], 1)), X])   #this concatenates the X(feature) vector with the column vector of one 
          
          return X  #if bias does not exist , we return feature vector itself 
     
     def fit(self , X , y):  

        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target values

        Returns 
        ----------
        self : object
            Returns the instance of the class
        """
    

        X = self._add_intercept(X)        #calling the addintercept function
        m,n = X.shape    
        self.w = np.zeros(n)              #weights are defined in terms of instance so that it could be accessed by other functions calls
             
        for i in range(self.epoch):       #no of iterations to set the correct parameters for weights using batch gradient descent 
             y_pred = X.dot(self.w)       #predicted value of y by the model 
             error = y_pred - y           #error found  
             grad = (2/m)*X.T.dot(error)  #gradient descent
             self.w -= self.lr*grad       #updated weight  

             if self.verbose and i%(self.n_iter//10 or 1) == 0: 
                  loss = mean_squared_error(y,y_pred)
                  print(f"Iter {i:4d} | Loss: {loss:.4f}")

        return self 
                 
     def predict(self, X):

          """
          Predict using the linear model.

          Parameters
          ----------
          X : array-like of shape (n_samples, n_features)
              Samples.

          Returns
          ----------
          y_pred : array of shape (n_samples,)
              Returns predicted values.
          """

          X = self._add_intercept(X)
          return X.dot(self.w) 
     

X, y = make_regression(n_samples=200, n_features=3, noise=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

model = LinearRegressionScratch(lr=0.05, n_iter=1000, verbose=True)

model.fit(X_train, y_train) 

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred)) 

print("R²:", r2_score(y_test, y_pred)) 