import numpy as np                                              #used for vector/dot operations 
from sklearn.datasets import make_regression                    #a helper that generates a synthetic regression dataset
from sklearn.preprocessing import StandardScaler                #imports standardscaler, used to standardize features(0 mean , unit variance)
from sklearn.metrics import mean_squared_error, r2_score        #imports two evaluation metrices MSE(mean squared error) and R^2 

# this class implements a simple linear regression using gradient descent
class LinearRegressionScratch: 

     def __init__(self , learning_rate = 0.01 , n_iter = 1000 , fit_intercept = True , verbose = False):
          
          """
          Parameters
          ----------
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
          

          self.learning_rate = learning_rate 
          self.n_iter = n_iter 
          self.fit_intercept = fit_intercept
          self.verbose = verbose 

     def _add_intercept(self,X):
          
          """ 
          If fit_intercept is True, this adds a column of ones to the 
          beginning of the design matrix X. This is used to represent the 
          intercept term in the linear model. 
          """

          if self.fit_intercept:
               return np.hstack([np.ones((X.shape[0], 1)), X])  