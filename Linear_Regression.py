import numpy as np                                              #used for vector/dot operations 
from sklearn.datasets import make_regression                    #a helper that generates a synthetic regression dataset
from sklearn.preprocessing import StandardScaler                #imports standardscaler, used to standardize features(0 mean , unit variance)
from sklearn.metrics import mean_squared_error, r2_score        #imports two evaluation metrices MSE(mean squared error) and R^2 


class LinearRegressionScratch: 
    