#Part_2 of Data Preprocssing 
#Regression 
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression 
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error  


data = {
    'Study_Hours': [1 , 2 , np.nan , 3 , np.nan],
    'Pass-Fail':[np.nan , 4  , 6 , np.nan , 9] 
} 

df = pd.DataFrame(data) 

X = df[['Study_Hours']]
y = df['Pass-Fail'] 

X_train,X_test,y_train,y_test = train_test_split(X , y , test_size = 0.2 , random_state = 36) 

imputer = KNNImputer(n_neighbors = 2)   #KNN checks the nearest two_rows and then calculates the average of two columns

X_train_imputed = imputer.fit_transform(X_train) 
X_test_imputed = imputer.transform(X_test)  

model = LinearRegression()  

model.fit(X_train_imputed , y_train)  

y_pred = model.predict(X_test) 

mse = mean_squared_error()





