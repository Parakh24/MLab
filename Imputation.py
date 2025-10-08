#Data Preprocessing 

from sklearn.impute import SimpleImputer 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error 

data = {
    'Study_Hours' : [5 , 8 , np.nan , 2 , 6 , 7 , np.nan , 4], 
    'Attendance' : [90 , 85 , 80 , np.nan , 95 , 88 , 92 , np.nan],
    'Score' : [80 , 85 , 70 , 65 , 30 , 45 , 50 , 70] 
}

df = pd.DataFrame(data) 


X = df[['Study_Hours' , 'Attendance']] 
y = df['Score'] 


X_train,X_test,y_train,y_test = train_test_split(X , y , test_size = 0.2 , random_state = 42) 

imputer = SimpleImputer(strategy = 'mean')   #fills the missing values with the mean of the column 

X_train_imputed = imputer.fit_transform(X_train) 
X_test_imputed = imputer.transform(X_test) 

model = LinearRegression() 
model.fit(X_train_imputed , y_train) 

y_pred = model.predict(X_test_imputed) 

#Evaluate 
mse = mean_squared_error(y_test , y_pred) 





