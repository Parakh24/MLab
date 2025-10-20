import pandas as pd 
from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score #metrics to evaluate model performance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 


data = { 
  "Study_Hours" : [1 , 2 , 3 , 4 , 5],
  "Marks": [20 , 40 , 60 , 80 , 100] 
}

df = pd.DataFrame(data , index = [1,2,3,4,5]) 


X = df[["Study_Hours"]]  #features must be 2D 
y = df["Marks"] #labels must be 1D 


X_train,X_test,y_train,y_test = train_test_split(X , y , test_size = 0.3 , random_state = 36) 

model = LinearRegression() 
model.fit(X, y) 

#Predict 
y_pred = model.predict(X_test) 

#Evaluate 
print("mean squared error: " , mean_squared_error(y_test , y_pred)) 
print("mean_absolute_error: " , mean_absolute_error(y_test , y_pred)) 
print("r2_score: " , r2_score(y_test , y_pred)) 


