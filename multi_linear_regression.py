import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score 


data = { 
    "Hours_Studied" : [1 , 2 , 3 , 4 , 5],
    "Attendance" : [60 , 65 , 70 , 75 , 80],
    "Marks" : [30 , 50 , 65 , 80 , 95] 
} 

df = pd.DataFrame(data , index = [1 , 2 , 3 , 4 , 5])

X = df[["Hours_Studied" , "Attendance"]] 
y = df["Marks"] 


X_train,X_test,y_train,y_test = train_test_split(X , y , test_size = 0.2 , random_state = 36) 

model = LinearRegression() 
model.fit(X_train,y_train) 

y_pred = model.predict(X_test) 

#check intercept and coef 
print("Intercept: " , model.intercept_) 
print("Coefficient: " , model.coef_) 

