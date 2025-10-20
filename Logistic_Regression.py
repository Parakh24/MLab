import pandas as pd 
from sklearn.metrics import accuracy_score , precision_score , f1_score , recall_score , confusion_matrix 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 


#Dataset 
data = { 
    "Study_Hours" : [1 , 2 , 3 , 4 , 5 , 6],
    "Pass_Fail" : [0 , 0 , 0 , 1 , 1 , 1]
} 

df = pd.DataFrame(data , index = [1 , 2 , 3 , 4 , 5 , 6]) 


#Features and labels 
X = df[["Study_Hours"]]   #independent variable 
y = df["Pass_Fail"]       #dependent variable 

X_train,y_train,X_test,y_test = train_test_split(X,y,test_size=0.2,random_state=42) 

#model
model = LogisticRegression() 
model.fit(X_train,y_train) 

#Prediction 
y_pred = model.predict(X_test) 

#Evaluation 
print("Accuracy: " , accuracy_score(y_test , y_pred)) 
print("Precision: " , precision_score(y_test , y_pred)) 
print("Recall: " , recall_score(y_test , y_pred)) 
print("f1_score: " , f1_score(y_test , y_pred)) 
print("confusion_matrix: " , confusion_matrix(y_test , y_pred)) 




