import pandas as pd
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#Classification
data = {

'X' : [1,2,3,4,5],
'Y' : [2,3,4,7,8],
'Z' : ["Red","Blue","Red","Blue","Blue"]

} 

df = pd.DataFrame(data , index = [1,2,3,4,5]) 

X = df[['X' , 'Y']]
y = df['Z'] 

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3 , random_state = 36) 

model = KNeighborsClassifier(n_neighbors = 3 , weight = 'uniform') 

model.fit(X_train,y_train) 

y_pred = model.predict(X_test) 

print('Accuracy_score: ' , accuracy_score(y_pred , y_test))

print('Precision_score: ' , precision_score(y_pred , y_test)) 

print('Recall: ' , recall_score(y_pred , y_test))

print('F-1_Score: ' , f1_score(y_pred , y_test)) 