import pandas as pd
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import mean_squared_error,mean_absolute_error 
from sklearn.model_selection import train_test_split 

#Regression
data = {

'House_size' : [20000,30000,40000,50000],
'Price' : [100000, 150000, 300000, 450000]

} 

df = pd.DataFrame(data) 

X = df[['House_size']]
y = df['Price'] 

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 36) 

model = KNeighborsClassifier(n_neighbors = 3 , weights = 'uniform') 

model.fit(X_train,y_train) 

y_pred = model.predict(X_test) 

print('Mean_Squared_Error : ' , mean_squared_error(y_pred , y_test))  

print('Mean_Absolute_error : ' , mean_absolute_error(y_pred , y_test))  

