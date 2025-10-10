from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier 
import numpy as np 
from sklearn.datasets import load_iris 

#this kfold is used for continuous values 
#there is another term i.e. stratified kfold which is used for discrete values i.e.  

data = load_iris() 
X, y = data.data , data.target 

# K-fold Cross Validation 
k = 5 

kf = KFold(n_splits=k, shuffle=True, random_state=42)

#Initialize the RandomClassifier model 
model = RandomForestClassifier(random_state = 42) 

#Perform Cross Validation 
scores = cross_val_score(model , X , y , cv = kf , scoring = 'accuracy') 




