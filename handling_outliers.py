#Handling Outliers means detecting and treating data points that are significantly different(much higher or lower
# from most other values in a dataset 

import pandas as pd 
import numpy as np 
from scipy import stats
from sklearn.model_selection import train_test_split 

data = {
    'Age' : [22 , 25 , 27 , 29 , 31 , 34 , 40 , 100],
    'Salary' : [20000 , 25000 , 47000 , 52000 , 46000 , 48000 , 70000 , 80000] 
} 

df = pd.DataFrame(data) 

X = df[['Age']] 
y = df['Salary'] 

X_train,X_test,y_train,y_test = train_test_split(X , y , test_size = 0.2 , random_state = 36) 

z_scores = np.abs(stats.zscore(X_train)) 




