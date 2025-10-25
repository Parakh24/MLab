#onehotEncoder 
from sklearn.preprocessing import OneHotEncoder 
import numpy as np 

data = np.array([['Red'] , ['Blue'] , ['Green'] , ['Purple']]) 
one = OneHotEncoder(sparse_output= False) 
encoded = one.fit_transform(data)
print(encoded) 

