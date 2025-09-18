# Iris dataset classification 
# This function is used when features are classification
import pandas as pd 
from sklearn.datasets import load_iris                       # loads the famous iris dataset (flowers classification)
from sklearn.model_selection import train_test_split         # splits data into training and testing sets 
from sklearn.naive_bayes import GaussianNB                   # Gaussian Naive Bayes Classifier 
from sklearn.metrics import accuracy_score, confusion_matrix # evaluation metrics  


# Load Dataset 
iris = load_iris()                                           # an object containing features and labels 
X, y = iris.data, iris.target                                # X variable contains features and y variable contains target

# train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42) 

# Initialize Gaussian Naive Bayes 
model = GaussianNB()                                         # creates an instance of the Gaussian Naive Bayes Model.

# train
model.fit(X_train, y_train)                                  # algorithm learns from the training data

# Predictions 
y_pred = model.predict(X_test)                               # predicts the model 

# Evaluate 
print("Accuracy: " , accuracy_score(y_test, y_pred))  
print("Confusion_Matrix: " , confusion_matrix(y_test, y_pred))  
