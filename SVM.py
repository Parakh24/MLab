import numpy as np                  #used for numerical operations (not heavily used here but often required)
from sklearn import datasets        #gives acces to built-in datasets like iris, digits, etc....
from sklearn.model_selection import train_test_split  #splits data into training and testing sets
from sklearn.svm import SVC         #support vector classifier from sklearn.svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score #metric to evaluate prediction accuracy 


#Load Dataset 
iris = datasets.load_iris()         
X = iris.data[:100 , :]         #takes the first 100 samples (only two classes: Setosa = 0 , Versicolor = 1)   
y = iris.target[:100]           #the labels for these samples  


# Split data 
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.2,random_state=42) 

#kernel = 'linear' tells the SVM to make a linear decision boundary between two datapoints to separate classes
#C = sets the regularization strength 
#Controls how strict the model is about avoiding misclassifications 
#Higher C = tries harder to classify all training points correctly (risk of overfitting) 
#Lower C = allows some mistakes but make a smoother, and more generizable boundary
clf = SVC(kernel = 'linear' , C=1.0)  # C = regularization parameter 


#trains the SVM
clf.fit(X_train , y_train)   #weight vector(w) --> direction of the hyperplane
                             #bias(b) --> offset from the origin
                             #These parameters are stored inside the model(clf) so that later we can use .predict()
                               
#predicts labels on test set 
y_pred = clf.predict(X_test) 


#Evaluate 
print("Accuracy:" , accuracy_score(y_test, y_pred))
print("Precision_score:" , precision_score(y_test, y_pred))
print("Recall_score:" , recall_score(y_test, y_pred)) 
print("F1_score:" , f1_score(y_test, y_pred))  

#Support vector 
print("Support Vectors:\n" , clf.support_vectors_)   



