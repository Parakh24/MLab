import numpy as np                                                   # used for numerical operations (not heavily used here but often required)
from sklearn import datasets                                         # gives access to built-in datasets like iris, digits, etc.
from sklearn.model_selection import train_test_split                 # splits data into training and testing sets
from sklearn.svm import SVC                                          # support vector classifier from sklearn.svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score   # metrics to evaluate model performance

# Load Dataset 
iris = datasets.load_iris()                                          # load iris dataset 
X = iris.data[:100, :]                                               # take the first 100 samples (only two classes: Setosa=0 , Versicolor=1) 
y = iris.target[:100]                                                # labels for these samples  

# Split data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# kernel='linear' → tells SVM to make a linear decision boundary between two classes
# C = regularization parameter
#   - Higher C → tries harder to classify all training points correctly (risk of overfitting)
#   - Lower C → allows some mistakes but makes a smoother, more generalizable boundary
clf = SVC(kernel='linear', C=1.0)                                    # create linear SVM model

# Train the SVM
clf.fit(X_train, y_train)                                            # weight vector (w) → direction of the hyperplane
                                                                     # bias (b) → offset from the origin
                                                                     # parameters stored inside model (clf) for later prediction

# Predict labels on test set 
y_pred = clf.predict(X_test)

# Evaluate 
print("Accuracy:        ", accuracy_score(y_test, y_pred))
print("Precision_score: ", precision_score(y_test, y_pred))
print("Recall_score:    ", recall_score(y_test, y_pred))
print("F1_score:        ", f1_score(y_test, y_pred))

# Support vectors 
print("Support Vectors:\n", clf.support_vectors_)
