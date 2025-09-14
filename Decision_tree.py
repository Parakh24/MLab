import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier, plot_tree 


#sample dataset 
data = {
    'Age': [25, 30, 45, 35, 40, 50, 23, 34],
    'Income': [40000, 60000, 80000, 120000, 70000, 90000, 30000, 100000],
    'Buy': ['No', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes']
} 
df = pd.DataFrame(data , index = [1,2,3])

X = df[['Age' , 'Income' , 'Buy']] 
y = df['Buy'] 

#Train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X , y , test_size = 0.3 , random_state = 42) 


#Decision tree classifier
clf = DecisionTreeClassifier(criterion = 'gini' , max_depth = 3 , random_state = 42) 
clf.fit(X_train, y_train) 

#Predictions
predictions = clf.predict(X_test) 

print("Predictions: " , predictions)
print("Actual: " , y_test.values) 

#Plot the tree 
plt.figure(figsize=(10,6)) 
plot_tree(clf, feature_names=['Age', 'Income'], class_names=['No' , 'Yes'], filled=True) 
plt.show() 
