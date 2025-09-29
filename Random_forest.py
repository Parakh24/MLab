import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix 

# Sample dataset 
data = {
    'Age': [25, 30, 45, 35, 40, 50, 23, 34],
    'Income': [40000, 60000, 80000, 120000, 70000, 90000, 30000, 100000],
    'Buy': ['No', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)

# Features and target     
X = df[['Age', 'Income']]        # Independent variables
y = df['Buy']                    # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Random Forest Classifier
rf_clf = RandomForestClassifier(
    n_estimators=100,   # Number of trees in the forest
    max_depth=3,        # Maximum depth of each tree (prevents overfitting)
    random_state=42
)                           

# Training the model                                                                                 
rf_clf.fit(X_train, y_train)                                

# Predictions
y_pred = rf_clf.predict(X_test)

# Evaluation
print("Predictions:", y_pred)
print("Actual:", y_test.values)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

