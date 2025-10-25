# Decision Tree Regression Example
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

# Sample dataset
data = {
    'Size_sqft': [800, 1000, 1200, 1500, 1800, 2000, 2200, 2500],
    'Price': [150000, 180000, 200000, 250000, 300000, 320000, 340000, 400000]
}

df = pd.DataFrame(data)

# Features and target
X = df[['Size_sqft']] 
y = df['Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Decision Tree Regressor
reg = DecisionTreeRegressor(criterion='squared_error', max_depth=3, random_state=42)
reg.fit(X_train, y_train) 

# Predictions
predictions = reg.predict(X_test)
print("Predicted Prices:", predictions)
print("Actual Prices:", y_test.values)

# Plot the tree
plt.figure(figsize=(10,6))
plot_tree(reg, feature_names=['Size_sqft'], filled=True)
plt.show()
