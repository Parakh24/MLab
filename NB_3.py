from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Example dataset  
texts = ["Free money now!!!", 
         "Hi, how are you?", 
         "Win lottery prize", 
         "Let's meet tomorrow"] 
labels = [1, 0, 1, 0]   # 1 = spam, 0 = not spam 

# Convert text into binary feature vectors (presence/absence of words, not counts)
# Example: "Free money now!!!" -> [1,1,1,0,...] where 1 = word present, 0 = absent
vectorizer = CountVectorizer(binary=True)  
X = vectorizer.fit_transform(texts) 

# Split data into training and testing sets (50% train, 50% test for this tiny dataset)
# random_state ensures reproducibility of the split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.5, random_state=42) 

# BernoulliNB is ideal when features are binary (0/1), i.e., presence/absence of words
clf = BernoulliNB()     
clf.fit(X_train, y_train)   # Learn probabilities from training data

# Predict class labels (spam/not spam) for unseen test messages
y_pred = clf.predict(X_test)

# Evaluate performance: Accuracy = correct predictions / total predictions
print("Accuracy:", accuracy_score(y_test, y_pred))  

