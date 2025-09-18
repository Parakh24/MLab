from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Example dataset  
texts = ["Free money now!!!", 
         "Hi, how are you?", 
         "Win lottery prize", 
         "Let's meet tomorrow"] 
labels = [1, 0, 1, 0]   # 1=spam, 0=not spam 
                 
# Convert text to feature vectors 
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts) 

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.5, random_state=42)

# Multinomial NB for text data
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
