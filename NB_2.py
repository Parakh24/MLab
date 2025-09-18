from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
 
# Example dataset  
texts = ["Free money now!!!", 
         "Hi, how are you?", 
         "Win lottery prize", 
         "Let's meet tomorrow"] 
labels = [1, 0, 1, 0]   # 1 = spam, 0 = not spam 
                 
# Convert text into word count feature vectors (bag-of-words model)
# Example: "Free money now!!!" -> [1,1,1,0,...] where values are word counts
vectorizer = CountVectorizer()   
X = vectorizer.fit_transform(texts)  

# Split into training and testing sets (50% training, 50% testing for demo)
# random_state ensures reproducibility of results
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.5, random_state=42) 

# MultinomialNB is best for word counts (frequency of words matters)
# Unlike BernoulliNB, repeated words increase influence
clf = MultinomialNB()     
clf.fit(X_train, y_train)   # Learn word likelihoods for spam vs not spam

# Predict class labels for unseen test texts
y_pred = clf.predict(X_test)

# Evaluate with accuracy (correct predictions / total predictions)
print("Accuracy:", accuracy_score(y_test, y_pred)) 
