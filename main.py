import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load the dataset (assumes CSV with 'text' and 'label' columns)
data = pd.read_csv('Test.csv')

# Split data into features (X) and labels (y)
X = data['text']
y = data['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)  # You can adjust max_features

# Fit and transform training text data
X_train_vec = vectorizer.fit_transform(X_train)

# Transform testing text data
X_test_vec = vectorizer.transform(X_test)

# Initialize the classifier (Multinomial Naive Bayes in this case)
classifier = MultinomialNB()

# Train the classifier
classifier.fit(X_train_vec, y_train)

# Predict on testing data
y_pred = classifier.predict(X_test_vec)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
