import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load datasets
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

# Add labels
fake_df["label"] = 0   # fake
true_df["label"] = 1   # true

# Combine datasets
df = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)

# Clean text
def clean_text(text):
    text = re.sub(r'\n', ' ', text)       # remove line breaks
    text = re.sub(r'\W|\d', ' ', text)    # remove special chars and digits
    text = text.lower()                   # lowercase
    return text

df['text'] = df['text'].apply(clean_text)

# Split dataset
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation
score = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("✅ Fake News Detector Results")
print(f"Accuracy: {score*100:.2f}%")
print("Confusion Matrix:")
print(cm)
