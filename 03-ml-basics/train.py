import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

# Read dataset
df = pd.read_csv("./datasets/dataset.csv")

X = df["text"] # Features
y = df["label"] # Labels

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorize text
vectorizer = TfidfVectorizer() # Normalize

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model - using Naive Bayes
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Make prediction with new data
y_pred = model.predict(X_test_vec)

print("y_test:", list(y_test))
print("y_pred:", list(y_pred))

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy del modelo: {accuracy:.2f}")

# Save data model
joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Model and vectorizer saved")


