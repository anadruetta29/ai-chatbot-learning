import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

from nlp_basics.preprocessing import preprocess_text

def clean_previous_models():
    model_path = Path("models/model.pkl")
    vectorizer_path = Path("models/vectorizer.pkl")

    for path in [model_path, vectorizer_path]:
        if path.exists():
            path.unlink()

def read_intents_json(path):
    with open(path, "r") as f:
        data = json.load(f)

    texts = []
    intents = []

    for intent in data["intents"]:
        tag = intent["tag"]
        for pattern in intent["patterns"]:
            texts.append(pattern)
            intents.append(tag)

    return pd.DataFrame({
        "text": texts,
        "intent": intents
    })


def split_data(df):
    X = df["text"]
    y = df["intent"]
    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


def preprocess_series(text_series):
    return [" ".join(preprocess_text(t)) for t in text_series]


def vectorize_TF_IDF(X_train, X_test):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer


def train_model_NB(X_train_vec, y_train):
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    return model


def save_data_model(model, vectorizer):
    clean_previous_models()
    joblib.dump(model, "models/model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")
    print("Model and vectorizer saved")

# Training process
df = read_intents_json("./datasets/dataset.json")
X_train, X_test, y_train, y_test = split_data(df)
X_train = preprocess_series(X_train)
X_test = preprocess_series(X_test)
X_train_vec, X_test_vec, vectorizer = vectorize_TF_IDF(X_train, X_test)
model = train_model_NB(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
save_data_model(model, vectorizer)

