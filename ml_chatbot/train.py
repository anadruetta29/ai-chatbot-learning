import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib


def read_file(filename, file_type):
    if file_type == "csv":
        return pd.read_csv(filename)
    elif file_type == "json":
        return pd.read_json(filename)
    elif file_type == "excel":
        return pd.read_excel(filename)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def split_data(df):
    X = df["text"]
    y = df["intent"]
    return train_test_split(
        X, y, test_size=0.2, random_state=42
    )


def vectorize_TF_IDF(X_train, X_test):
    vectorizer = TfidfVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer


def train_model_NB(X_train_vec, y_train):
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    return model


def save_data_model(model, vectorizer):
    joblib.dump(model, "models/model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")
    print("Model and vectorizer saved")


def train():
    df = read_file("./datasets/dataset.json", "json")

    X_train, X_test, y_train, y_test = split_data(df)

    X_train_vec, X_test_vec, vectorizer = vectorize_TF_IDF(X_train, X_test)

    model = train_model_NB(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    save_data_model(model, vectorizer)


if __name__ == "__main__":
    train()
