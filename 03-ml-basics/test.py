import joblib

# Import model and vectorizer
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Test model with new phrases
new_phrases = [
    "What's up?",
    "What is your name?",
    "I'm leaving now",
    "Can I go?",
    "good to go",
    "are we done?",
    "i guess that's it",
    "hello, can you help me?",
    "ok bye then",
    "so... what now?",
]

# Vectorize and predict
phrases_vec = vectorizer.transform(new_phrases)
predictions = model.predict(phrases_vec)

# Show results
for phrase, label in zip(new_phrases, predictions):
    print(f'"{phrase}" → {label}')
