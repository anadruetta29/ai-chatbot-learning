import joblib
from nlp_basics import preprocessing

# Import model and vectorizer
model = joblib.load("./models/model.pkl")
vectorizer = joblib.load("./models/vectorizer.pkl")

# Test model with new phrases
new_phrases = [
    "What's up buddy?",
    "I'm heading out now",
    "I am stuck, help!",
    "Thanks for everything",
    "What is your function?",
    "Tell me your name",
    "Good night, see you"
]
processed_phrases = []

for phrase in new_phrases:
    tokens = preprocessing.preprocess_text(phrase)
    processed_phrases.append(" ".join(tokens))

new_phrases = processed_phrases

# Vectorize and predict
phrases_vec = vectorizer.transform(new_phrases)
predictions = model.predict(phrases_vec)

# Show results
probs = model.predict_proba(phrases_vec)

for phrase, label, prob in zip(new_phrases, predictions, probs):
    confidence = max(prob)
    print(f'"{phrase}" → {label} ({confidence:.2f})')

