# Machine Learning Chatbot

This project implements a simple Machine Learning–based chatbot capable of understanding user intentions and responding accordingly.
Unlike previous rule-based approaches, this chatbot uses text vectorization and a trained classifier to predict intents dynamically.

## Architecture
Dataset
- A JSON dataset containing:
- tag: intent name
- patterns: example user phrases
- responses: possible chatbot replies

Text Preprocessing
- Tokenization
- Stopword removal
- Lemmatization (NLTK + WordNet)
- Re-joining tokens into normalized text

Vectorization
- TF-IDF (scikit-learn)
- Supports n-grams for better intent recognition

Model Training
- Multinomial Naive Bayes
- Train / test split with stratification
- Accuracy evaluation

Inference
- User input is preprocessed
-Intent probabilities are predicted
- Response is selected based on confidence threshold

## Problems 
- Low confidence for short phrases
Phrases like "hello" or "thank you" often produced low confidence scores, even when the intent was correct.

- Incorrect intent prediction for unseen phrases
Example:
User: It's ok
Predicted intent: status


- Overly strict confidence threshold
Initially:
if confidence < 0.30:
    fallback
This caused correct predictions to be rejected.

- Dataset structure evolution
The dataset was refactored from a flat structure:
{ "text": "...", "intent": "..." }
To a more realistic chatbot format:
{
  "tag": "...",
  "patterns": [...],
  "responses": [...]
}
This separation improved: training clarity response management and project scalability. 