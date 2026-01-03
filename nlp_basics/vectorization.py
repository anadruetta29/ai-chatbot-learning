from preprocessing import preprocess_text
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

sentences = [
    "Hello! How are you doing today?",
    "I'm leaving now, see you later!",
    "Can you help me with this problem?",
    "Maybe I better not go",
    "Hello again, are you ready today?",
    "I have a problem today",
]

# Use lemmantized tokens from preprocessing.py
lemmatized_tokens = preprocess_text(sentences)

# Document
documents_str = [" ".join(lemmatized_tokens)]

# Bag of words
vectorizer_bow = CountVectorizer()
X_bow = vectorizer_bow.fit_transform(documents_str)

print("Vocabulary: ", vectorizer_bow.vocabulary_)
print("Matrix: ", X_bow.toarray())

# TF-IDF
vectorizer_tfidf = TfidfVectorizer()
X_tfidf = vectorizer_tfidf.fit_transform(documents_str)

print("Vocabulary: ", vectorizer_tfidf.vocabulary_)
print("Matrix: ", X_tfidf.toarray())




