# NLP Pipeline

This mini-project demonstrates the basic Natural Language Processing (NLP) pipeline using Python and NLTK.
The goal is to convert raw text into numerical representations that machine learning models can understand.

## Text processing
Purpose: Clean and normalize text so that it can be used by ML algorithms.
Steps:
- Tokenization: Split sentences into words
- Lowercasing: Convert all words to lowercase
- Stopwords removal: Remove common words like "the", "is", "and"
- Lemmatization: Convert words to their base form (e.g., "leaving" → "leave")
- POS tagging: Identify word types to improve lemmatization accuracy

Example: 
sentences = [
    "Hello! How are you doing today?",
    "I'm leaving now, see you later!",
    "Can you help me with this problem?",
    "Maybe I better not go",
    "Hello again, are you ready today?",
    "I have a problem today",
]

Output:
Original text: Hello! How are you doing today? I'm leaving now, see you later! Can you help me with this problem? Maybe I better not go Hello again, are you ready today? I have a problem today
Tokens: ['hello', '!', 'how', 'are', 'you', 'doing', 'today', '?', 'i', "'m", 'leaving', 'now', ',', 'see', 'you', 'later', '!', 'can', 'you', 'help', 'me', 'with', 'this', 'problem', '?', 'maybe', 'i', 'better', 'not', 'go', 'hello', 'again', ',', 'are', 'you', 'ready', 'today', '?', 'i', 'have', 'a', 'problem', 'today']
Filtered tokens: ['hello', 'today', 'leaving', 'see', 'later', 'help', 'problem', 'maybe', 'better', 'go', 'hello', 'ready', 'today', 'problem', 'today']
Lemmatized tokens: ['hello', 'today', 'leave', 'see', 'later', 'help', 'problem', 'maybe', 'well', 'go', 'hello', 'ready', 'today', 'problem', 'today']

## Bag of Words 
Concept: Represent text as a matrix of word counts.
- Each row = a document (I only used one document)
- Each column = a unique word (vocabulary)
- Each cell = count of the word in that document

Output:
Vocabulary:  {'hello': 1, 'today': 9, 'leave': 4, 'see': 8, 'later': 3, 'help': 2, 'problem': 6, 'maybe': 5, 'well': 10, 'go': 0, 'ready': 7}
Matrix:  [[1 2 1 1 1 1 2 1 1 3 1]]

## TF-IDF
Concept: Represent text by weighting words based on importance.
- Words that appear in many documents 
- Words that appear rarely

Output: 
Vocabulary:  {'hello': 1, 'today': 9, 'leave': 4, 'see': 8, 'later': 3, 'help': 2, 'problem': 6, 'maybe': 5, 'well': 10, 'go': 0, 'ready': 7}
Matrix:  [[0.2 0.4 0.2 0.2 0.2 0.2 0.4 0.2 0.2 0.6 0.2]]

