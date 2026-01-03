# Machine learning basics

This project is part of Phase 3 – Machine Learning Fundamentals of an AI learning path using Python.
The goal is to build a simple text classifier that can categorize short sentences into one of three intents:
- greeting
- question
- closing

## Dataset
The dataset is a small CSV file with two columns:
text,label
hello,greeting
good morning,greeting
how is it going,greeting
bye,closing
see you later,closing
take care,closing
how are you,question
what time is it,question
can you help me,question
what are you doing,question
good afternoon,greeting
hi there,greeting
good evening,greeting
see you soon,closing
goodbye,closing
where are you,question
how much is it,question
is it raining,question
have a nice day,closing
hey,greeting
good to go,closing
i guess that's it,closing

- Feature (X): text (raw sentences)
- Label (y): label (intent category)

Note: This dataset is an updated version of the original one. 
Additional phrases were included and the model was retrained by running train.py again.

# Model choice
Algorithm: Naive Bayes (MultinomialNB)

# Text vectorization
Technique: TF-IDF (Term Frequency – Inverse Document Frequency)

# Evaluation
Metric: Accuracy
accuracy = correct_predictions / total_predictions

Example result:
y_test: ['greeting', 'question', 'question', 'greeting']
y_pred: ['closing', 'question', 'question', 'greeting']
Accuracy: 0.75
This means the model correctly classified 3 out of 4 test samples.

# Testing with new phrases
The trained model was tested with unseen sentences, such as:

"What's up?" → question
"What is your name?" → question
"I'm leaving now" → closing
"Can I go?" → question

The model shows reasonable generalization, even for sentences not present in the training set.

# Limitations
Some ambiguous sentences are misclassified, for example:
"good to go" → greeting
"i guess that's it" → question
"hello, can you help me?" → closing (in some runs)

# Saved Artifacts
After training, the following files are saved:
model.pkl (trained Naive Bayes model)
vectorizer.pkl (fitted TF-IDF vectorizer)