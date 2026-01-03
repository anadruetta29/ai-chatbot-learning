## Rule based Chat Bot

This project implements a rule-based chatbot that detects user intent by performing partial text 
matching against predefined patterns.

## How it works
The chatbot follows this flow:
1. The user enters a sentence
2. The input text is normalized (tokenization, stopword removal, lemmatization)
3. The input is compared against the patterns of each intent
4. The most suitable intent is detected
5. A response associated with that intent is returned

## Text preprocessing
The chatbot uses NLTK for text normalization (reuse of preprocess_text function in npm_basics):
- Tokenization
- Lowercasing
- Stopword removal
- Lemmatization with POS tagging

This allows the chatbot to:
- Avoid relying on exact phrases
- Improve partial matching
- Handle natural language variations

## Intent detection
Intent detection is performed by comparing the normalized tokens from the user input with the normalized tokens of each intent pattern.
An intent is selected if at least one relevant token matches.

Example:
Input: "Can you help me?"
→ Tokens: ["help"]
→ Detected intent: help

## Limitations
The chatbot does not learn from new data
All patterns must be defined manually
It may fail with sentences that differ significantly from the predefined patterns