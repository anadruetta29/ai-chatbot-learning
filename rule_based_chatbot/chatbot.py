import pandas as pd
from nlp_basics import preprocessing
from nlp_basics.preprocessing import preprocess_text
import random
import json

with open("intents.json", "r") as f:
    data = json.load(f)
df = pd.json_normalize(data['intents'])

intents = {}
for _, row in df.iterrows():
    tag = row['tag']
    patterns = row['patterns']
    responses = row['responses']
    intents[tag] = {'patterns': patterns, 'responses': responses}


def find_intent(user_sentence):
    processed_user = " ".join(preprocess_text([user_sentence]))
    for tag, data in intents.items():
        for pattern in data['patterns']:
            processed_pattern = " ".join(preprocess_text([pattern]))
            if processed_pattern in processed_user or processed_user in processed_pattern:
                return tag
    return None

print("Chatbot (type 'quit' to exit)")
while True:
    user_input = input("Enter a sentence: ")
    if user_input.lower() in ["quit", "exit"]:
        print("Chatbot: Bye!")
        break

    intent_tag = find_intent(user_input)
    if intent_tag:
        response = random.choice(intents[intent_tag]['responses'])
    else:
        response = "I didn't understand, can you rephrase?"

    print("Chatbot:", response)