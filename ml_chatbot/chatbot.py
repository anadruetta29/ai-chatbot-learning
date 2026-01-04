import joblib
import json
import random
from nlp_basics.preprocessing import preprocess_text

# Load model
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")


def load_responses(path):
    with open(path, "r") as f:
        data = json.load(f)

    responses = {}
    for intent in data["intents"]:
        tag = intent["tag"]
        responses[tag] = intent.get("responses", [])

    return responses


def predict_intent(text):
    tokens = preprocess_text(text)
    processed_text = " ".join(tokens)

    vec = vectorizer.transform([processed_text])
    probs = model.predict_proba(vec)[0]
    idx = probs.argmax()

    return model.classes_[idx], probs[idx]

def get_response(intent, confidence, responses, threshold=0.20):
    if confidence <= threshold:
        return "Sorry,I didn't understand that."

    return random.choice(responses.get(intent, ["..."]))

def main():
    responses = load_responses("./datasets/dataset.json")

    print("Chatbot ready! (type 'quit' to exit)")
    print("-------------------------------------")
    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("ChatBot: Goodbye!")
            break

        intent, confidence = predict_intent(user_input)
        reply = get_response(intent, confidence, responses)


        # Save logs
        if reply is None:
            with open("logs/unknown.txt", "a") as f:
                f.write(user_input + "\n")
        print(f"ChatBot: {reply}")

if __name__ == "__main__":
    main()
