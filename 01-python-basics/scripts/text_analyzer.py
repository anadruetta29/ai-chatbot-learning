import json
import operator
from collections import Counter

# Turn text into lower case
def turn_lower_case(text):
    return text.lower()

# Remove punctuation signs
def remove_punctuation(text):
    punctuation = ["!", "?", ".", "-", ",", ";", ":"]
    for p in punctuation:
        text = text.replace(p, "")
    return text

# Remove line breaks
def remove_line_breaks(text):
    return text.replace("\n", " ")

# Normalize text -> result: a list of words
def normalize_text(text):
    text = turn_lower_case(text)
    text = remove_punctuation(text)
    text = remove_line_breaks(text)
    return text.split()

# Count words
def count_words(text):
    words = text.split()
    return len(words)

# Calculates the frequency of each word in a list of words
def word_frequency(words):
    return dict(Counter(words))

# Returns the most frequently used words
def most_used_words(frequency, top):
    frequency_sorted = sorted(frequency.items(), key=operator.itemgetter(1), reverse=True)
    return frequency_sorted[:top]

# Removes stop words from a list of words.
def remove_stop_words(text):
    stop_words = {
        "and", "is", "a", "in", "are", "the", "of", "to",
        "for", "on", "with", "as", "by", "an", "be", "this",
        "that", "it", "at", "from"
    }
    return [word for word in text if word not in stop_words]


def main():
    file_path = "../exercises/sample.txt"

    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    words = normalize_text(content)
    words = remove_stop_words(words)
    frequency = word_frequency(words)
    top_words = most_used_words(frequency, 5)

    print(json.dumps(top_words, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()



