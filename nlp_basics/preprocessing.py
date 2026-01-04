import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

nltk.data.path.append(r"C:\Users\anadr\AppData\Roaming\nltk_data")

sentences = [
    "Hello! How are you doing today?",
    "I'm leaving now, see you later!",
    "Can you help me with this problem?",
    "Maybe I better not go",
    "Hello again, are you ready today?",
    "I have a problem today",
]

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text):
    # Accept string or list of strings
    if isinstance(text, list):
        original_text = " ".join(text)
    else:
        original_text = text
    tokens = word_tokenize(original_text.lower())
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    pos_tags = pos_tag(filtered_tokens)
    lemmatized_tokens = [
        lemmatizer.lemmatize(token, get_wordnet_pos(pos))
        for token, pos in pos_tags
    ]
    return lemmatized_tokens


lemmatized_tokens = preprocess_text(sentences)
