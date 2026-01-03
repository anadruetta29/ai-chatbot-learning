import nltk
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet


nltk.data.path.append(r"C:\Users\anadr\AppData\Roaming\nltk_data")

#nltk.download("averaged_perceptron_tagger_eng")
# nltk.download("punkt")
# nltk.download("punkt_tab")
# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download("omw-1.4")

sentences = [
    "Hello! How are you doing today?",
    "I'm leaving now, see you later!",
    "Can you help me with this problem?",
    "Maybe I better not go"
]

# Join sentences
original_text = " ".join(sentences)
print("Original text:", original_text)

# Tokenize
tokens = word_tokenize(original_text.lower())
print("Tokens:", tokens)

# Remove stopwords
stop_words = set(stopwords.words("english"))
filtered_tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
print("Filtered tokens:", filtered_tokens)

# Lemmanization
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

lemmatizer = WordNetLemmatizer()
pos_tags = pos_tag(filtered_tokens)
lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(pos))
                     for token, pos in pos_tags]
print("Lemmatized tokens:", lemmatized_tokens)
