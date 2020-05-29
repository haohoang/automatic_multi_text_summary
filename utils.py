import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from string import punctuation

"""
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')"""

p = punctuation + "0123456789"
stop = set(stopwords.words('english'))
exclude = set(p)


def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence_not_retain_tag(sentence):
    lemmatizer = WordNetLemmatizer()
    # tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    # tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_sentence.append(word.lower())
        else:
            # else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word.lower(), tag))
    return lemmatized_sentence


def clean_not_remain_tag(doc):
    # remove stop words, punctuation, and lemmatize words
    lemm = lemmatize_sentence_not_retain_tag(doc)
    s_free = " ".join([i for i in lemm if i not in stop])
    p_free = ''.join([ch for ch in s_free if ch not in exclude])
    # only take words which are greater than 2 characters"
    cleaned = [word for word in p_free.split() if len(word) >= 2]
    return cleaned

