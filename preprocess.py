import re
import nltk
nltk.download('punkt', download_dir = '/mnt/d/mcgill/comp550/COMP550-Research-Project/venv/nltk_data')
nltk.download('stopwords', download_dir = '/mnt/d/mcgill/comp550/COMP550-Research-Project/venv/nltk_data')
nltk.download('wordnet', download_dir = '/mnt/d/mcgill/comp550/COMP550-Research-Project/venv/nltk_data')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from dataset.get_dataset import get_instances, get_labels

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def to_lowercase(s):
    return s.lower()

def remove_copyright_tag(s):
    s = s.split("Copyright (C)")[0]
    s = s.split("Copyright (c)")[0]
    return s

'''
an alternative that could be explored: convert numbers into their word representations
'''
def remove_numbers_and_punctuation(s):
    return re.sub(r'[^a-zA-Z]', ' ', s)

def remove_extra_whitespaces(s):
    return " ".join(s.split())

'''
explore better tokenizers suited to this task
'''
def tokenize(s):
    return word_tokenize(s)

'''
convert list of tokens to string
'''
def stringify(words): 
    return " ".join(words)

'''
applies to the next few functions:
words should be a list obtained by tokenizing a document/train example
returns a list of tokenized words without stopwords
'''
def remove_stopwords(words):
    return [w for w in words if w not in stopwords.words("english")]

def lemmatize(words):
    return [lemmatizer.lemmatize(w) for w in words]

def stem(words):
    return [stemmer.stem(w) for w in words]

def remove_single_characters(words):
    return [w for w in words if len(w) > 1]

'''
TODOs: 
-> method toremove words that appear not very frequently (they likely don't generalize well)
'''

def preprocess_text(s):
    s = to_lowercase(s)
    s = remove_copyright_tag(s)
    s = remove_numbers_and_punctuation(s)
    s = remove_extra_whitespaces(s)
    s = tokenize(s)
    s = remove_stopwords(s)
    s = remove_single_characters(s)
    s = lemmatize(s)
    s = stringify(s)
    return s

if __name__ == "__main__":
    s = get_instances()[0]
    print(s)
    s = to_lowercase(s)
    s = remove_copyright_tag(s)
    s = remove_numbers_and_punctuation(s)
    s = remove_extra_whitespaces(s)
    print(s)
    s = tokenize(s)
    s = remove_stopwords(s)
    s = remove_single_characters(s)
    s = lemmatize(s)
    s = stringify(s)
    print(s)



