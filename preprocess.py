import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def to_lowercase(s):
    return s.lower()

def remove_copyright_tag(s):
    s = s.split("Copyright (C)")[0]
    s = s.split("Copyright (c)")[0]
    return s

def remove_numbers_and_punctuation(s):
    return re.sub(r'[^a-zA-Z]', ' ', s)


