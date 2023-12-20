import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import math
from nltk.tokenize import word_tokenize
from dataset.get_dataset import get_preprocessed_instances, get_labels, get_llm_labels
from preprocess import preprocess_text
from seed_sampler import random_split, uniform_split
import random

x, y = get_preprocessed_instances(), get_labels()
y_llm = get_llm_labels()

train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

x, x_test, y, y_test, y_llm, _ = train_test_split(x, y, y_llm, test_size = 1 - train_ratio)
x_val, x_test, y_val, y_test = train_test_split(x, y, test_size = test_ratio/(validation_ratio + test_ratio))


def yarowsky_bootstrapping(labeled_sentences, labels, unlabeled_sentences, threshold):
    # Initialize vectorizer without a preprocessor since data is already preprocessed
    vectorizer = TfidfVectorizer(lowercase=True, tokenizer=word_tokenize, token_pattern = None, min_df = 2)
    # vectorizer = CountVectorizer(lowercase = True, tokenizer=word_tokenize, token_pattern = None, min_df = 2)

    x_train = labeled_sentences
    y_train = labels

    print(len(unlabeled_sentences))
    unlabeled_indices = list(range(len(unlabeled_sentences)))

    x_train = vectorizer.fit_transform(x_train).toarray()
    y_train = np.array(y_train)

    while True:
        x_unlabeled = [unlabeled_sentences[i] for i in unlabeled_indices]
        unlabeled_len = len(unlabeled_indices)

        if len(x_unlabeled) == 0:
            break

        # classifier = MultinomialNB(alpha = 1e-5)
        # classifier = LogisticRegression(C = 10.0, max_iter = 300)
        classifier = RandomForestClassifier(n_estimators=200)
        classifier.fit(x_train, y_train)

        x_unlabeled = vectorizer.transform(x_unlabeled).toarray()
        proba = classifier.predict_proba(x_unlabeled)
        max_proba = np.max(proba, axis = 1)
        confident_indices = [i for i, p in enumerate(max_proba) if p >= threshold]

        if len(confident_indices) == 0:
            break

        x_confident = x_unlabeled[confident_indices]
        y_confident = np.argmax(proba[confident_indices], axis = 1)
        x_train = np.vstack([x_train, x_confident])
        y_train = np.concatenate([y_train, y_confident])
        unlabeled_indices = [i for i in unlabeled_indices if i not in confident_indices]
        print(len(unlabeled_indices))

        if len(unlabeled_indices) == unlabeled_len:
            break
        
    y_pred = classifier.predict(vectorizer.transform(x_val).toarray())
    print(f"Validation Accuracy: {accuracy_score(y_val, y_pred)}")
    y_pred = classifier.predict(vectorizer.transform(x_test).toarray())
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred)}")

    return classifier

def main():

    seed_set_size = math.ceil(0.1*len(x)) # for random_split
    # seed_set_size = math.ceil((len(x)*0.05)/7) # for uniform_split
    threshold = 0.3
    
    # seed_x, seed_y = uniform_split(x, y, seed_set_size)  # Adjust the per_class_size as needed
    seed_x, seed_y = random_split(x, y_llm, seed_set_size = seed_set_size)

    # for LLM generated seed set
   
    unlabeled_x = [sentence for sentence in x if sentence not in seed_x]

    classifier = yarowsky_bootstrapping(seed_x, seed_y, unlabeled_x, threshold = threshold)

if __name__ == "__main__":
    main()
