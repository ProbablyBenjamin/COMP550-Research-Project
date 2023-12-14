import re
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import pickle
from nltk.tokenize import word_tokenize
from dataset.get_dataset import get_preprocessed_instances, get_labels
from preprocess import preprocess_text
from seed_sampler import random_split, uniform_split
import random

x, y = get_preprocessed_instances(), get_labels()
x, x_test, y, y_test = train_test_split(x, y, test_size=0.2)

def preprocessor(s):
    s = s.lower()
    s = re.sub(r'[^a-zA-Z]', ' ', s)
    s = " ".join(s.split())
    return s

def yarowsky_bootstrapping(labeled_sentences, labels, unlabeled_sentences, threshold):
    # Initialize vectorizer without a preprocessor since data is already preprocessed
    # vectorizer = TfidfVectorizer(lowercase=True, tokenizer=word_tokenize, token_pattern = None)
    vectorizer = CountVectorizer(lowercase = True, tokenizer=word_tokenize, token_pattern = None, min_df = 1)
    # Use with_mean=False for sparse matrices
    # scaler = StandardScaler()
    # scaler.fit(vectorizer.fit_transform(x))

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

        # classifier = GaussianNB()
        classifier = LogisticRegression(C = 1.0, max_iter = 200)
        # classifier = RandomForestClassifier(n_estimators=100)
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
        y_pred = classifier.predict(vectorizer.transform(x_test).toarray())
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
        if len(unlabeled_indices) == unlabeled_len:
            break
        
    y_pred = classifier.predict(vectorizer.transform(x_test).toarray())
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    # print(classification_report(y_test, y_pred))

    return classifier

def main():
    
    # Assuming uniform_split is defined in seedsample.py
    seed_x, seed_y = uniform_split(x, y, 15)  # Adjust the per_class_size as needed
    # seed_x, seed_y = random_split(x, y, seed_set_size = 105)

    unlabeled_x = [sentence for sentence in x if sentence not in seed_x]

    classifier = yarowsky_bootstrapping(seed_x, seed_y, unlabeled_x, threshold=0.6)

    # Evaluation and saving the classifier

    # y_pred = classifier.predict(x_test)
    # print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    # print(classification_report(y_test, y_pred))

    # with open('classifier_model.pkl', 'wb') as file:
    #     pickle.dump(classifier, file)

if __name__ == "__main__":
    main()
