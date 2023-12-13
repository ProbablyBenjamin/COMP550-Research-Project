import re
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import pickle
from nltk.tokenize import word_tokenize
from preprocess import get_instances, get_labels
from seedsample import uniform_split
from scipy.sparse import vstack
import random

def preprocessor(s):
    s = s.lower()
    s = re.sub(r'[^a-zA-Z]', ' ', s)
    s = " ".join(s.split())
    return s

def yarowsky_bootstrapping(labeled_sentences, labels, unlabeled_sentences, threshold=0.75):
    # Initialize vectorizer without a preprocessor since data is already preprocessed
    vectorizer = TfidfVectorizer(lowercase=True, tokenizer=word_tokenize)
    X_labeled = vectorizer.fit_transform(labeled_sentences)
    y_labeled = np.array(labels)

    # Scale the data
    scaler = StandardScaler(with_mean=False)  # Use with_mean=False for sparse matrices
    X_labeled = scaler.fit_transform(X_labeled)

    #classifier = LogisticRegression(max_iter=1000)
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    unlabeled_indices = list(range(len(unlabeled_sentences)))

    while unlabeled_indices:
        classifier.fit(X_labeled, y_labeled)

        processed_unlabeled = [preprocessor(unlabeled_sentences[i]) for i in unlabeled_indices]
        X_unlabeled = vectorizer.transform(processed_unlabeled)
        proba = classifier.predict_proba(X_unlabeled)
        max_proba = np.max(proba, axis=1)
        confident_indices = [i for i, p in enumerate(max_proba) if p >= threshold]

        if not confident_indices:
            break

        X_confident = X_unlabeled[confident_indices]
        y_confident = np.argmax(proba[confident_indices], axis=1)

        X_labeled = vstack([X_labeled, X_confident])
        y_labeled = np.concatenate([y_labeled, y_confident])

        unlabeled_indices = [i for i in unlabeled_indices if i not in confident_indices]

    return classifier

def main():
    # Assuming get_instances() and get_labels() fetch your initial dataset
    x, y = get_instances(), get_labels()
    
    # Assuming uniform_split is defined in seedsample.py
    seed_x, seed_y = uniform_split(x, y, 1)  # Adjust the per_class_size as needed

    unlabeled_sentences = [sentence for sentence in x if sentence not in seed_x]
    # Process each sentence individually
    processed_seed_x = [preprocessor(sentence) for sentence in seed_x]
    processed_unlabeled_sentences = [preprocessor(sentence) for sentence in unlabeled_sentences]

    classifier = yarowsky_bootstrapping(processed_seed_x, seed_y, processed_unlabeled_sentences, threshold=0.75)

    # Evaluation and saving the classifier
    vectorizer = TfidfVectorizer(lowercase=True, tokenizer=word_tokenize)
    X = vectorizer.fit_transform(x)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

    with open('classifier_model.pkl', 'wb') as file:
        pickle.dump(classifier, file)

if __name__ == "__main__":
    main()
