import preprocess
import numpy as np
import pickle
import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from dataset.get_dataset import get_preprocessed_instances, get_labels
 
def train_test_data_split(X, y, test_size=0.20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test
 
def vectorize(X_train, X_val, X_test):
    vectorizer = TfidfVectorizer(lowercase = True, tokenizer=word_tokenize, token_pattern = None, min_df = 2)
    # vectorizer = CountVectorizer(lowercase = True, tokenizer=word_tokenize, token_pattern = None, min_df = 2)
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)
    X_test = vectorizer.transform(X_test)
    return X_train, X_val, X_test
 
# Function to train and evaluate a model
def train_and_evaluate(model, X_train, y_train, X_val, y_val, model_name):
    start_time = time.time()
 
    # Train the model
    model.fit(X_train, y_train)
 
    # Measure the elapsed time
    elapsed_time = time.time() - start_time
    print(f"{model_name} Training Complete. Elapsed time: {elapsed_time:.2f} seconds")
 
    # Make predictions
    y_pred = model.predict(X_val)
 
    # Evaluate the model
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")
 

def main():
    # Get the training data
    X, y = get_preprocessed_instances(), get_labels()

    train_ratio = 0.8
    validation_ratio = 0.1
    test_ratio = 0.1

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1 - train_ratio)
    X_val, X_test, y_val, y_test = train_test_split(X, y, test_size=test_ratio/(test_ratio + validation_ratio))

 
    # Vectorize the data
    X_train, X_val, X_test = vectorize(X_train, X_val, X_test)
 
    # Train and evaluate Logistic Regression
    # model = LogisticRegression(C = 20.0, max_iter = 300)
    # model = MultinomialNB(alpha = 1e-5)
    model = RandomForestClassifier(n_estimators=200)
    train_and_evaluate(model, X_train, y_train, X_val, y_val, "RandomForest")

    y_pred = model.predict(X_test)
 
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
