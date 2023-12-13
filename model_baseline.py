import preprocess
import numpy as np
import pickle
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
 
def train_test_data_split(X, y, test_size=0.20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test
 
def vectorize(X_train, X_test):
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    return X_train, X_test
 
# Function to train and evaluate a model
def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name):
    start_time = time.time()
 
    # Train the model
    model.fit(X_train, y_train)
 
    # Measure the elapsed time
    elapsed_time = time.time() - start_time
    print(f"{model_name} Training Complete. Elapsed time: {elapsed_time:.2f} seconds")
 
    # Make predictions
    y_pred = model.predict(X_test)
 
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Result for {model_name}: {accuracy:.2f}")
 
    # Save the model
    pickle.dump(model, open(f'models/model_{model_name.lower()}.sav', 'wb'))

def main():
    # Get the training data
    X = preprocess.get_instances()
    y = preprocess.get_labels()

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_data_split(X, y)
 
    # Vectorize the data
    X_train, X_test = vectorize(X_train, X_test)
 
    # Train and evaluate Logistic Regression
    logreg = LogisticRegression(max_iter=1000)
    train_and_evaluate(logreg, X_train, y_train, X_test, y_test, "Logistic Regression")
 
    # Train and evaluate Random Forest
    rf = RandomForestClassifier(n_estimators=1000, random_state=42)
    train_and_evaluate(rf, X_train, y_train, X_test, y_test, "Random Forest")
 
    # Train and evaluate SVC
    svclassifier = SVC(kernel='linear')
    train_and_evaluate(svclassifier, X_train, y_train, X_test, y_test, "SVC")


if __name__ == "__main__":
    main()
# For 12k dataset:
# SVC Training Complete. Elapsed time: 131.55 seconds
# Result for SVC: 0.90
# Logistic Regression Training Complete. Elapsed time: 19.48 seconds
# Result for Logistic Regression: 0.89
# Random Forest Training Complete. Elapsed time: 861.96 seconds
# Result for Random Forest: 0.90

# For larger dataset:
# Logistic Regression Training Complete. Elapsed time: 41.31 seconds
# Result for Logistic Regression: 0.83
# Random Forest Training Complete. Elapsed time: 8916.07 seconds
# Result for Random Forest: 0.76
# SVC Training Complete. Elapsed time: 20155.70 seconds
# Result for SVC: 0.84
