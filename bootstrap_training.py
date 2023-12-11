from sklearn.calibration import LabelEncoder
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
from sklearn.utils import resample
from model_baseline import train_test_data_split, vectorize, train_and_evaluate

# Set a random seed for reproducibility
np.random.seed(42)

def vectorize_data(X, y):
    # Vectorizing text data in X
    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    # Encoding categorical labels in y
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X_vectorized, y_encoded

def bootstrap_training(model, X, y, n_iterations=100, sample_size=1.0):
    accuracies = []
    for i in tqdm(range(n_iterations)):
        # Resampling with replacement
        # Use X.shape[0] instead of len(X) for sparse matrix
        X_sample, y_sample = resample(X, y, n_samples=int(X.shape[0] * sample_size))
        X_train, X_test, y_train, y_test = train_test_data_split(X_sample, y_sample)

        # Train the model
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        accuracies.append(accuracy)

    return np.mean(accuracies), np.std(accuracies)

def main():
    # Get the training data
    X = preprocess.get_instances()
    y = preprocess.get_labels()

    # Vectorize the data
    X_vectorized, y_encoded = vectorize_data(X, y)

    # Train and evaluate Logistic Regression with Bootstrap Aggregation
    logreg = LogisticRegression(max_iter=1000)
    mean_accuracy, std_accuracy = bootstrap_training(logreg, X_vectorized, y_encoded)
    print(f"Logistic Regression with Bootstrap Aggregation: {mean_accuracy:.2f} +/- {std_accuracy:.2f}")

    # Train and evaluate Random Forest with Bootstrap Aggregation
    rf = RandomForestClassifier(n_estimators=1000, random_state=42)
    mean_accuracy, std_accuracy = bootstrap_training(rf, X_vectorized, y_encoded)
    print(f"Random Forest with Bootstrap Aggregation: {mean_accuracy:.2f} +/- {std_accuracy:.2f}")

    # # Train and evaluate SVM with Bootstrap Aggregation
    svm = SVC()
    mean_accuracy, std_accuracy = bootstrap_training(svm, X_vectorized, y_encoded)
    print(f"SVM with Bootstrap Aggregation: {mean_accuracy:.2f} +/- {std_accuracy:.2f}")

if __name__ == "__main__":
    main()