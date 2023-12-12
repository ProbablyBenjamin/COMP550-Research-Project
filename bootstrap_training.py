import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from tqdm import tqdm
from model_baseline import train_test_data_split, vectorize, train_and_evaluate
from preprocess import get_instances, get_labels
from scipy.sparse import vstack, hstack, csr_matrix
from sklearn.preprocessing import StandardScaler

# Mockup for vectorization function to be replaced with actual preprocessing logic
def vectorize_data(X, y):
    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)
    return X_vectorized, y

def yarowsky_bootstrap_corrected(initial_model, X_labeled, y_labeled, X_unlabeled, threshold=0.9, max_iterations=100):
    model = clone(initial_model)
    # Ensure y_labeled is a numpy array for proper indexing
    y_labeled = np.array(y_labeled)

    # Initialize an array to keep track of which samples are currently labeled
    labeled_indices = np.arange(X_labeled.shape[0])

    # Combine labeled and unlabeled data for easy indexing
    X_combined = vstack([X_labeled, X_unlabeled])
    y_combined = np.concatenate([y_labeled, np.zeros(X_unlabeled.shape[0])])  # Placeholder zeros for unlabeled data

    for iteration in range(max_iterations):
        # Fit the model on the currently labeled data
        X_train = X_combined[labeled_indices].toarray() if isinstance(X_combined, csr_matrix) else X_combined[labeled_indices]
        y_train = y_combined[labeled_indices]  # Indexing with a numpy array
        # Optionally, scale the data
        scaler = StandardScaler(with_mean=False)  # Use with_mean=False for sparse matrices
        X_train_scaled = scaler.fit_transform(X_train)

        # Fit the model on the currently labeled data
        model.fit(X_train_scaled, y_train)  # Use scaled data
        # model.fit(X_train, y_train)
        
        # Predict probabilities on the unlabeled data
        unlabeled_indices = [i for i in range(X_combined.shape[0]) if i not in labeled_indices]
        X_test_unlabeled = X_combined[unlabeled_indices].toarray() if isinstance(X_combined, csr_matrix) else X_combined[unlabeled_indices]
        proba = model.predict_proba(X_test_unlabeled)
        
        # Find the instances in the unlabeled data with a predicted probability above the threshold
        confident_indices = np.flatnonzero(np.max(proba, axis=1) > threshold)
        actual_confident_indices = [unlabeled_indices[i] for i in confident_indices]

        # Break the loop if no confident predictions are made
        if not actual_confident_indices:
            break

        # Add the confident predictions to the labeled dataset
        y_confident = np.argmax(proba[confident_indices], axis=1)
        y_combined[actual_confident_indices] = y_confident

        # Update the labeled indices
        labeled_indices = np.concatenate([labeled_indices, actual_confident_indices])

        print(f"Iteration {iteration}: {len(confident_indices)} new instances added.")
    
    return model, X_combined[labeled_indices], y_combined[labeled_indices]

# Get unlabeled instances
def get_unlabeled_instances():
    f = open(f'dataset/WOS/WOS11967/X.txt')
    x = f.read().splitlines()
    return x

# The main function to run the bootstrapping process
def main():
    X_labeled = get_instances()  # Replace with actual data retrieval
    y_labeled = get_labels()     # Replace with actual data retrieval
    X_unlabeled = get_unlabeled_instances()  # Replace with actual data retrieval

    # Vectorize the data
    X_vectorized, y_encoded = vectorize_data(X_labeled, y_labeled)
    X_unlabeled_vectorized, _ = vectorize_data(X_unlabeled, None)  # No labels for unlabeled data

    # Initialize the model
    # initial_model = LogisticRegression(max_iter=100)
    initial_model = LogisticRegression(max_iter=1000, solver='saga', C=1.0)

    # Run Yarowsky bootstrapping
    refined_model, X_refined, y_refined = yarowsky_bootstrap_corrected(
        initial_model, X_vectorized, y_encoded, X_unlabeled_vectorized
    )

    # Split the refined data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_refined, y_refined, test_size=0.2, random_state=42)

    # Train the refined model on the training set
    refined_model.fit(X_train, y_train)

    # Evaluate the refined model on the testing set
    predictions = refined_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy of the refined Logistic Regression model is: {accuracy}")

if __name__ == "__main__":
    main()
