import preprocess
import numpy as np

# Apply tf-idf to the training data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Get the training data
X = preprocess.get_instances()
y = preprocess.get_labels()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

# Apply tf-idf to the training data
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train the model
svclassifier = SVC(kernel = 'linear')
svclassifier.fit(X_train, y_train)

# Make predictions
y_pred = svclassifier.predict(X_test)

# Evaluate the model
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# Result: [[263   7   4   7  11   2   3]
#  [ 10 200   1  13   8   0   1]
#  [  0   0 329   0   1  14   3]
#  [ 12  13   1 350  15   2   4]
#  [ 17   1   2   8 384   0   8]
#  [  3   0  16   0   1 281  22]
#  [  4   2   4   6   4  34 323]]
#               precision    recall  f1-score   support

#            0       0.85      0.89      0.87       297
#            1       0.90      0.86      0.88       233
#            2       0.92      0.95      0.93       347
#            3       0.91      0.88      0.90       397
#            4       0.91      0.91      0.91       420
#            5       0.84      0.87      0.86       323
#            6       0.89      0.86      0.87       377

#     accuracy                           0.89      2394
#    macro avg       0.89      0.89      0.89      2394
# weighted avg       0.89      0.89      0.89      2394

# 0.8897243107769424