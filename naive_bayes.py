import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
class_names = iris.target_names

class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = np.array([X[y == c].mean(axis=0) for c in self.classes])
        self.var = np.array([X[y == c].var(axis=0) for c in self.classes])
        self.priors = np.array([X[y == c].shape[0] / len(y) for c in self.classes])

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        posteriors = [np.log(self.priors[i]) + np.sum(np.log(self._pdf(i, x)))
                      for i in range(len(self.classes))]
        return self.classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean, var = self.mean[class_idx], self.var[class_idx]
        numerator = np.exp(- (x - mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train Naive Bayes model
nb = NaiveBayes()
nb.fit(X_train, y_train)

# Predict on test set
y_pred = nb.predict(X_test)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.4f}")

# Map predicted classes to species names
predicted_species = class_names[y_pred]
print("Predictions:", predicted_species)

# Print confusion matrix and classification report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

# Asking the user for a test example
print("\nEnter a test example for prediction:")
sepal_length = float(input("Enter sepal length: "))
sepal_width = float(input("Enter sepal width: "))
# Assuming features match with training data (4 features in total)
user_input = np.array([[sepal_length, sepal_width, 0, 0]])  # The other features are set to 0 for simplicity

# Predicting for the user input
user_pred = nb.predict(user_input)
predicted_species_user = class_names[user_pred][0]  # Extracting first prediction

print(f"Predicted species for the entered test example: {predicted_species_user}")
