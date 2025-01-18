import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic regression implementation
def logistic_regression(X, y, iterations=200, lr=0.001):
    weights = np.zeros(X.shape[1])  # Initialize weights
    for _ in range(iterations):
        z = np.dot(X, weights)
        predictions = sigmoid(z)
        gradient = np.dot(X.T, (predictions - y)) / y.size
        weights -= lr * gradient  # Update weights
    return weights

# Load dataset and preprocess
iris = load_iris()
X = iris.data[:, :2]  # Use the first two features
y = (iris.target != 0).astype(int)  # Binary classification: 0 (Setosa), 1 (Not Setosa)

# Map classes to species
species_map = {0: "Setosa", 1: "Versicolor or Virginica"}

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=9)

# Standardize features
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Train logistic regression
weights = logistic_regression(X_train_std, y_train)

# Predictions
y_train_pred = sigmoid(np.dot(X_train_std, weights)) > 0.5
y_test_pred = sigmoid(np.dot(X_test_std, weights)) > 0.5

# Accuracy
train_accuracy = np.mean(y_train_pred == y_train)
test_accuracy = np.mean(y_test_pred == y_test)

print(f'Training Accuracy: {train_accuracy:.4f}')
print(f'Testing Accuracy: {test_accuracy:.4f}')

# User input for prediction
print("\nEnter the features for a test example (sepal length and sepal width):")
sepal_length = float(input("Sepal length: "))
sepal_width = float(input("Sepal width: "))

# Standardize user input
user_input = scaler.transform([[sepal_length, sepal_width]])

# Predict using the logistic regression model
user_prediction = sigmoid(np.dot(user_input, weights)) > 0.5
predicted_species = species_map[int(user_prediction[0])]
print(f"Predicted species for the given example: {predicted_species}")

# Plot decision boundary
x_min, x_max = X_train_std[:, 0].min() - 1, X_train_std[:, 0].max() + 1
y_min, y_max = X_train_std[:, 1].min() - 1, X_train_std[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = sigmoid(np.dot(np.c_[xx.ravel(), yy.ravel()], weights)) > 0.5
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_train_std[:, 0], X_train_std[:, 1], c=y_train, edgecolor='k', label='Train Data')
plt.scatter(X_test_std[:, 0], X_test_std[:, 1], c=y_test, marker='x', label='Test Data')
plt.title('Logistic Regression Decision Boundary')
plt.xlabel('Sepal length (standardized)')
plt.ylabel('Sepal width (standardized)')
plt.legend()

plt.savefig('plot.png')
