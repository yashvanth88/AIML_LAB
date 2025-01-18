import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Load iris dataset from sklearn
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['class'] = iris.target_names[iris.target]

# Split the data into features and labels
X = df.iloc[:, :-1]  # features
y = df.iloc[:, -1]   # labels

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Euclidean distance function
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# KNN prediction function
def knn_predict(train_data, train_labels, test_point, k=3):
    distances = []

    for i in range(len(train_data)):
        distance = euclidean_distance(test_point.values, train_data.iloc[i, :].values)
        distances.append((distance, train_labels.iloc[i]))

    sorted_distances = sorted(distances, key=lambda x: x[0])
    k_nearest_neighbors = sorted_distances[:k]

    class_counts = {}
    for neighbor in k_nearest_neighbors:
        label = neighbor[1]
        class_counts[label] = class_counts.get(label, 0) + 1
    predicted_class = max(class_counts, key=class_counts.get)
    return predicted_class

# Make predictions on the test set
pred = [knn_predict(X_train, y_train, X_test.iloc[i, :], k=3) for i in range(len(X_test))]
accuracy = accuracy_score(y_test, pred)
print("Accuracy:", accuracy)

# Ask for user input for a test example
print("Enter the features for a test example:")
sepallength = float(input("Sepal length: "))
sepalwidth = float(input("Sepal width: "))
petallength = float(input("Petal length: "))
petalwidth = float(input("Petal width: "))

# Create a DataFrame for the user's test example
a = pd.DataFrame([[sepallength, sepalwidth, petallength, petalwidth]],
                 columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"])

# Predict for the user's input
pred_class = knn_predict(X_train, y_train, a.iloc[0], k=3)
print(f"Predicted class: {pred_class}")
