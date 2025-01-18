import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features (sepal length, sepal width, petal length, petal width)
y = iris.target  # Labels (species)

# Function for KMeans
def kmeans(X, k):
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]  # Initialize centroids randomly

    for _ in range(100):  # Run for 100 iterations
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)  # Compute distances
        labels = np.argmin(distances, axis=1)  # Assign points to nearest centroid
        centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])  # Update centroids

    return centroids, labels

# Apply custom k-means clustering with 3 clusters
k = 3
centroids, labels = kmeans(X, k)

# Plot the original data points with different colors for each cluster (using the first two features)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)

# Plot the final cluster centroids (using the first two features for simplicity)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')

plt.title('K-means Clustering on Iris Dataset')
plt.xlabel('Sepal Length')  # X-axis label (e.g., Sepal Length)
plt.ylabel('Sepal Width')   # Y-axis label (e.g., Sepal Width)
plt.legend()

# Save the plot as a PNG file
plt.savefig('kmeans_plot.png')

# Optional: Show the plot if you want to see it immediately
plt.show()
