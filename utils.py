import random

import numpy as np
from sklearn.neighbors import NearestNeighbors


def nearest_neighbour(X, k=5):
    """
    Find k-nearest neighbors for each sample in X using Euclidean distance.

    Parameters:
    - X: np.array, feature set for a class (shape: [n_samples, n_features])
    - k: number of neighbors (default: 5)

    Returns:
    - indices: np.array, indices of k-nearest neighbors for each sample
    """
    k = min(k, len(X))

    nbs = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(X)
    _, indices = nbs.kneighbors(X)
    return indices


def SMOTE(X, num_samples):
    """
    Generate synthetic samples using SMOTE technique by interpolating between a sample
    and one of its nearest neighbors.

    Parameters:
    - X: np.array, feature set for the class
    - num_samples: number of synthetic samples to generate

    Returns:  -
    - synthetic_samples: np.array, the generated synthetic samples
    """
    # Find the k-nearest neighbors for each point in X
    k = min(5, len(X))

    # You can change this number of neighbors based on your requirement
    neighbors = NearestNeighbors(n_neighbors=k).fit(X)

    # Get the indices of the k-nearest neighbors
    indices = neighbors.kneighbors(X, return_distance=False)

    synthetic_samples = []

    # Generate synthetic samples
    for _ in range(num_samples):
        # Randomly select a sample from the original dataset
        sample_idx = random.randint(0, len(X) - 1)
        sample = X[sample_idx]

        # Randomly select one of the k-nearest neighbors
        neighbor_idx = random.choice(indices[sample_idx][1:])  # Avoid selecting the point itself
        neighbor = X[neighbor_idx]

        # Generate a synthetic sample by linear interpolation between sample and neighbor
        gap = np.random.rand()  # Random number between 0 and 1
        synthetic_sample = sample + gap * (neighbor - sample)  # Interpolating

        synthetic_samples.append(synthetic_sample)

    return np.array(synthetic_samples)


def balance_classes(X, y):
    """
    Balance all classes by generating synthetic samples for minority classes so each class has
    the same number of instances as the majority class.

    Parameters:
    - X: np.array, the feature matrix (shape: [n_samples, n_features])
    - y: np.array, the label vector (shape: [n_samples])

    Returns:
    - X_balanced: np.array, balanced feature matrix
    - y_balanced: np.array, balanced label vector
    """

    unique_classes, class_counts = np.unique(y, return_counts=True)
    max_count = np.max(class_counts)

    X_balanced = []
    y_balanced = []

    for class_label in unique_classes:

        class_indices = np.where(y == class_label)[0]
        X_class = X[class_indices]


        X_balanced.extend(X_class)
        y_balanced.extend([class_label] * len(X_class))


        if len(X_class) < max_count:
            num_samples_to_generate = max_count - len(X_class)
            synthetic_samples = SMOTE(X_class, num_samples_to_generate)

            X_balanced.extend(synthetic_samples)
            y_balanced.extend([class_label] * len(synthetic_samples))

    return np.array(X_balanced), np.array(y_balanced)
