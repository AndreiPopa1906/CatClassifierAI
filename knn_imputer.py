import numpy as np

class KNN_Imputer():
    def __init__(self, k):
        self.k = k
        self.data = None
        self.numeric_data = None

    def fit(self, X):
        """ Fit the imputer to the data by separating numeric and non-numeric columns """
        self.data = np.array(X, dtype=object)
        # Filter out only the numeric columns
        self.numeric_data = np.array([[float(x) if self.is_numeric(x) else np.nan for x in row] for row in X])

    def is_numeric(self, value):
        """ Check if a value is numeric """
        try:
            float(value)
            return True
        except ValueError:
            return False

    def nan_euclidean_distance(self, row1, row2, penalizing=True): # we don't penalize sparse datasets
        """ Compute Euclidean distance between two rows, ignoring NaN values, and apply the weight """
        total_coordinates = len(row1)
        mask = ~np.isnan(row1) & ~np.isnan(row2)
        valid_count = np.sum(mask)

        if valid_count == 0:
            return np.inf

        if penalizing:
            # weigthted distance
            squared_distance = np.sum((row1[mask] - row2[mask]) ** 2)
            weight = total_coordinates / valid_count
            distance = np.sqrt(weight * squared_distance)
        else:
            # normalized distance
            distance = np.sqrt(np.sum((row1[mask] - row2[mask]) ** 2) / valid_count)

        return distance

    def get_neighbors(self, row):
        """ Find k nearest neighbors based on Euclidean distance """
        distances = []
        for i, other_row in enumerate(self.numeric_data):
            if not np.array_equal(row, other_row):
                distance = self.nan_euclidean_distance(row, other_row)
                distances.append((other_row, distance))
        distances.sort(key=lambda x: x[1])
        neighbors = [distances[i][0] for i in range(self.k)]
        return neighbors

    def predict(self):
        """ Impute missing values using the mean of k nearest neighbors """
        X_imputed = self.numeric_data.copy()
        for i, row in enumerate(self.numeric_data):
            for j in range(len(row)):
                if np.isnan(row[j]):
                    neighbors = self.get_neighbors(row)
                    imputed_value = np.nanmean([neighbor[j] for neighbor in neighbors])
                    X_imputed[i][j] = imputed_value

        final_data = self.data.copy()
        for i, row in enumerate(final_data):
            for j in range(len(row)):
                if self.is_numeric(row[j]):
                    final_data[i][j] = X_imputed[i][j]

        return final_data
