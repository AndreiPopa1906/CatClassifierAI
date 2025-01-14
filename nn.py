import os
import random

import numpy as nmp
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.decomposition import PCA
import joblib


race_mapping = []

# Activation Functions
def leaky_relu(x, alpha=0.01):
    return nmp.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x, alpha=0.01):
    return nmp.where(x > 0, 1, alpha)


def cross_entropy_loss(y_true, y_pred):
    return -nmp.sum(y_true * nmp.log(y_pred + 1e-9)) / y_true.shape[0]


def softmax(x):
    x = nmp.clip(x, -500, 500)  # Clip values to avoid overflow in exp
    exp_x = nmp.exp(x - nmp.max(x, axis=1, keepdims=True))
    return exp_x / nmp.sum(exp_x, axis=1, keepdims=True)


# Data Handling
def extract_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir,  'Final5_CatDataset.xlsx')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The dataset at the path {file_path} is not existent.")

    data = pd.read_excel(file_path, dtype={'Race': 'category'}, engine='openpyxl')
    X = data.drop(columns=['Race'])
    X = X.select_dtypes(include=[nmp.number]).apply(pd.to_numeric, downcast='float').values
    data['Race'] = data['Race'].astype('category')
    y = data['Race'].cat.codes.values
    global race_mapping
    race_mapping = dict(enumerate(data['Race'].cat.categories))
    # One-Hot Encoding for labels
    unique_classes = nmp.unique(y)
    y_encoded = nmp.zeros((y.shape[0], len(unique_classes)))
    for i, label in enumerate(y):
        y_encoded[i, nmp.where(unique_classes == label)[0][0]] = 1
    y = y_encoded
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8  # Add a small value to avoid division by zero
    X = (X - X_mean) / X_std

    return X, y

import  numerical_preprocess as np
# Data Splitting
def data_split(X, y, train_size=0.8):
    unique_classes, class_counts = nmp.unique(y.argmax(axis=1), return_counts=True)
    train_indices = []
    test_indices = []
    for cls in unique_classes:
        cls_indices = nmp.where(y.argmax(axis=1) == cls)[0]
        nmp.random.shuffle(cls_indices)
        split_idx = int(train_size * len(cls_indices))
        train_indices.extend(cls_indices[:split_idx])
        test_indices.extend(cls_indices[split_idx:])
    nmp.random.shuffle(train_indices)
    nmp.random.shuffle(test_indices)
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


# Layer Class
class Layer:
    def __init__(self, input_size, output_size, activation='relu', dropout_rate=0.0):
        self.learning_rate = 0.001
        self.dropout_rate = dropout_rate
        self.dropout_mask = None
        self.activation = activation
        self.weights = nmp.random.randn(input_size, output_size) * nmp.sqrt(2 / input_size)  # He Initialization
        self.bias = nmp.zeros((1, output_size))  # Initialize biases to zero
        self.m = nmp.zeros_like(self.weights)
        self.v = nmp.zeros_like(self.weights)
        self.m_b = nmp.zeros_like(self.bias)
        self.v_b = nmp.zeros_like(self.bias)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0

    def forward(self, x, use_dropout=True):
        self.input = x
        self.z = nmp.dot(x, self.weights) + self.bias

        if self.activation == 'relu':
            self.output = leaky_relu(self.z)
        elif self.activation == 'softmax':
            self.output = softmax(self.z)
        else:
            raise ValueError("Unsupported activation function")

        if use_dropout and self.dropout_rate > 0.0:
            self.dropout_mask = nmp.random.binomial(1, 1 - self.dropout_rate, size=self.output.shape) / (
                        1 - self.dropout_rate)
            self.output *= self.dropout_mask

        return self.output

    def adam_optimizer(self, dW, dB):
        # Adam Optimizer
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * dW
        self.v = self.beta2 * self.v + (1 - self.beta2) * (dW ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        self.weights -= self.learning_rate * m_hat / (nmp.sqrt(v_hat) + self.epsilon)

        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * dB
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (dB ** 2)
        m_hat_b = self.m_b / (1 - self.beta1 ** self.t)
        v_hat_b = self.v_b / (1 - self.beta2 ** self.t)
        self.bias -= self.learning_rate * m_hat_b / (nmp.sqrt(v_hat_b) + self.epsilon)

    def backward(self, error):
        lambda_reg = 0.005

        if self.dropout_rate > 0.0 and self.dropout_mask is not None:
            error *= self.dropout_mask

        if self.activation == 'relu':
            delta = error * leaky_relu_derivative(self.z)
        elif self.activation == 'softmax':
            delta = error
        else:
            raise ValueError("Unsupported activation function")

        dW = nmp.dot(self.input.T, delta) / self.input.shape[0]
        dB = nmp.sum(delta, axis=0, keepdims=True) / self.input.shape[0]

        dW += lambda_reg * self.weights

        self.adam_optimizer(dW, dB)

        return nmp.dot(delta, self.weights.T)

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate


# Neural Network Class
class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, input_size, output_size, activation='relu', dropout_rate=0.0):
        layer = Layer(input_size, output_size, activation, dropout_rate)
        self.layers.append(layer)

    def forward(self, x, use_dropout=True):
        for layer in self.layers:
            x = layer.forward(x, use_dropout=use_dropout)
        return x

    def backward(self, error):
        for layer in reversed(self.layers):
            error = layer.backward(error)

    def fit(self, X, y, X_val, y_val, epochs=10, batch_size=64, learning_rate=0.001, patience=5, decay_factor=0.1):
        best_val_loss = float('inf')
        patience_counter = 0
        training_losses = []
        validation_losses = []
        initial_learning_rate = learning_rate

        for layer in self.layers:
            layer.set_learning_rate(learning_rate)

        for epoch in range(epochs):
            if epoch % 50 == 0 and epoch != 0:
                # Learning rate restart
                learning_rate = initial_learning_rate * 0.8
                for layer in self.layers:
                    layer.set_learning_rate(learning_rate)
                print(f"Learning rate restarted to {learning_rate:.6f}")

            mean_loss = 0
            for i in range(0, X.shape[0], batch_size):
                x_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                y_pred = self.forward(x_batch, use_dropout=True)
                error = y_pred - y_batch
                self.backward(error)
                loss = cross_entropy_loss(y_batch, y_pred)
                mean_loss += loss
            mean_loss /= (X.shape[0] / batch_size)
            train_accuracy = self.evaluate(X, y)
            val_accuracy = self.evaluate(X_val, y_val)
            val_loss = cross_entropy_loss(y_val, self.forward(X_val, use_dropout=False))
            training_losses.append(mean_loss)
            validation_losses.append(val_loss)
            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {mean_loss:.4f}, Training Accuracy: {train_accuracy * 100:.2f}%, Validation Accuracy: {val_accuracy * 100:.2f}%")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    learning_rate *= decay_factor
                    for layer in self.layers:
                        layer.set_learning_rate(learning_rate)
                    print(f"Learning rate reduced to {learning_rate:.6f} due to plateau in validation loss.")
                    patience_counter = 0
                    if learning_rate < initial_learning_rate * 0.01:
                        print("Early stopping triggered due to no improvement in validation loss.")
                        break

        self.plot_losses(training_losses, validation_losses, len(training_losses))
        self.plot_misclassified_points(X_val, y_val)

    def plot_losses(self, training_losses, validation_losses, epochs):
        plt.figure(figsize=(10, 6))
        plt.plot(range(epochs), training_losses, label='Training Loss')
        plt.plot(range(epochs), validation_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.show()

    def plot_misclassified_points(self, X_val, y_val):
        y_pred = self.forward(X_val, use_dropout=False)
        y_pred_labels = nmp.argmax(y_pred, axis=1)
        y_true_labels = nmp.argmax(y_val, axis=1)
        misclassified_indices = nmp.where(y_pred_labels != y_true_labels)[0]

        # Use PCA to reduce dimensions to 2 if data is not 2D
        if X_val.shape[1] != 2:
            pca = PCA(n_components=2)
            X_val_2d = pca.fit_transform(X_val)
            X_val = X_val_2d

        plt.figure(figsize=(10, 6))
        plt.scatter(X_val[:, 0], X_val[:, 1], c=y_true_labels, cmap='viridis', marker='o', alpha=0.5,
                    label='True Labels')
        plt.scatter(X_val[misclassified_indices, 0], X_val[misclassified_indices, 1], facecolors='none', edgecolors='r',
                    s=100, label='Misclassified Points')
        plt.xlabel('')
        plt.ylabel('')
        plt.title('Misclassified Points Visualization')
        plt.legend()
        plt.show()


    def evaluate(self, X, y):
        y_pred = self.forward(X, use_dropout=False)
        y_pred_labels = nmp.argmax(y_pred, axis=1)
        y_true_labels = nmp.argmax(y, axis=1)
        accuracy = nmp.sum(y_pred_labels == y_true_labels) / y.shape[0]
        precision = precision_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0)
        recall = recall_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0)
        f1 = f1_score(y_true_labels, y_pred_labels, average='weighted', zero_division=0)
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

        misclassified_indices = nmp.where(y_pred_labels != y_true_labels)[0]
        print(f"Number of misclassifications: {len(misclassified_indices)}")
        return accuracy

    def classify(self, input_array):
        columns = [
            "Row.names",
            "Nombre", "Ext", "Obs", "Timide", "Calme", "Effrayé", "Intelligent", "Vigilant",
            "Perséverant", "Affectueux", "Amical", "Solitaire", "Brutal", "Dominant",
            "Agressif", "Impulsif", "Prévisible", "Distrait", "PredOiseau", "PredMamm",
            "Age_1a2", "Age_2a10", "Age_Moinsde1", "Age_Plusde10", "Logement_AAB",
            "Logement_ASB", "Logement_MI", "Logement_ML", "Zone_PU", "Zone_R", "Zone_U",
            "Abondance_1", "Abondance_2", "Abondance_3", "Sexe"
        ]
        values = {
            "Row.names": [0],
            "Nombre": [1, 2, 3, 4, 5],
            "Ext": [1, 2, 3, 4, 5],
            "Obs": [1, 2, 3, 4],
            "Timide": [1, 2, 3, 4, 5],
            "Calme": [1, 2, 3, 4, 5],
            "Effrayé": [1, 2, 3, 4, 5],
            "Intelligent": [1, 2, 3, 4, 5],
            "Vigilant": [1, 2, 3, 4, 5],
            "Perséverant": [1, 2, 3, 4, 5],
            "Affectueux": [1, 2, 3, 4, 5],
            "Amical": [1, 2, 3, 4, 5],
            "Solitaire": [1, 2, 3, 4, 5],
            "Brutal": [1, 2, 3, 4, 5],
            "Dominant": [1, 2, 3, 4, 5],
            "Agressif": [1, 2, 3, 4, 5],
            "Impulsif": [1, 2, 3, 4, 5],
            "Prévisible": [1, 2, 3, 4, 5],
            "Distrait": [1, 2, 3, 4, 5],
            "PredOiseau": [1, 2, 3, 4, 5],
            "PredMamm": [1, 2, 3, 4, 5],
            "Age_1a2": [0, 1],
            "Age_2a10": [0, 1],
            "Age_Moinsde1": [0, 1],
            "Age_Plusde10": [0, 1],
            "Logement_AAB": [0, 1],
            "Logement_ASB": [0, 1],
            "Logement_MI": [0, 1],
            "Logement_ML": [0, 1],
            "Zone_PU": [0, 1],
            "Zone_R": [0, 1],
            "Zone_U": [0, 1],
            "Abondance_1": [0, 1],
            "Abondance_2": [0, 1],
            "Abondance_3": [0, 1],
            "Sexe": [0, 1],
        }
        modified_input = input_array.copy()
        for i, val in enumerate(input_array):
            values["Row.names"][0] = random.randint(1, 2000)
            if nmp.isnan(val):
                column_name = columns[i]
                possible_values = values.get(column_name)
                modified_input[i] = random.choice(possible_values)
        X = modified_input
        # print(X)

        y_pred = self.forward(X, use_dropout=False)
        y_pred_value = np.argmax(y_pred)
        return race_mapping[y_pred_value]



def main():
    start_time = time.time()
    X_train, X_test, y_train, y_test = data_split(*extract_data())
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]

    nn = NeuralNetwork()
    # nn.add_layer(input_size=input_size, output_size=256, activation='relu', dropout_rate=0.2)
    # nn.add_layer(input_size=256, output_size=128, activation='relu', dropout_rate=0.2)
    # nn.add_layer(input_size=128, output_size=output_size, activation='softmax')
    # print(X_test[0])
    # print(y_test[0])
    # # Train
    # nn.fit(X_train, y_train, X_test, y_test, epochs=200, batch_size=16, learning_rate=0.001, patience=8,
    #        decay_factor=0.1)
    # joblib.dump(nn, 'trained_data.pkl')
    # # Evaluate
    # test_accuracy = nn.evaluate(X_test, y_test)
    # print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    # print(f"Time taken: {time.time() - start_time:.2f} seconds")
    # joblib.dump(nn, 'trained_data.pkl')
    nn = joblib.load('trained_data.pkl')
    test_accuracy = nn.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'Final5_CatDataset.xlsx')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The dataset at the path {file_path} is not existent.")

    data = pd.read_excel(file_path, dtype={'Race': 'category'}, engine='openpyxl')
    X = data.drop(columns=['Race'])
    X = X.select_dtypes(include=[nmp.number]).apply(pd.to_numeric, downcast='float').values
    data['Race'] = data['Race'].astype('category')
    y = data['Race'].cat.codes.values
    data = {}
    correct = {}
    # print(X_test[0])
    print("Race map")
    print(race_mapping)
    for x, y in zip(X, y):
        y_pred_value = int(nmp.argmax(y))
        data[race_mapping[y_pred_value]] = data.get(race_mapping[y_pred_value], 0) + 1
        if nn.classify(x) == race_mapping[y_pred_value]:
            correct[race_mapping[y_pred_value]] = correct.get(race_mapping[y_pred_value], 0) + 1
    print(data)
    print(correct)


if __name__ == "__main__":
    main()