import pandas as pd
import numpy as np
from knn_imputer import KNN_Imputer
from utils import SMOTE, balance_classes



# def load_and_preprocess(file_path, output_file_path):
#     df = pd.read_excel(file_path)
#
#     # Drop unnecessary columns
#     y = df['Race'].values
#     X = df.drop(columns=['Race', 'Horodateur'])
#
#     # 'F' -> 1, 'M' -> 0, 'NSP' -> NaN
#     X['Sexe'] = X['Sexe'].replace({'F': 1, 'M': 0, 'NSP': np.nan})
#
#     # One-hot encode
#     X_encoded = pd.get_dummies(X.drop(columns=['Sexe']))
#     X_encoded['Sexe'] = X['Sexe']
#
#     # Handle missing values using KNN imputer
#     if X_encoded.isnull().values.any():
#         knn_imputer = KNN_Imputer(k=5)
#         knn_imputer.fit(X_encoded.values)
#         X_imputed = knn_imputer.predict()
#         X_encoded = pd.DataFrame(X_imputed, columns=X_encoded.columns)
#
#     # Replace True/False with 1/0
#     X_encoded = X_encoded.replace({True: 1, False: 0})
#
#     # Convert to numpy array
#     X_encoded = X_encoded.values
#
#     # Balance the classes
#     X_balanced, y_balanced = balance_classes(X_encoded, y)
#
#     # Round all numerical values to the nearest integer
#     X_balanced = np.round(X_balanced)
#
#     # Create DataFrames for balanced data
#     X_balanced_df = pd.DataFrame(X_balanced, columns=pd.get_dummies(X.drop(columns=['Sexe'])).columns.tolist() + ['Sexe'])
#     y_balanced_df = pd.DataFrame(y_balanced, columns=['Race'])
#
#     # Concatenate the balanced features and target
#     balanced_df = pd.concat([X_balanced_df, y_balanced_df], axis=1)
#
#     # Save the modified dataset to an Excel file
#     balanced_df.to_excel(output_file_path, index=False)
#
#     return X_balanced, y_balanced

def load_and_preprocess(file_path, output_file_path):
    df = pd.read_excel(file_path)

    # Drop unnecessary columns
    y = df['Race'].values
    X = df.drop(columns=['Race', 'Horodateur'])

    # Replace 'NSP' with NaN in the entire dataframe
    X = X.replace('NSP', np.nan)

    # Transform 'Sexe' column: 'F' -> 1, 'M' -> 0
    X['Sexe'] = X['Sexe'].replace({'F': 1, 'M': 0})

    # Drop the 'Plus' column before one-hot encoding and save it for later
    high_cardinality_column = 'Plus'
    if high_cardinality_column in X.columns:
        plus_column = X[high_cardinality_column]
        X = X.drop(columns=[high_cardinality_column])

    # One-hot encode other categorical features
    X_encoded = pd.get_dummies(X.drop(columns=['Sexe']))
    X_encoded['Sexe'] = X['Sexe']

    # Add back the 'Plus' column
    if high_cardinality_column in locals():
        X_encoded[high_cardinality_column] = plus_column

    # Handle missing values using KNN imputer
    if X_encoded.isnull().values.any():
        knn_imputer = KNN_Imputer(k=5)
        knn_imputer.fit(X_encoded.values)
        X_imputed = knn_imputer.predict()
        X_encoded = pd.DataFrame(X_imputed, columns=X_encoded.columns)

    # Replace True/False with 1/0
    X_encoded = X_encoded.replace({True: 1, False: 0})

    # Balance the classes
    X_balanced, y_balanced = balance_classes(X_encoded.values, y)

    # Round all numerical values to the nearest integer
    X_balanced = np.round(X_balanced)

    # Create DataFrames for balanced data
    X_balanced_df = pd.DataFrame(X_balanced, columns=X_encoded.columns)
    y_balanced_df = pd.DataFrame(y_balanced, columns=['Race'])

    # Concatenate the balanced features and target
    balanced_df = pd.concat([X_balanced_df, y_balanced_df], axis=1)

    # Save the modified dataset to an Excel file
    balanced_df.to_excel(output_file_path, index=False)

    return X_balanced, y_balanced

if __name__ == "__main__":
    file_path = r"Data\Refined_CatDataset.xlsx"
    output_file_path = r"Data\Final_CatDataset.xlsx"

    X_balanced, y_balanced = load_and_preprocess(file_path, output_file_path)

    # Check for NaN values in the numpy array
    nan_count = np.isnan(X_balanced).sum()
    print(f'Checking for NaN values left... Hope not: {nan_count}')

    print(f"Modified data successfully saved to {output_file_path}")

