import random
import numpy as np
import pandas as pd
from knn_imputer import KNN_Imputer
from utils import balance_classes

# Load the workbook and select a worksheet
path_database = r'C:\Users\matei\Documents\AI_project\cat_database.xlsx'
df = pd.read_excel(path_database, sheet_name=None)  
path_modified_database = r'C:\Users\matei\Documents\AI_project\Modified_cat_database.xlsx'


def return_column_index(df, column_header):
    """
    Return the column index for a given column header.
    """
    try:
        return df.columns.get_loc(column_header)
    except KeyError:
        raise ValueError(f"Column '{column_header}' not found.")


def modify_values_column(df, column_header):
    """
    Modify values in the specified column by incrementing the integer values by 1.
    """
    column_index = return_column_index(df, column_header)
    print(f"Modifying values in the '{column_header}' column (column {column_index})...")

    for i in range(len(df)):
        cell_value = df.iloc[i, column_index]
        if pd.isna(cell_value):
            continue
        try:
            aux = int(cell_value)
            aux += 1
            df.iloc[i, column_index] = aux
        except ValueError:
            print(f"Warning: Cell at row {i + 2}, column {column_index + 1} contains non-integer value '{cell_value}'")


def modify_values_column_nombre(df):
    """
    Modify values in the 'Nombre' column by converting 'Plusde5' to '5'.
    """
    column_index = return_column_index(df, "Nombre")
    for i in range(len(df)):
        cell_value = df.iloc[i, column_index]
        if cell_value == "Plusde5":
            df.iloc[i, column_index] = "5"


def init():
    """
    Initialize the workbook and modify specific columns.
    """
    sheet_data = df['Data']
    sheet_code = df['Code']

    sheet_code.at[4, 'B'] = "1/2/3/4/5"
    print(f"Updated 'Code' sheet cell B5: 1/2/3/4/5")

    columns_to_modify = ["Ext", "Obs", "PredMamm", "PredOiseau"]
    for column in columns_to_modify:
        modify_values_column(sheet_data, column)

    modify_values_column_nombre(sheet_data)

    # Save modified DataFrames back to Excel
    with pd.ExcelWriter(path_modified_database) as writer:
        for sheet_name, sheet_df in df.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
    print("All changes saved successfully.")


# Run the modifications
if __name__ == "__main__":
    init()
