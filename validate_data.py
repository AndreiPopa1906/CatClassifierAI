import random
import numpy as np
import pandas as pd
from knn_imputer import KNN_Imputer
from utils import balance_classes

# Load the workbook and select a worksheet
path_database = r'C:\Users\matei\PycharmProjects\CatClassifierAI\DropColumn_InitialDataset.xlsx'
df = pd.read_excel(path_database, sheet_name=None)  # Load all sheets into a dictionary of DataFrames
path_modified_database = r'C:\Users\matei\Documents\AI_project\DropColumn_NEWNEW_InitialDataset.xlsx'


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


def delete_rows(df, rows_to_delete):
    """
    Delete specified rows from the DataFrame.
    """
    return df.drop(rows_to_delete)


def get_categories(df):
    """
    Get category mappings from the 'Code' sheet.
    """
    map_structure = {}
    for row_index in range(1, len(df)):
        cell_value = df.iloc[row_index, 0]
        if pd.isna(cell_value):
            break
        possible_values = get_possible_values(df, row_index)
        map_structure[cell_value] = possible_values
    return map_structure


def get_possible_values(df, row_index):
    """
    Get possible values for a given row from the 'Code' sheet.
    """
    cell_value = df.iloc[row_index, 1]
    if pd.isna(cell_value):
        return []
    return list(map(str.strip, cell_value.split('/')))


def search_for_row_duplicates(df):
    """
    Search for duplicate rows based on values from columns C to AB.
    """
    seen_rows = {}
    duplicate_rows = []
    for row_index in range(len(df)):
        row_values = tuple(df.iloc[row_index, 2:28])
        if row_values in seen_rows:
            duplicate_rows.append(row_index)
            print(f"Duplicate found in row {row_index + 2}, matches row {seen_rows[row_values] + 2}")
        else:
            seen_rows[row_values] = row_index
    return duplicate_rows


def validate_data():
    """
    Validate data in the 'Data' sheet.
    """
    sheet_data = df['Data']
    sheet_code = df['Code']
    map_structure = get_categories(sheet_code)
    invalid_data_category_set = set()
    to_delete_rows = set()

    column_names = {col_index: sheet_data.columns[col_index] for col_index in range(2, 28)}

    for row_index in range(len(sheet_data)):
        row_invalid = False
        for col_index in range(2, 28):
            cell_value = sheet_data.iloc[row_index, col_index]
            column_name = column_names[col_index]
            if column_name in map_structure:
                valid_values = map_structure[column_name]
                if isinstance(cell_value, int):
                    cell_value = str(cell_value)
                if cell_value not in valid_values:
                    row_invalid = True
                    invalid_data_category_set.add((column_name, cell_value))
                    if column_name == "Race":
                        to_delete_rows.add(row_index)
            else:
                try:
                    cell_value = int(cell_value)
                    if not (1 <= cell_value <= 5):
                        row_invalid = True
                        invalid_data_category_set.add((column_name, cell_value))
                except (ValueError, TypeError):
                    row_invalid = True
        if row_invalid:
            to_delete_rows.add(row_index)

    duplicates = search_for_row_duplicates(sheet_data)
    to_delete_rows.update(duplicates)

    updated_df = delete_rows(sheet_data, list(to_delete_rows))
    df['Data'] = updated_df

    with pd.ExcelWriter(path_modified_database) as writer:
        for sheet_name, sheet_df in df.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
    print("All changes saved successfully.")

    for element in invalid_data_category_set:
        print(element)


def init():
    """
    Initialize the workbook and modify specific columns.
    """
    sheet_code = df['Code']
    sheet_code.at[4, 'B'] = "1/2/3/4/5"
    print(f"Updated 'Code' sheet cell B5: 1/2/3/4/5")
    validate_data()


if __name__ == "__main__":
    init()
