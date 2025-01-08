import pandas as pd

def filter_sort_and_save_excel(excel_path, sort_column, min_value=None, max_value=None, ascending=True):
    """
    Filters and sorts the Excel file by a specified column and saves the resulting data into a new file.
    
    Parameters:
    - excel_path (str): Path to the Excel file to be filtered and sorted.
    - sort_column (str): Name of the column to sort by.
    - min_value (float): Minimum value for filtering. Default is None (no minimum filter).
    - max_value (float): Maximum value for filtering. Default is None (no maximum filter).
    - ascending (bool): Sort order. True for ascending, False for descending. Default is True.
    
    Returns:
    - str: Path to the filtered and sorted Excel file.
    """
    
    # Load the Excel file into a DataFrame
    df = pd.read_excel(excel_path, engine='openpyxl')
    
    # Check if the sort_column exists in the DataFrame
    if sort_column not in df.columns:
        raise ValueError(f"The column '{sort_column}' does not exist in the Excel file.")
    
    # Filter the DataFrame based on min_value and max_value
    if min_value is not None:
        df = df[df[sort_column] >= min_value]
    if max_value is not None:
        df = df[df[sort_column] <= max_value]
    
    # Sort the filtered DataFrame by the specified column
    df_sorted = df.sort_values(by=sort_column, ascending=ascending)
    
    # Save the sorted DataFrame to a new Excel file
    filtered_sorted_excel_path = excel_path.replace(
        '.xlsx', f'_filtered_sorted_by_{sort_column}.xlsx'
    )
    df_sorted.to_excel(filtered_sorted_excel_path, index=False, engine='openpyxl')
    
    return filtered_sorted_excel_path



# Example Excel file path
excel_path = '/Users/jeremiah/phd_thesis/data/main_data/excel2_modified.xlsx'

# User input for filtering and sorting
sort_column = 'Total Seconds'  # Column to sort by
min_value = 10.0  # Filter for values greater than or equal to 10 seconds
max_value = 50.0  # Filter for values less than or equal to 300 seconds
ascending = True  # Sort in ascending order

# Call the function
filtered_sorted_excel_path = filter_sort_and_save_excel(
    excel_path, sort_column, min_value, max_value, ascending
)
print(f"The filtered and sorted Excel file is saved as: {filtered_sorted_excel_path}")
