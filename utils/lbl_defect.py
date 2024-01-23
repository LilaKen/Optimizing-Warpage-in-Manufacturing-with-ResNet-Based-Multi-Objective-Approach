"""
Manual Visual Defect Inspection (LBL)
    After weighing the parts, they were visually inspected to identify typical defect types.
    A list of the possible defects is given below.
"""
import pandas as pd

def lbl_defect():
    # Set Pandas display options to show all content
    pd.set_option('display.max_columns', None)  # Shows all columns
    pd.set_option('display.max_rows', None)     # Shows all rows
    pd.set_option('display.max_colwidth', None) # Shows full content of each column
    pd.set_option('display.width', None)        # Auto-detects the width of the terminal

    # Load the Parquet file
    df = pd.read_parquet('dataset/dataset.parquet', engine='pyarrow')

    # Use regex to filter columns starting with "LBL"
    filtered_df = df.filter(regex=r'^LBL', axis=1)

    # 填充缺失值为特定的值，比如0
    filtered_df = filtered_df.fillna(0)

    # Display the filtered data
    # print(filtered_df)

    return filtered_df


# if __name__ == '__main__':
#     x = lbl_defect()
#     x = x.sum(axis=1)
#     print(sum(x))
#     average_defect = sum(x) / len(x)
#     print("LBL_defect 平均值:", average_defect)

