"""
Meta Data (MET)
    The meta data includes, for example,
    the injection molding cycle number or the name of the respective test series.
    All associated parameters are listed below.
"""
import pandas as pd


def meta_data():
    # Set Pandas display options to show all content
    pd.set_option('display.max_columns', None)  # Shows all columns
    pd.set_option('display.max_rows', None)  # Shows all rows
    pd.set_option('display.max_colwidth', None)  # Shows full content of each column
    pd.set_option('display.width', None)  # Auto-detects the width of the terminal

    # Load the Parquet file
    df = pd.read_parquet('dataset/dataset.parquet', engine='pyarrow')

    # Use regex to filter columns starting with "MET"
    filtered_df = df.filter(regex=r'^MET', axis=1)

    # 填充缺失值为特定的值，比如0
    filtered_df = filtered_df.fillna(0)

    # Display the filtered data
    # print(filtered_df)

    return filtered_df
