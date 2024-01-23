"""
Thermal / IR camera (IR)
    The thermal camera images were analyzed to identify areas where temperatures
    varied widely between different test series and components.
    As a result, the mean temperatures of four areas are extracted:
    sprue, dome, horizontal edge and vertical edge (see figure below).
    In addition, the average temperature of the complete component is extracted,
    excluding the area around the cutout.
"""
import pandas as pd


def thermal_data():
    # Set Pandas display options to show all content
    pd.set_option('display.max_columns', None)  # Shows all columns
    pd.set_option('display.max_rows', None)  # Shows all rows
    pd.set_option('display.max_colwidth', None)  # Shows full content of each column
    pd.set_option('display.width', None)  # Auto-detects the width of the terminal

    # Load the Parquet file
    df = pd.read_parquet('dataset/dataset.parquet', engine='pyarrow')

    # Use regex to filter columns starting with "IR"
    filtered_df = df.filter(regex=r'^IR', axis=1)

    # List of columns to exclude
    exclude_columns = [
        "IR_Image1Name",
        "IR_Image2Name",
        "IR_Image3Name"
    ]

    # 填充缺失值为特定的值，比如0
    filtered_df = filtered_df.fillna(0)

    # Remove the excluded columns
    filtered_df = filtered_df.drop(columns=exclude_columns, errors='ignore')

    # Display the filtered data
    # print(filtered_df)

    return filtered_df
