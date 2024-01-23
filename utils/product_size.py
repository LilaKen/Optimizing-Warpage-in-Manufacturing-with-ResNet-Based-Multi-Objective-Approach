"""
Image Processing System / Computer Vision (CV)
    After the thermal camera has taken three photos the robot moves the part in front of a standard camera.
    This camera then takes close-up images of each of the three sections in the upper area of the injection molded part.
    Subsequently, computer vision is used to measure the distances between certain edges.
    The extracted features are listed below and can also be seen in the figure below.
"""
import pandas as pd


def product_size():
    # Set Pandas display options to show all content
    pd.set_option('display.max_columns', None)  # Shows all columns
    pd.set_option('display.max_rows', None)  # Shows all rows
    pd.set_option('display.max_colwidth', None)  # Shows full content of each column
    pd.set_option('display.width', None)  # Auto-detects the width of the terminal

    # Load the Parquet file
    df = pd.read_parquet('dataset/dataset.parquet', engine='pyarrow')

    # Use regex to filter columns starting with "CV"
    filtered_df = df.filter(regex=r'^CV', axis=1)

    # List of columns to exclude
    exclude_columns = [
        "CV_Image1Name",
        "CV_Image2Name",
        "CV_Image3Name",
        "CV_Image4Name"
    ]

    # 填充缺失值为特定的值，比如0
    filtered_df = filtered_df.fillna(0)

    # Remove the excluded columns
    filtered_df = filtered_df.drop(columns=exclude_columns, errors='ignore')

    # Display the filtered data
    # print(filtered_df)

    return filtered_df
