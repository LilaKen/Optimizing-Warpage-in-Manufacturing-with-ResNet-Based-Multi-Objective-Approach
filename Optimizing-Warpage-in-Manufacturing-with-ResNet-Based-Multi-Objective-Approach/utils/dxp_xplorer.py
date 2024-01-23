"""
DataXplorer (DXP)
    The dataXplorer is a preconfigured data storage device for automatic recording of sensor and
    control data of an injection molding machine, in whose control cabinet the hardware and software are integrated.
    The naming of the parameters was done by the machine manufacturer KraussMaffei.
    A list to look up the parameter names (German and English) is appended to this data description.

    Values are recorded for the parameters at a frequency of 0.005 seconds,
    i.e. the data are available here as time series.
    Since each row in the data set corresponds to one injection molding cycle,
    the time series associated with the various parameters were each packed into an array and stored in a cell.
    In addition to the classic parameters such as pressures or temperatures, there are also various triggers.
    These triggers are either true (= 1) or false (= 0). All the different triggers are listed below.
"""
import pandas as pd


def dxp_xplorer():
    # Set Pandas display options to show all content
    pd.set_option('display.max_columns', None)  # Shows all columns
    pd.set_option('display.max_rows', None)  # Shows all rows
    pd.set_option('display.max_colwidth', None)  # Shows full content of each column
    pd.set_option('display.width', None)  # Auto-detects the width of the terminal

    # Load the Parquet file
    df = pd.read_parquet('dataset/dataset.parquet', engine='pyarrow')

    # Use regex to filter columns starting with "DXP"
    filtered_df = df.filter(regex=r'^DXP', axis=1)

    # 填充缺失值为特定的值，比如0
    filtered_df = filtered_df.fillna(0)

    # Display the filtered data
    print(filtered_df[:5])

    return filtered_df


if __name__ == '__main__':
    dxp_xplorer()
