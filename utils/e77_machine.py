"""
EUROMAP77 (E77)
    The EUROMAP77 is equivalent to the OPC-UA 40077 Companion Specification.
    Check out the documentation here for more details about the parameters.
    Also the EUROMAP77 is built on top of EUROMAP83, so some parameter descriptions can be found here.
    Some example parameters are listed below.
    All parameters with prefix E77 are single values, i.e. there is exactly one value for each cycle.
"""
import pandas as pd


def e77_machine():
    # Set Pandas display options to show all content
    pd.set_option('display.max_columns', None)  # Shows all columns
    pd.set_option('display.max_rows', None)  # Shows all rows
    pd.set_option('display.max_colwidth', None)  # Shows full content of each column
    pd.set_option('display.width', None)  # Auto-detects the width of the terminal

    # Load the Parquet file
    df = pd.read_parquet('../dataset/dataset.parquet', engine='pyarrow')

    # Use regex to filter columns starting with "E77"
    filtered_df = df.filter(regex=r'^E77', axis=1)

    # 填充缺失值为特定的值，比如0
    filtered_df = filtered_df.fillna(0)

    # Display the filtered data
    # print(filtered_df)

    return filtered_df


if __name__ == '__main__':
    x = e77_machine()
    # 计算平均值
    # average_cycle_time = sum(x['E77_CycleTime']) / len(x['E77_CycleTime'])
    #
    # # 打印平均值
    # print("E77_CycleTime 平均值:", average_cycle_time)

    column_means = x.mean()
    print(column_means)