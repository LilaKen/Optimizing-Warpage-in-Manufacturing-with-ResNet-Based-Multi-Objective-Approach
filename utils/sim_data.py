"""
Simulation data (SIM)
    To generate additional data from simulation,
    Moldex3D 2021 (a plastic injection molding simulation software) was used.
    The same test series were carried out in the simulation environment as on the real injection molding machine.
    The simulated data contains the part weight, geometrical dimensions of the part
    (corresponding to the real parameters with prefix "CV"),
    temperatures of the demolded part (corresponding to the real parameters with prefix "IR")
    and some process parameters.
    It should be noted that the simulated dimensions correspond to those of the part cooled down to room temperature,
    while the real measured dimensions ("CV") are determined shortly after demolding on the still warm part.
    This influence must be taken into account,
    since even more warpage and shrinkage can take place until the part has cooled down to room temperature.
    Some of the simulated parameters are listed below.
    Please note that certain simulation data are missing for some test series.
"""
import pandas as pd


def sim_data():
    # Set Pandas display options to show all content
    pd.set_option('display.max_columns', None)  # Shows all columns
    pd.set_option('display.max_rows', None)  # Shows all rows
    pd.set_option('display.max_colwidth', None)  # Shows full content of each column
    pd.set_option('display.width', None)  # Auto-detects the width of the terminal

    # Load the Parquet file
    df = pd.read_parquet('dataset/dataset.parquet', engine='pyarrow')

    # Use regex to filter columns starting with "SIM"
    filtered_df = df.filter(regex=r'^SIM', axis=1)

    # 填充缺失值为特定的值，比如0
    filtered_df = filtered_df.fillna(0)

    # Display the filtered data
    # print(filtered_df)

    return filtered_df
