import pandas as pd

# 读取 Parquet 文件
df = pd.read_parquet('../dataset/dataset.parquet', engine='pyarrow')

# 打印所有的列名
print(df.columns.tolist())