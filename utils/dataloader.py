from utils.dxp_xplorer import dxp_xplorer
from utils.e77_machine import e77_machine
from utils.env_data import env_set
from utils.lbl_defect import lbl_defect
from utils.machine_set import machine_set
from utils.meta_data import meta_data
from utils.product_size import product_size
from utils.sca_data import sca_data
from utils.sim_data import sim_data
from utils.thermal_data import thermal_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np

"""
Simulation(SIM):
    meta_data:
    env_data:
    machine_set:
    e77_machine:
    product_size
"""


def simulation_defect_data(args):
    meta_set_data = meta_data()
    env_set_data = env_set()
    machine_set_data = machine_set()
    e77_machine_set_data = e77_machine()
    product_size_data = product_size()

    lbl_defect_data = lbl_defect()
    # Concatenate the dataframes vertically (along rows)
    x_sim = pd.concat(
        [e77_machine_set_data, lbl_defect_data],
        axis=1)

    # 排除非数值列
    excluded_features = ['MET_Timestamp', 'MET_MaterialName', 'MET_ExperimentNumber', 'MET_MachineCycleID',
                         'MET_JobCycleID', 'LBL_NOK', 'LBL_OldGranulate', 'LBL_SinkMarks', 'LBL_SprueCircle',
                         'LBL_StreaksLevel1',
                         'LBL_StreaksLevel2', 'LBL_StreaksLevel3', 'LBL_Underfilled', 'E77_CycleTime']
    # 将分类特征'PP'和'ABS'映射为0和1
    # x_sim['MET_MaterialName'] = x_sim['MET_MaterialName'].replace({'PP': 0, 'ABS': 1})

    # 现在'MET_MaterialName'已转换为数值，可以将其添加到数值特征中
    numeric_features = [col for col in x_sim.columns if col not in excluded_features]
    # print(len(numeric_features))

    # Reset the index if needed
    x_sim.reset_index(drop=True, inplace=True)

    # 构建预处理管道
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
        ])

    # 提取LBL标签列
    lbl_columns = [col for col in x_sim.columns if col.startswith('LBL_NOK')]
    lbl_data = x_sim[lbl_columns]

    # 转换标签为单个类别标签
    y = lbl_data  # 将列名转换为类别代码

    # 删除LBL标签列，保留其他特征作为X
    X = x_sim.drop(lbl_columns, axis=1)

    # 划分数据集为训练集和测试集（80% 训练集，20% 测试集）
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 划分测试集为测试集和验证集（50% 测试集，50% 验证集）
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

    # 应用预处理
    # X_train = preprocessor.fit_transform(X_train[numeric_features])
    # X_val = preprocessor.transform(X_val[numeric_features])
    # X_test = preprocessor.transform(X_test[numeric_features])

    # not norm
    X_train_tensor = torch.tensor(X_train[numeric_features].to_numpy().astype(np.float32))
    X_val_tensor = torch.tensor(X_val[numeric_features].to_numpy().astype(np.float32))
    X_test_tensor = torch.tensor(X_test[numeric_features].to_numpy().astype(np.float32))
    y_train_tensor = torch.tensor(y_train.values.astype(np.float32))
    y_val_tensor = torch.tensor(y_val.values.astype(np.float32))
    y_test_tensor = torch.tensor(y_test.values.astype(np.float32))

    # norm
    # X_train_tensor = torch.tensor(X_train.astype(np.float32))
    # X_val_tensor = torch.tensor(X_val.astype(np.float32))
    # X_test_tensor = torch.tensor(X_test.astype(np.float32))
    # y_train_tensor = torch.tensor(y_train.values.astype(np.float32))
    # y_val_tensor = torch.tensor(y_val.values.astype(np.float32))
    # y_test_tensor = torch.tensor(y_test.values.astype(np.float32))

    # 创建 TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def simulation_cycletime_data(args):
    meta_set_data = meta_data()
    env_set_data = env_set()
    machine_set_data = machine_set()
    e77_machine_set_data = e77_machine()
    product_size_data = product_size()

    # Concatenate the dataframes vertically (along rows)
    x_sim = pd.concat(
        [e77_machine_set_data],
        axis=1)

    # 排除非数值列
    excluded_features = ['MET_Timestamp', 'MET_MaterialName', 'MET_ExperimentNumber', 'MET_MachineCycleID',
                         'MET_JobCycleID', 'E77_CycleTime']
    # 将分类特征'PP'和'ABS'映射为0和1
    # x_sim['MET_MaterialName'] = x_sim['MET_MaterialName'].replace({'PP': 0, 'ABS': 1})

    # 现在'MET_MaterialName'已转换为数值，可以将其添加到数值特征中
    numeric_features = [col for col in x_sim.columns if col not in excluded_features]
    # print(len(numeric_features))

    # Reset the index if needed
    x_sim.reset_index(drop=True, inplace=True)

    # 构建预处理管道
    # preprocessor = ColumnTransformer(
    #     transformers=[
    #         ('num', StandardScaler(), numeric_features),
    #     ])

    # 提取E77_CycleTime标签列
    cycletime_columns = [col for col in x_sim.columns if col.startswith('E77_CycleTime')]
    cycletime_data = x_sim[cycletime_columns]

    # 转换标签为单个类别标签
    y = cycletime_data  # 将列名转换为类别代码

    # 删除E77_CycleTime标签列，保留其他特征作为X
    X = x_sim.drop(cycletime_columns, axis=1)

    # 划分数据集为训练集和测试集（80% 训练集，20% 测试集）
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 划分测试集为测试集和验证集（50% 测试集，50% 验证集）
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

    # 应用预处理
    # X_train = preprocessor.fit_transform(X_train[numeric_features])
    # X_val = preprocessor.transform(X_val[numeric_features])
    # X_test = preprocessor.transform(X_test[numeric_features])

    # not norm
    X_train_tensor = torch.tensor(X_train[numeric_features].to_numpy().astype(np.float32))
    X_val_tensor = torch.tensor(X_val[numeric_features].to_numpy().astype(np.float32))
    X_test_tensor = torch.tensor(X_test[numeric_features].to_numpy().astype(np.float32))
    y_train_tensor = torch.tensor(y_train.values.astype(np.float32))
    y_val_tensor = torch.tensor(y_val.values.astype(np.float32))
    y_test_tensor = torch.tensor(y_test.values.astype(np.float32))

    # norm
    # X_train_tensor = torch.tensor(X_train.astype(np.float32))
    # X_val_tensor = torch.tensor(X_val.astype(np.float32))
    # X_test_tensor = torch.tensor(X_test.astype(np.float32))
    # y_train_tensor = torch.tensor(y_train.values.astype(np.float32))
    # y_val_tensor = torch.tensor(y_val.values.astype(np.float32))
    # y_test_tensor = torch.tensor(y_test.values.astype(np.float32))

    # 创建 TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


"""
OptimSimulation(OSIM):
    meta_data:
    env_data:
    machine_set:
    e77_machine:
    product_size:
    lbl_defect:
"""


def optim_simulation_data(args):
    meta_set_data = meta_data()
    env_set_data = env_set()

    # SET_CylinderTemperature、SET_ToolTemperature 工艺参数
    machine_set_data = machine_set()

    # E77_CycleTime、E77_InjectionTime 工艺参数
    e77_machine_set_data = e77_machine()

    # size of product 成型产品尺寸参数
    product_size_data = product_size()

    # LBL_SinkMarks、LBL_SprueCircle 缺陷参数
    lbl_defect_data = lbl_defect()

    # print(f"Meta data columns: {meta_set_data.shape[1] - 5}") 1
    # print(f"Machine setting data columns: {machine_set_data.shape[1]}") 8
    # print(f"E77 machine setting data columns: {e77_machine_set_data.shape[1]}") 43
    # print(f"Product size data columns: {product_size_data.shape[1]}")
    # print(f"Environmental data columns: {env_set_data.shape[1]}")
    # print(f"Label defect data columns: {lbl_defect_data.shape[1]}")

    # Concatenate the dataframes vertically (along rows)
    x = pd.concat(
        [meta_set_data, machine_set_data, e77_machine_set_data, product_size_data, env_set_data, lbl_defect_data],
        axis=1)

    # 排除非数值列
    excluded_features = ['MET_Timestamp', 'MET_MaterialName', 'MET_ExperimentNumber', 'MET_MachineCycleID',
                         'MET_JobCycleID']
    # 将分类特征'PP'和'ABS'映射为0和1
    x['MET_MaterialName'] = x['MET_MaterialName'].replace({'PP': 0, 'ABS': 1})

    # 现在'MET_MaterialName'已转换为数值，可以将其添加到数值特征中
    numeric_features = [col for col in x.columns if col not in excluded_features]
    # print(len(numeric_features))

    # numeric_features.append('MET_MaterialName')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features)
        ])

    # 应用预处理
    # x = preprocessor.fit_transform(x)

    # 未进行预处理
    x = x[numeric_features]
    return x


