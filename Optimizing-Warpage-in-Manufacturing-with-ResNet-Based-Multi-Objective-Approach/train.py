from utils.dataloader import simulation_defect_data, simulation_cycletime_data
import torch
import torch.nn as nn
import torch.optim as optim
from models.resnet18 import resnet18_features as ResNet18
from models.resnet18_for_cycletime import resnet18_features as ResNet18_for_cycletime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from tabulate import tabulate


def train(args):
    # 检查是否有可用的CUDA设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    train_loader, val_loader, test_loader = simulation_defect_data(args)

    # 创建模型并移到适当的设备
    model = ResNet18().to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    #
    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    else:
        raise Exception("optimizer not implement")

    # Define the learning rate decay
    if args.lr_scheduler == 'step':
        steps = [int(step) for step in args.steps.split(',')]
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma=args.gamma)
    elif args.lr_scheduler == 'exp':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)
    elif args.lr_scheduler == 'stepLR':
        steps = int(args.steps)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, steps, args.gamma)
    elif args.lr_scheduler == 'fix':
        lr_scheduler = None
    else:
        raise Exception("lr schedule not implement")

    # 初始化最佳验证损失为正无穷
    best_val_loss = float('inf')
    best_model_state = None

    # 训练模型
    num_epochs = args.epoch  # 迭代次数
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            # 将数据移到适当的设备
            inputs = inputs.to(device).unsqueeze(1)
            targets = targets.to(device)

            # 前向传播
            outputs = model(inputs)
            outputs = outputs.squeeze(-1)
            targets = targets.squeeze(-1)
            loss = criterion(outputs, targets)

            # 后向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device).unsqueeze(1)
                targets = targets.to(device)

                outputs = model(inputs)
                outputs = outputs.squeeze(-1)
                targets = targets.squeeze(-1)
                val_loss += criterion(outputs, targets).item()
            val_loss /= len(val_loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

            # 检查是否为最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()

    # 保存最佳模型
    torch.save(best_model_state, 'checkpoint/resnet18_best_defect_model.pth')
    #
    # # 加载最优模型
    model.load_state_dict(torch.load('checkpoint/resnet18_best_defect_model.pth'))
    model = model.to(device)

    # 测试
    y_true, y_pred = [], []
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device).unsqueeze(1)
            targets = targets.to(device)

            outputs = model(inputs)
            # outputs = outputs.squeeze(-1)
            # targets = targets.squeeze(-1)
            test_loss += criterion(outputs, targets).item()

            # 收集真实标签和预测标签
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(outputs.argmax(dim=1).cpu().numpy())

    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.4f}')

    # 将列表转换为NumPy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_true_single = np.argmax(y_true, axis=1)

    # 计算指标
    accuracy = accuracy_score(y_true_single, y_pred)
    precision = precision_score(y_true_single, y_pred, average='macro')  # 'macro' 用于多分类
    recall = recall_score(y_true_single, y_pred, average='macro', zero_division=1)  # 'macro' 用于多分类
    f1 = f1_score(y_true_single, y_pred, average='macro')  # 'macro' 用于多分类

    # print("Accuracy:", accuracy)
    # print("Precision:", precision)
    # print("Recall:", recall)
    # print("F1 Score:", f1)

    # 将性能指标放入 Pandas DataFrame
    data = {'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Value': [accuracy, precision, recall, f1]}

    df = pd.DataFrame(data)

    # 输出为表格
    table = tabulate(df, headers='keys', tablefmt='pretty', showindex=False)

    print(table)


def train_cycletime(args):
    # 检查是否有可用的CUDA设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    train_loader, val_loader, test_loader = simulation_cycletime_data(args)

    # 创建模型并移到适当的设备
    model = ResNet18_for_cycletime().to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()

    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    else:
        raise Exception("optimizer not implement")

    # Define the learning rate decay
    if args.lr_scheduler == 'step':
        steps = [int(step) for step in args.steps.split(',')]
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma=args.gamma)
    elif args.lr_scheduler == 'exp':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)
    elif args.lr_scheduler == 'stepLR':
        steps = int(args.steps)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, steps, args.gamma)
    elif args.lr_scheduler == 'fix':
        lr_scheduler = None
    else:
        raise Exception("lr schedule not implement")

    # 初始化最佳验证损失为正无穷
    best_val_loss = float('inf')
    best_model_state = None
    #
    # # 训练模型
    num_epochs = args.epoch  # 迭代次数
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            # 将数据移到适当的设备
            inputs = inputs.to(device).unsqueeze(1)
            targets = targets.to(device)

            # 前向传播
            outputs = model(inputs)
            outputs = outputs

            loss = criterion(outputs, targets)

            # 后向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device).unsqueeze(1)
                targets = targets.to(device)
                outputs = model(inputs)

                outputs = outputs
                val_loss += criterion(outputs, targets).item()
            val_loss /= len(val_loader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

            # 检查是否为最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()

    # 保存最佳模型
    torch.save(best_model_state, 'checkpoint/resnet18_best_cycletime_model_norm.pth')

    # 加载最优模型
    model.load_state_dict(torch.load('checkpoint/resnet18_best_cycletime_model_norm.pth'))
    model = model.to(device)
    #
    # # 测试
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device).unsqueeze(1)
            targets = targets.to(device)
            outputs = model(inputs)
            outputs = outputs

            test_loss += criterion(outputs, targets).item()

    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.4f}')


