import numpy as np
import torch
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.optimize import minimize
from utils.parse_args import parse_args
from utils.seed import set_seeds
from utils.dataloader import optim_simulation_data
import pandas as pd
from models.resnet18 import resnet18_features as ResNet18
from models.resnet18_for_cycletime import resnet18_features as ResNet18_for_cycletime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
e77_cycletime = 22


def is_scalar(x):
    return isinstance(x, (int, float, complex)) and not isinstance(x, bool)


# Function to simulate moldflow analysis (this is a placeholder for your actual analysis function)
def moldflow_simulation(indices):
    # Simulate max. deflection and cycle time based on input parameters
    x = optim_simulation_data(args)
    cycle_time = x.iloc[indices, e77_cycletime]
    max_deflection = x.iloc[indices, -8:-7].sum(axis=0)
    return max_deflection, cycle_time


def deeplearning_predict(inputs):
    inputs = torch.tensor(inputs).float()
    inputs = inputs.to(device).unsqueeze(1).unsqueeze(0).permute(0, 2, 1)
    # print(inputs.shape)

    model_cycle = ResNet18_for_cycletime().to(device)
    # load best cycletime parameter model
    model_cycle.load_state_dict(torch.load('checkpoint/resnet18_best_cycletime_model.pth'))
    model_cycle = model_cycle.to(device)
    outputs_cycletime = model_cycle(inputs)

    model_defect = ResNet18().to(device)
    # load best defect parameter model
    model_defect.load_state_dict(torch.load('checkpoint/resnet18_best_defect_model.pth'))
    model_defect = model_defect.to(device)
    outputs_defect = model_defect(inputs)
    outputs_defect = (outputs_defect.argmax(dim=1).detach().cpu().numpy())

    outputs_cycletime = outputs_cycletime.detach().cpu().numpy().flatten()
    outputs = np.column_stack((outputs_defect, outputs_cycletime))
    outputs = outputs.flatten()

    return outputs


# Acquisition function for Bayesian Optimization
def acquisition_function(x, model, evaluated_loss):
    # Compute the expected improvement (placeholder for actual acquisition function implementation)
    evaluated_loss = np.mean(evaluated_loss)
    mean, std = model.predict(x.reshape(1, -1), return_std=True)
    mean = np.mean(mean)
    std = np.mean(std)
    z = (mean - evaluated_loss) / std
    return (mean - evaluated_loss) * norm.cdf(z) + std * norm.pdf(z)


def x_initial():
    x = optim_simulation_data(args)

    selected_indices = np.random.choice(x.shape[0], size=10, replace=False)
    process_parameters_sample = pd.concat([x.iloc[selected_indices, 9:e77_cycletime], x.iloc[selected_indices, (e77_cycletime+1):52]], axis=1)

    return selected_indices, process_parameters_sample


# LBL_defect average: 0.9485815602836879
# E77_CycleTime average: 54.936560283687925
def optimization_criteria(data_point):

    # set optimization threshold
    lbl_threshold = 0
    e77_threshold = 54.936560283687925
    # (data_point[0] < lbl_threshold) and
    # check the data satisified the cretia
    return (data_point[0] <= lbl_threshold) and (data_point[1] < e77_threshold)


args = parse_args()
set_seeds(seed_value=args.seed)

# Initialize data
indices, X_initial = x_initial()  # Assuming 42 input parameters
X_initial = X_initial.values
Y_initial = np.array([moldflow_simulation(x) for x in indices])  # Assuming max_deflection and cycle_time parameters
Y_initial_copy = Y_initial
# print("X_initial:", X_initial.shape)
# print("Y_initial:", Y_initial.shape)

from sklearn.gaussian_process.kernels import DotProduct
# Define Gaussian Process Regression Model
kernel = ConstantKernel(1e-1) * Matern(length_scale=1e-1, nu=2.5) + WhiteKernel(noise_level=1e-2) + DotProduct()
# from sklearn.gaussian_process.kernels import RBF
#
# kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
# from sklearn.gaussian_process.kernels import RationalQuadratic
#
# kernel = RationalQuadratic(length_scale=1.0, alpha=0.1)
# from sklearn.gaussian_process.kernels import DotProduct
#
# kernel = DotProduct() + WhiteKernel()
# from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct
#
# kernel = RBF(length_scale=1.0) * DotProduct() + WhiteKernel(noise_level=1e-2)
gpr = GaussianProcessRegressor(kernel=kernel)

"""
E77_AreaCavityPressure1Dr1: 156.0 - 6220.0
E77_AreaCavityPressure1Max: 55.0 - 266.0
E77_AreaInjectionSpeed: 77.1 - 140.2
E77_AreaInjectionSpeedMax: 15.6 - 108.6
E77_AreaInjectionStroke: 189.2 - 588.5
E77_AreaMeltPressure: 2065.0 - 12146.0
E77_AreaMeltPressureMax: 189.0 - 631.0
E77_AreaPower: 0.2 - 188.4
E77_AreaPowerMax: 0.203 - 4295.0
E77_BackPressureMax: 114.0 - 320.0
E77_CavityPressure1Max: 55.0 - 267.0
E77_CoolingTime: 14.0 - 26.0
E77_CushionVolume: 5.37 - 25.2
E77_CylinderTemperature01: 189.0 - 251.0
E77_CylinderTemperature02: 194.0 - 255.0
E77_CylinderTemperature03: 195.0 - 260.0
E77_CylinderTemperature04: 200.0 - 260.0
E77_CylinderTemperature11: 199.0 - 261.0
E77_EjectionTime: 2.56 - 3.18
E77_FlangeTemperature: 39.0 - 51.0
E77_HoldingPressureMax: 0.0 - 620.0
E77_HoldingTime: 15.0 - 25.0
E77_InjectionPressureMax: 190.0 - 633.0
E77_InjectionSpeed1: 0.0 - 99.4
E77_InjectionSpeed2: 0.0 - 100.0
E77_InjectionTime: 1.28 - 8.76
E77_InjectionUnitBackwardsTime: 0.52 - 0.84
E77_InjectionUnitForwardsTime: 0.14 - 0.15
E77_MeltPressure2Max: 0.0 - 122.0
E77_OilTemperature: 18.0 - 22.0
E77_PlastificationSpeed: 0.0 - 352.1
E77_PlastificationTime: 10.3 - 14.47
E77_SchlAbb Zeit: 0.37 - 0.41
E77_SchlAuf Zeit: 0.32 - 0.33
E77_ToolClosingTime: 1.58 - 1.75
E77_ToolOpeningTime: 1.78 - 2.02
E77_ToolTemperature1: 0.0 - 85.0
E77_ToolTemperature2: 0.0 - 86.0
E77_TransferPressure: 170.0 - 622.0
E77_TransferSpecificPressure1: 43.0 - 267.0
E77_TransferStroke: 19.6 - 27.69
E77_WaitingTime: 0.01 - 0.02
"""
new_data_points = []
number_of_iterations = 250
# Optimization loop
for iteration in range(number_of_iterations):
    # Fit GPR model
    gpr.fit(X_initial, Y_initial)

    # 创建包含参数上下界的列表
    param_bounds = [(156.0, 6220.0), (55.0, 266.0), (77.1, 140.2), (15.6, 108.6),
                    (189.2, 588.5), (2065.0, 12146.0), (189.0, 631.0), (0.2, 188.4),
                    (0.203, 4295.0), (114.0, 320.0), (55.0, 267.0), (14.0, 26.0),
                    (5.37, 25.2), (189.0, 251.0), (194.0, 255.0), (195.0, 260.0),
                    (200.0, 260.0), (199.0, 261.0), (2.56, 3.18), (39.0, 51.0),
                    (0.0, 620.0), (15.0, 25.0), (190.0, 633.0), (0.0, 99.4),
                    (0.0, 100.0), (1.28, 8.76), (0.52, 0.84), (0.14, 0.15), (0.0, 122.0),
                    (18.0, 22.0), (0.0, 352.1), (10.3, 14.47), (0.37, 0.41), (0.32, 0.33),
                    (1.58, 1.75), (1.78, 2.02), (0.0, 85.0), (0.0, 86.0), (170.0, 622.0),
                    (43.0, 267.0), (19.6, 27.69), (0.01, 0.02)]

    x_next = minimize(lambda x: -acquisition_function(x, gpr, np.min(Y_initial)),
                      x0=X_initial[1], bounds=param_bounds)


    # Obtain new data point from moldflow analysis
    new_data_point = deeplearning_predict(x_next.x)
    new_data_points.append(new_data_point)
    # Check if optimization criteria are satisfied
    optim = optimization_criteria(new_data_point)
    if optim:
        print("Optimization criteria met.")
    else:
        print("Optimization criteria not met.")

    # Add the new data point to the dataset
    X_initial = np.vstack((X_initial, x_next.x))
    new_data_point = new_data_point.reshape(1, 2)
    Y_initial = np.concatenate((Y_initial, new_data_point))

# new_data_points = np.array(new_data_points)
# file_path = '../exper_txt/new_data_points.txt'
# np.savetxt(file_path, new_data_points, delimiter=',', header='X-axis, Y-axis', comments='')

# plt.scatter(new_data_points[:, 0], new_data_points[:, 1], c='blue', marker='o', label='Optimum data')
plt.scatter(Y_initial_copy[:, 0], Y_initial_copy[:, 1], c='red', marker='x', label='Initial (10 data)')
plt.xlabel('Defect')
plt.ylabel('Cycle time')
plt.title('')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig("Initial optimization figure")
# 显示图形
# plt.show()

# Optimized process parameters
optimized_params = X_initial[np.argmin(Y_initial), :]
print(optimized_params)
optimized_params = np.array(optimized_params)
file_path_sec = 'exper_txt/optimized_params.txt'
np.savetxt(file_path_sec, optimized_params)