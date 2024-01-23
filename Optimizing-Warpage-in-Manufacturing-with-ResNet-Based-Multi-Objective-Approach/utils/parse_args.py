import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    # data parameters
    parser.add_argument('--logging', type=str, default='log/logging', help='log')
    parser.add_argument('--seed', type=int, default=2023, help='the number of the seed')
    parser.add_argument('--data_name', type=str, default='PU', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default='Bearing dataset/PU',
                        help='the best model pth of the data')
    parser.add_argument('--transfer_task', type=list, default=[[0], [1]], help='transfer learning tasks')
    parser.add_argument('--normlizetype', type=str, default='mean-std', help='nomalization type')
    parser.add_argument('--batch_size', type=int, default=64, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')
    parser.add_argument('--d_model', type=int, default=16, help='model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='number of heads')
    parser.add_argument('--num_layers', type=int, default=6, help='number of layers')
    parser.add_argument('--dim_feedforward', type=int, default=256, help='dimension of feedforward network')
    parser.add_argument('--num_classes', type=int, default=13, help='number of classes')
    parser.add_argument('--model_name', type=str, default='InformerEncoder', help='the class of the model')
    parser.add_argument('--epoch', type=int, default=10000, help='the epochs of the train')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='the learning_rate of the train')
    parser.add_argument('--gpu_id', type=str, default='1', help='the gpu id of the train')
    parser.add_argument('--input_size', type=int, default=1024, help='the inputsize of the model')
    parser.add_argument('--final_dim', type=int, default=16, help='Time is 16 and Frequency is 8 about CNN')
    parser.add_argument('--work_condition', type=int, default=0, help='The work_condition about PU_type and PUFFT_type')

    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='step',
                        help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--steps', type=str, default='150, 250', help='the learning rate decay for step and stepLR')

    return parser.parse_args()
