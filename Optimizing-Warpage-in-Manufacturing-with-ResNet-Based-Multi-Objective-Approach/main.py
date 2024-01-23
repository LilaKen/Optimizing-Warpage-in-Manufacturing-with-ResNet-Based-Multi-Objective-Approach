import os
from utils.parse_args import parse_args
from utils.seed import set_seeds
from train import train, train_cycletime

if __name__ == '__main__':
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    set_seeds(seed_value=args.seed)
    train(args)
    train_cycletime(args)
