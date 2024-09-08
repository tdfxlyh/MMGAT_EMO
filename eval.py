
import numpy as np, argparse, time, random
import math
import torch
import os

from main import parse_args, seed_everything
from trainer import train_or_eval_model
from dataloader import get_data_loaders
from model import *

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    args = parse_args()
    print(args)
    # 固定随机种子
    seed_everything(args.seed)

    # 创建模型路径
    path = './saved_models/'  
    if not os.path.exists(path):
        os.makedirs(path)
    # 如果有已保存的最佳模型，路径设置
    model_full_path_name = path + '{}_best_model_2_moe_{}_{}.pth'.format(args.dataset_name, args.num_experts, args.moe_routing_method)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", args.device)

    device = args.device
    n_epochs = args.epochs
    batch_size = args.batch_size

    train_loader, valid_loader, test_loader = get_data_loaders(dataset_name=args.dataset_name, batch_size=batch_size, num_workers=0, args=args)

    # n_classes
    n_classes = 6 if 'IEMOCAP' in args.dataset_name else 7

    print('building model..')
    model = MMGATs(args, n_classes)
    model.to(device)

    # loss函数
    loss_function = nn.CrossEntropyLoss(ignore_index=-1) # 忽略掉label=-1 的类
    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=args.lr)
    # 定义学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, factor=0.95, patience=6, threshold=1e-6, verbose=True)

    # 如果有已保存的最佳模型，选择是否加载
    if os.path.exists(model_full_path_name):
        print('loading model..')
        model.load_state_dict(torch.load(model_full_path_name))
        # 计算模型的验证集的fscore
        _, _, _, _, test_fscore = train_or_eval_model(model, loss_function, test_loader, device, args, print_loss=False)
        print("->>>>>  Loaded best model from saved checkpoint. test fscore: {}".format(test_fscore))
    else:
        print("->>>>>  No saved model found. Training model...")

    print("over")





