
import numpy as np, argparse, time, random
import math
import torch
import os

from trainer import train_or_eval_model
from dataloader import get_data_loaders
from model import *

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

def plt_fscore_and_loss(all_test_fscore, all_train_loss, all_valid_loss, all_test_loss, test_fscore, train_loss, valid_loss, test_loss):
    if not os.path.exists('./pic/'):
        os.makedirs('./pic/')
    all_test_fscore.append(test_fscore)
    all_train_loss.append(train_loss)
    all_valid_loss.append(valid_loss)
    all_test_loss.append(test_loss)
    plt_fscore(all_test_fscore) 
    plt_loss(all_train_loss, all_valid_loss, all_test_loss)

def plt_fscore(values):
    indices = np.arange(len(values))
    plt.figure()  # 创建新的图形
    plt.plot(indices, values)
    plt.title('test_fscore over Training Steps')
    plt.xlabel('Training Step')
    plt.ylabel("test_fscore")
    plt.savefig('pic/test_fscore.png')  # 保存到当前目录


def plt_loss(train_loss, valid_loss, test_loss):
    indices = np.arange(len(train_loss))
    plt.figure()  # 创建新的图形
    plt.plot(indices, train_loss, color="green")  # 训练损失 绿色
    plt.plot(indices, valid_loss, color="blue")  # 验证损失 蓝色
    plt.plot(indices, test_loss, color="red")    # 测试损失 红色

    plt.title('loss over Training Steps')
    plt.xlabel('Training Step')
    plt.ylabel("loss")
    plt.savefig('pic/loss.png')

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--hidden_dim', type=int, default=300, help='hidden_dim')
    parser.add_argument('--gnn_layers', type=int, default=2, help='Number of gnn layers.')

    parser.add_argument('--emb_dim', type=int, default=768, help='Text Feature size.')
    parser.add_argument('--audio_emb_dim', type=int, default=512, help='Audio Feature size.')
    parser.add_argument('--visual_emb_dim', type=int, default=1000, help='Visual Feature size.')

    parser.add_argument('--dataset_name', default='MELD', type=str, help='dataset name, IEMOCAP, MELD')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    

    parser.add_argument('--mlp_layers', type=int, default=2, help='Number of output mlp layers.')

    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR', help='learning rate')  #####
    parser.add_argument('--dropout', type=float, default=0.2, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=32, metavar='BS', help='batch size') ##
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')

    # 自定义参数
    parser.add_argument('--num_experts', type=int, default=3, help='num_experts')
    parser.add_argument('--moe_routing_method', type=str, default='soft', help='soft or hard')
    parser.add_argument('--moe_hidden_dim', type=int, default=300, help='moe_hidden_dim')
    parser.add_argument('--temperature', type=float, default=0.5, help='temperature')

    # use model
    parser.add_argument('--load_save_model', type=int, default=0, help='load_save_model. 0:not load model 1:load model')

    # loss部分
    parser.add_argument('--use_diff_loss', type=bool, default=True, help='use_diff_loss')
    parser.add_argument('--use_nce_loss', type=bool, default=True, help='use_nce_loss')
    parser.add_argument('--use_moe_lb_loss', type=bool, default=True, help='use_moe_loss')
    # nce使用使用单模态
    parser.add_argument('--nce_single_or_cross', type=int, default=2, help='nce_single_or_cross,1:single_mode, 2:cross_mode')

    parser.add_argument('--seed', type=int, default=100, help='random seed') ##

    return parser.parse_args()


if __name__ == '__main__':

    path = './saved_models/'  # 模型保存路径
    if not os.path.exists(path):
        os.makedirs(path)

    args = parse_args()
    print(args)
    
    # 固定随机种子
    seed_everything(args.seed)

    

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", args.device)

    device = args.device
    n_epochs = args.epochs
    batch_size = args.batch_size

    train_loader, valid_loader, test_loader = get_data_loaders(
        dataset_name=args.dataset_name, batch_size=batch_size, num_workers=0, args=args)


    # n_classes
    n_classes = 6 if 'IEMOCAP' in args.dataset_name else 7

    print('building model..')
    model = MMGATs(args, n_classes)
    if torch.cuda.device_count() > 1:
        print('Multi-GPU...........')
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.to(device)

    # loss函数
    loss_function = nn.CrossEntropyLoss(ignore_index=-1) # 忽略掉label=-1 的类
    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=args.lr)
    # 定义学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, factor=0.95, patience=6, threshold=1e-6, verbose=True)

    all_fscore, all_test_fscore = [], []
    all_train_loss, all_valid_loss, all_test_loss = [], [], []
    best_epoch, best_test_fscore, best_train_acc = 0, 0, 0
    best_model = None

    # 如果有已保存的最佳模型，路径设置
    model_full_path_name = path + '{}_best_model_2_moe_{}_{}.pth'.format(args.dataset_name, args.num_experts, args.moe_routing_method)
    # 如果有已保存的最佳模型，选择是否加载
    if args.load_save_model==1 and os.path.exists(model_full_path_name):
        print("loading model...")
        model.load_state_dict(torch.load(model_full_path_name))
        # 计算模型的验证集的fscore
        valid_loss, valid_acc, _, _, valid_fscore = train_or_eval_model(model, loss_function, valid_loader, device, args, print_loss=False)
        test_loss, test_acc, _, _, test_fscore    = train_or_eval_model(model, loss_function, test_loader,  device, args, print_loss=False)
        best_test_fscore = test_fscore
        plt_fscore_and_loss(all_test_fscore, all_train_loss, all_valid_loss, all_test_loss, test_fscore, 0, valid_loss, test_loss)
        print("loaded best model from saved checkpoint. best_test_fscore={}, best_epoch={}".format(best_test_fscore, best_epoch))

    for e in range(n_epochs):  # 遍历每个epoch
        start_time = time.time()

        train_loss, train_acc, _, _, train_fscore = train_or_eval_model(model, loss_function, train_loader, device, args, optimizer, True)
        valid_loss, valid_acc, _, _, valid_fscore = train_or_eval_model(model, loss_function, valid_loader, device, args)
        test_loss, test_acc, _, _, test_fscore    = train_or_eval_model(model, loss_function, test_loader,  device, args)

        # 根据验证集loss调整
        scheduler.step(valid_loss)

        all_fscore.append([valid_fscore, test_fscore])
        
        # 画loss和test_fscore的图
        plt_fscore_and_loss(all_test_fscore, all_train_loss, all_valid_loss, all_test_loss, test_fscore, train_loss, valid_loss, test_loss)

        if best_test_fscore < test_fscore:
            best_test_fscore = test_fscore
            best_epoch = e+1
            # 保存最优模型
            torch.save(model.state_dict(), model_full_path_name)
            print(f"Best model saved at epoch {e + 1}.")

        best_train_acc = max(best_train_acc, train_acc)

        print('Epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, valid_loss: {}, valid_acc: {}, valid_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec, best_epoch: {}, best_test_fscore: {}, best_train_acc: {}'. \
            format(e + 1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, valid_fscore, test_loss, test_acc, test_fscore, round(time.time() - start_time, 2), best_epoch, best_test_fscore, best_train_acc))

        e += 1

    print('finish training!')
    plt_fscore(all_test_fscore)
    plt_loss(all_train_loss, all_valid_loss, all_test_loss)

    all_fscore = sorted(all_fscore, key=lambda x: (x[0], x[1]), reverse=True)  # 优先按照验证集 f1 进行排序

    print('Best val F-Score:{}'.format(all_fscore[0][0]))  # 验证集最好性能 
    print('Best test F-Score based on validation:{}'.format(all_fscore[0][1]))  # 验证集取得最好性能时 对应测试集的下性能
    print('Best test F-Score based on test:{}'.format(max([f[1] for f in all_fscore])))  # 测试集 最好的性能
    print('Best test F-Score is epoch={}'.format(best_epoch))  # 测试集 最好性能的epoch
    print("over")





