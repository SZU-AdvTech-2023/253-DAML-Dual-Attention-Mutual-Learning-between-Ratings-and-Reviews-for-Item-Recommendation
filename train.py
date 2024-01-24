import random
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import config
from reviewdata import ReviewData
from frameworks import Model
import models
from predict import unpack_input, predict
import matplotlib.pyplot as plt


def collate_fn(batch):
    data, label = zip(*batch)
    return data, label


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def train(kwargs):
    if 'dataset' not in kwargs:
        opt = getattr(config, 'Digital_Music_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.use_gpu:
        torch.cuda.manual_seed_all(opt.seed)

    if len(opt.gpu_ids) == 0 and opt.use_gpu:
        torch.cuda.set_device(opt.gpu_id)

    model = Model(opt, getattr(models, opt.model))
    if opt.use_gpu:
        model.cuda()
        if len(opt.gpu_ids) > 0:
            model = nn.DataParallel(model, device_ids=opt.gpu_ids)

    if model.net.num_fea != opt.num_fea:
        raise ValueError(f"the num_fea of {opt.model} is error, please specific --num_fea={model.net.num_fea}")

    # 3 data
    train_data = ReviewData(opt.data_root, mode="Train")
    train_data_loader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn)
    val_data = ReviewData(opt.data_root, mode="Val")
    val_data_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f'train data: {len(train_data)}; test data: {len(val_data)}')
    if 'optimizer' not in opt.optimizer:
        optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.optimizer=='SGD':
        optimizer = optim.SGD(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    # training
    print("start training....")

    min_loss = 1e+10
    best_res = 1e+10
    mse_func = nn.MSELoss()
    mae_func = nn.L1Loss()
    smooth_mae_func = nn.SmoothL1Loss()
    loss_list = []
    for epoch in range(opt.num_epochs):
        total_loss = 0.0
        total_maeloss = 0.0
        model.train()
        print(f"{now()}  Epoch {epoch}...")

        # 加入进度条

        # for idx, (train_datas, scores) in (enumerate(train_data_loader)):
        for idx, (train_datas, scores) in enumerate(tqdm(train_data_loader)):
            if opt.use_gpu:
                scores = torch.FloatTensor(scores).cuda()
            else:
                scores = torch.FloatTensor(scores)
            train_datas = unpack_input(opt, train_datas)

            optimizer.zero_grad()
            output = model(train_datas)
            mse_loss = mse_func(output, scores)
            total_loss += mse_loss.item() * len(scores)

            mae_loss = mae_func(output, scores)
            total_maeloss += mae_loss.item()
            smooth_mae_loss = smooth_mae_func(output, scores)
            if opt.loss_method == 'mse':
                loss = mse_loss
            if opt.loss_method == 'rmse':
                loss = torch.sqrt(mse_loss) / 2.0
            if opt.loss_method == 'mae':
                loss = mae_loss
            if opt.loss_method == 'smooth_mae':
                loss = smooth_mae_loss
            loss.backward()
            optimizer.step()
            if opt.fine_step:
                if idx % opt.print_step == 0 and idx > 0:
                    print("\t{}, {} step finised;".format(now(), idx))
                    val_loss, val_mse, val_mae = predict(model, val_data_loader, opt)
                    if val_loss < min_loss:
                        model.save(name=opt.dataset, opt=opt.print_opt)
                        min_loss = val_loss
                        print("\tmodel save")
                    if val_loss > min_loss:
                        best_res = min_loss

        scheduler.step()
        mse = total_loss * 1.0 / len(train_data)
        print(f"\ttrain data: loss:{total_loss:.4f}, mse: {mse:.4f};")
        loss_list.append(total_loss)
        val_loss, val_mse, val_mae = predict(model, val_data_loader, opt)
        if val_loss < min_loss:
            print("model save now")
            model.save(name=opt.dataset, opt=opt.print_opt)
            print("model save end")
            min_loss = val_loss
        if val_mse < best_res:
            best_res = val_mse
        print("*" * 30)
    print("----" * 20)
    print(f"{now()} {opt.dataset} {opt.print_opt} best_res:  {best_res}")
    print("----" * 20)
    #画出loss曲线
    plt.plot(loss_list)
    plt.show()

    pass


# if __name__ == '__main__':
#     train(kwargs={
#         'dataset': 'Musical_Instruments_5',
#         'model': 'DAML',
#         'num_fea': 2,
#         'batch_size': 128,
#         'gpu_id': 2,
#         'filters_num': 10,
#         'output': 'nfm',
#         'num_epochs':30,
#     })
