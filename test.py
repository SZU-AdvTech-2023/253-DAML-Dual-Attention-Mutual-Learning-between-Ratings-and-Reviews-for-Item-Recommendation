# -*- encoding: utf-8 -*-
import time
import random
import math
import fire

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from reviewdata import ReviewData
from frameworks import Model
import models
from config import config
from predict import predict
from train import now, collate_fn


def test(kwargs):
    if 'dataset' not in kwargs:
        opt = getattr(config, 'Digital_Music_data_Config')()
    else:
        opt = getattr(config, kwargs['dataset'] + '_Config')()
    opt.parse(kwargs)
    assert (len(opt.pth_path) > 0)
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
    print(opt.pth_path)
    model.load(opt.pth_path)
    print(f"load model: {opt.pth_path}")
    test_data = ReviewData(opt.data_root, mode="Test")
    test_data_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"{now()}: test in the test datset")
    predict_loss, test_mse, test_mae = predict(model, test_data_loader, opt)


# if __name__ == '__main__':
#     test(
#         kwargs={
#             'dataset': 'Musical_Instruments_5',
#             'model': 'DAML',
#             'num_fea': 2,
#             'batch_size': 128,
#             'gpu_id': 2,
#             'filters_num': 10,
#             'output': 'nfm',
#             'num_epochs': 30,
#             'pth_path' : './checkpoints/DAML_Musical_Instruments_5_default.pth'

#         }
#     )
