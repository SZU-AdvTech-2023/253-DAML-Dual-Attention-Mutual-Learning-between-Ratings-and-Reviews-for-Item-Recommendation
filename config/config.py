# -*- coding: utf-8 -*-

import numpy as np


class DefaultConfig:
    model = 'DAML'
    dataset = 'Digital_Music_data'

    # -------------base config-----------------------#
    use_gpu = True
    gpu_id = 2
    multi_gpu = False
    gpu_ids = []

    seed = 2019
    num_epochs = 15
    num_workers = 0

    optimizer = 'Adam'
    weight_decay = 1e-3  # optimizer rameteri
    lr = 1e-4
    loss_method = 'mse'
    drop_out = 0.2

    use_word_embedding = True

    id_emb_size = 8
    query_mlp_size = 128  # useless
    fc_dim = 8

    doc_len = 500
    filters_num = 100
    kernel_size = 3

    num_fea = 1  # id feature, review feature, doc feature
    use_review = True
    use_doc = True
    self_att = False

    r_id_merge = 'cat'  # review and ID feature
    ui_merge = 'cat'  # cat/add/dot
    output = 'lfm'  # 'fm', 'lfm', 'other: sum the ui_feature'

    fine_step = False  # save mode in step level, defualt in epoch
    pth_path = ""  # the saved pth path for test
    print_opt = 'default'

    def set_path(self, name):
        '''
        specific
        '''
        self.data_root = f'./dataset/{name}'
        prefix = f'{self.data_root}/train'

        self.user_list_path = f'{prefix}/userReview2Index.npy'
        self.item_list_path = f'{prefix}/itemReview2Index.npy'

        self.user2itemid_path = f'{prefix}/user_item2id.npy'
        self.item2userid_path = f'{prefix}/item_user2id.npy'

        self.user_doc_path = f'{prefix}/userDoc2Index.npy'
        self.item_doc_path = f'{prefix}/itemDoc2Index.npy'

        self.w2v_path = f'{prefix}/w2v.npy'

    def parse(self, kwargs):
        '''
        user can update the default hyperparamter
        '''
        print("load npy from dist...")
        self.users_review_list = np.load(self.user_list_path, encoding='bytes')
        self.items_review_list = np.load(self.item_list_path, encoding='bytes')
        self.user2itemid_list = np.load(self.user2itemid_path, encoding='bytes')
        self.item2userid_list = np.load(self.item2userid_path, encoding='bytes')
        self.user_doc = np.load(self.user_doc_path, encoding='bytes')
        self.item_doc = np.load(self.item_doc_path, encoding='bytes')

        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and k != 'user_list' and k != 'item_list':
                print("{} => {}".format(k, getattr(self, k)))
        print('*************************************************')


class Digital_Music_data_Config(DefaultConfig):

    def __init__(self):
        self.set_path('Digital_Music_5')

    vocab_size = 50002
    word_dim = 300

    r_max_len = 202

    u_max_r = 13
    i_max_r = 24

    train_data_size = 51764
    test_data_size = 6471
    val_data_size = 6471

    user_num = 5541 + 2
    item_num = 3568 + 2

    batch_size = 16
    print_step = 100


class Musical_Instruments_5_Config(DefaultConfig):

    def __init__(self):
        self.set_path('Musical_Instruments_5')
    vocab_size = 50002
    word_dim = 100

    r_max_len = 75

    u_max_r = 9
    i_max_r = 24

    train_data_size = 185121
    test_data_size = 23111
    val_data_size = 23112

    user_num = 27528 + 2
    item_num = 10620 + 2

    batch_size = 16
    print_step = 100

    num_heads = 2


class Sports_and_Outdoors_5_Config(DefaultConfig):

    def __init__(self):
        self.set_path('Sports_and_Outdoors_5')

    vocab_size = 50002
    word_dim = 100

    r_max_len = 62

    u_max_r = 10
    i_max_r = 30

    train_data_size = 2271305
    test_data_size = 283760
    val_data_size = 283761

    user_num = 332421 + 2
    item_num = 104687 + 2

    batch_size = 16
    print_step = 100

    num_heads = 2


class Video_Games_5_Config(DefaultConfig):

    def __init__(self):
        self.set_path('Video_Games_5')

    vocab_size = 50002
    word_dim = 100

    r_max_len = 172

    u_max_r = 10
    i_max_r = 35

    train_data_size = 397985
    test_data_size = 49717
    val_data_size = 49717

    user_num = 55217 + 2
    item_num = 17408 + 2

    batch_size = 16
    print_step = 100

    num_heads = 2


class Office_Products_5_Config(DefaultConfig):

    def __init__(self):
        self.set_path('Office_Products_5')

    vocab_size = 50002
    word_dim = 100

    r_max_len = 63

    u_max_r = 9
    i_max_r = 31

    train_data_size = 640220
    test_data_size = 79962
    val_data_size = 79962

    user_num = 101498 + 2
    item_num = 27965 + 2

    batch_size = 16
    print_step = 100
    num_heads = 2


class Grocery_and_Gourmet_Food_5_Config(DefaultConfig):

    def __init__(self):
        self.set_path('Grocery_and_Gourmet_Food_5')

    vocab_size = 50002
    word_dim = 100

    r_max_len = 56

    u_max_r = 10
    i_max_r = 29

    train_data_size = 914903
    test_data_size = 114283
    val_data_size = 114284

    user_num = 127487 + 2
    item_num = 41320 + 2

    batch_size = 16
    print_step = 100
    num_heads = 2
