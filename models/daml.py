# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class DAML(nn.Module):
    '''
    KDD 2019 DAML
    '''

    def __init__(self, opt):
        super(DAML, self).__init__()

        self.opt = opt
        self.num_fea = 2  # ID + DOC
        self.user_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)  # vocab_size * 300
        self.item_word_embs = nn.Embedding(opt.vocab_size, opt.word_dim)  # vocab_size * 300

        # share
        self.word_cnn = nn.Conv2d(1, 1, (opt.kernel_size, opt.word_dim), padding=(1, 0))
        # document-level cnn
        self.user_doc_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim), padding=(1, 0))
        self.item_doc_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.word_dim), padding=(1, 0))
        # abstract-level cnn
        self.user_abs_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.filters_num))
        self.item_abs_cnn = nn.Conv2d(1, opt.filters_num, (opt.kernel_size, opt.filters_num))

        self.unfold = nn.Unfold((3, opt.filters_num), padding=(1, 0))

        # fc layerz
        self.user_fc = nn.Linear(opt.filters_num, opt.id_emb_size)
        self.item_fc = nn.Linear(opt.filters_num, opt.id_emb_size)

        self.uid_embedding = nn.Embedding(opt.user_num + 2, opt.id_emb_size)
        self.iid_embedding = nn.Embedding(opt.item_num + 2, opt.id_emb_size)

        self.reset_para()

    def forward(self, datas):
            '''
            前向传播函数，用于计算模型的输出。

            参数:
                datas: 包含用户评论、物品评论、用户ID、物品ID等数据的元组

            返回:
                use_fea: 用户特征向量和物品ID嵌入的组合特征
                item_fea: 物品特征向量和用户ID嵌入的组合特征
            '''
            _, _, uids, iids, _, _, user_doc, item_doc = datas

            # ------------------ review encoder ---------------------------------

            user_word_embs = self.user_word_embs(user_doc)
            item_word_embs = self.item_word_embs(item_doc)
            # (BS, filters_num, DOC_LEN, 1)
            user_local_fea = self.local_attention_cnn(user_word_embs, self.user_doc_cnn)
            item_local_fea = self.local_attention_cnn(item_word_embs, self.item_doc_cnn)

            # DOC_LEN * DOC_LEN
            euclidean = (user_local_fea - item_local_fea.permute(0, 1, 3, 2)).pow(2).sum(1).sqrt()
            attention_matrix = 1.0 / (1 + euclidean)
            # (?, DOC_LEN)
            user_attention = attention_matrix.sum(2)
            item_attention = attention_matrix.sum(1)

            # (?, 32)
            user_doc_fea = self.local_pooling_cnn(user_local_fea, user_attention, self.user_abs_cnn, self.user_fc)
            item_doc_fea = self.local_pooling_cnn(item_local_fea, item_attention, self.item_abs_cnn, self.item_fc)

            # ------------------ id embedding ---------------------------------
            uid_emb = self.uid_embedding(uids)
            iid_emb = self.iid_embedding(iids)

            use_fea = torch.stack([user_doc_fea, iid_emb], 1)
            item_fea = torch.stack([item_doc_fea, uid_emb], 1)

            return use_fea, item_fea

    def local_attention_cnn(self, word_embs, doc_cnn):
        '''
        :Eq1 - Eq7
        '''
        # 使用卷积神经网络处理单词嵌入，增加维度以适应卷积层的输入要求
        local_att_words = self.word_cnn(word_embs.unsqueeze(1))

        # 对卷积结果进行 Sigmoid 操作，获得单词级别的局部注意力权重
        local_word_weight = torch.sigmoid(local_att_words.squeeze(1))

        # 将单词嵌入乘以局部注意力权重
        word_embs = word_embs * local_word_weight

        # 使用卷积神经网络处理更新后的单词嵌入，增加维度以适应卷积层的输入要求
        d_fea = doc_cnn(word_embs.unsqueeze(1))

        # 返回卷积后的结果，即文档的特征表示
        return d_fea

    def local_pooling_cnn(self, feature, attention, cnn, fc):
        '''
        :Eq11 - Eq13
        feature: (?, 100, DOC_LEN ,1)
        attention: (?, DOC_LEN)
        '''
        # 获取特征的形状信息
        bs, n_filters, doc_len, _ = feature.shape

        # 调整特征的维度顺序以适应下一步操作
        feature = feature.permute(0, 3, 2, 1)  # bs * 1 * doc_len * embed

        # 调整注意力权重的形状
        attention = attention.reshape(bs, 1, doc_len, 1)  # bs * doc

        # 将特征乘以注意力权重
        pools = feature * attention

        # 使用 unfold 操作展开 pools
        pools = self.unfold(pools)

        # 调整 pools 的形状
        pools = pools.reshape(bs,self.opt.kernel_size, n_filters, doc_len)

        # 在第一个维度上求和，保持维度
        pools = pools.sum(dim=1, keepdims=True)  # bs * 1 * n_filters * doc_len

        # 调整 pools 的维度顺序
        pools = pools.transpose(2, 3)  # bs * 1 * doc_len * n_filters

        # 使用卷积神经网络处理 pools
        abs_fea = cnn(pools).squeeze(3)  # ? (DOC_LEN-2), n_filters

        # 使用平均池化操作
        abs_fea = F.avg_pool1d(abs_fea, abs_fea.size(2))  # ? n_filters

        # 使用 ReLU 激活函数
        abs_fea = F.relu(fc(abs_fea.squeeze(2)))  # ? 32

        return abs_fea

    def reset_para(self):

        cnns = [self.word_cnn, self.user_doc_cnn, self.item_doc_cnn, self.user_abs_cnn, self.item_abs_cnn]
        for cnn in cnns:
            nn.init.xavier_normal_(cnn.weight)  # 高斯分布初始化权重
            nn.init.uniform_(cnn.bias, -0.1, 0.1)  # 使用均匀分布初始化偏置，范围在[-0.1, 0.1]

        fcs = [self.user_fc, self.item_fc]
        for fc in fcs:
            nn.init.uniform_(fc.weight, -0.1, 0.1)  # 使用均匀分布初始化权重，范围在[-0.1, 0.1]
            nn.init.constant_(fc.bias, 0.1)  # 使用常数初始化偏置，值为0.1

        nn.init.uniform_(self.uid_embedding.weight, -0.1, 0.1)  # 初始化嵌入层的权重
        nn.init.uniform_(self.iid_embedding.weight, -0.1, 0.1)  # 使用均匀分布初始化权重，范围在[-0.1, 0.1]
        # 从预训练的 Word2Vec 模型中加载词向量，并将其复制到用户和物品的词嵌入层
        w2v = torch.from_numpy(np.load(self.opt.w2v_path))
        self.user_word_embs.weight.data.copy_(w2v.cuda())
        self.item_word_embs.weight.data.copy_(w2v.cuda())
