import json
import os
import re
import time
from operator import itemgetter

import gensim
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

P_REVIEW = 0.85
MAX_DF = 0.7
MAX_VOCAB = 50000
DOC_LEN = 500
PRE_W2V_BIN_PATH = "./dataset/glove.twitter.27B.100d.word2vec.txt"  # the pre-trained word2vec files

new_config = {
    'data_name': '',
    'vocab_size': 50002,
    'word_dim': 100,

    'r_max_len': 202,

    'u_max_r': 13,
    'i_max_r': 24,

    'train_data_size': 51764,
    'test_data_size': 6471,
    'val_data_size': 6471,

    'user_num': 5541 + 2,
    'item_num': 3568 + 2,

    'batch_size': 16,
    'print_step': 100,
}


def countNum(xDict):
    minNum = 100
    maxNum = 0
    sumNum = 0
    maxSent = 0
    minSent = 3000
    # pSentLen = 0
    ReviewLenList = []
    SentLenList = []
    for (i, text) in xDict.items():
        sumNum = sumNum + len(text)
        if len(text) < minNum:
            minNum = len(text)
        if len(text) > maxNum:
            maxNum = len(text)
        ReviewLenList.append(len(text))
        for sent in text:
            # SentLenList.append(len(sent))
            if sent != "":
                wordTokens = sent.split()
            if len(wordTokens) > maxSent:
                maxSent = len(wordTokens)
            if len(wordTokens) < minSent:
                minSent = len(wordTokens)
            SentLenList.append(len(wordTokens))
    averageNum = sumNum // (len(xDict))

    x = np.sort(SentLenList)
    xLen = len(x)
    pSentLen = x[int(P_REVIEW * xLen) - 1]
    x = np.sort(ReviewLenList)
    xLen = len(x)
    pReviewLen = x[int(P_REVIEW * xLen) - 1]

    return minNum, maxNum, averageNum, maxSent, minSent, pReviewLen, pSentLen


def build_doc(u_reviews_dict, i_reviews_dict):
    '''
    Build document representations for user and item reviews.

    Args:
        u_reviews_dict (dict): A dictionary containing user reviews.
        i_reviews_dict (dict): A dictionary containing item reviews.

    Returns:
        tuple: A tuple containing the following:
            - vocab (dict): The vocabulary dictionary.
            - u_doc (list): The cleaned document representation of user reviews.
            - i_doc (list): The cleaned document representation of item reviews.
            - u_reviews_dict (dict): The cleaned user reviews dictionary.
            - i_reviews_dict (dict): The cleaned item reviews dictionary.
    '''
    u_reviews = []
    for ind in range(len(u_reviews_dict)):
        u_reviews.append(' <SEP> '.join(u_reviews_dict[ind]))

    i_reviews = []
    for ind in range(len(i_reviews_dict)):
        i_reviews.append('<SEP>'.join(i_reviews_dict[ind]))

    vectorizer = TfidfVectorizer(max_df=MAX_DF, max_features=MAX_VOCAB)
    vectorizer.fit(u_reviews)
    vocab = vectorizer.vocabulary_
    vocab[MAX_VOCAB] = '<SEP>'

    def clean_review(rDict):
        '''
        Clean the reviews in the given dictionary.

        Args:
            rDict (dict): A dictionary containing reviews.

        Returns:
            dict: A dictionary with cleaned reviews.
        '''
        new_dict = {}
        for k, text in rDict.items():
            new_reviews = []
            for r in text:
                words = ' '.join([w for w in r.split() if w in vocab])
                new_reviews.append(words)
            new_dict[k] = new_reviews
        return new_dict

    def clean_doc(raw):
        '''
        Clean the document representation.

        Args:
            raw (list): A list of document representations.

        Returns:
            list: A list of cleaned document representations.
        '''
        new_raw = []
        for line in raw:
            review = [word for word in line.split() if word in vocab]
            if len(review) > DOC_LEN:
                review = review[:DOC_LEN]
            new_raw.append(review)
        return new_raw

    u_reviews_dict = clean_review(u_reviews_dict)
    i_reviews_dict = clean_review(i_reviews_dict)

    u_doc = clean_doc(u_reviews)
    i_doc = clean_doc(i_reviews)

    return vocab, u_doc, i_doc, u_reviews_dict, i_reviews_dict


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def clean_str(string):
    # 清理字符串中的特殊字符
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"sssss ", " ", string)
    return string.strip().lower()


def pre_process(filename):
    start_time = time.time()
    save_folder = "./dataset/" + filename.split('/')[-1].split('.')[0]
    new_config['data_name'] = filename.split('/')[-1].split('.')[0]
    print("数据集名称：", filename)
    if not os.path.exists(save_folder + '/train'):
        os.makedirs(save_folder + '/train')
    if not os.path.exists(save_folder + '/val'):
        os.makedirs(save_folder + '/val')
    if not os.path.exists(save_folder + '/test'):
        os.makedirs(save_folder + '/test')
    if PRE_W2V_BIN_PATH is None:
        print("无词向量")
    file = open(filename, errors='ignore')
    users_id = []
    items_id = []
    ratings = []
    reviews = []
    for line in file:
        js = json.loads(line)
        if str(js['reviewerID']) == 'unknown' or str(js['asin']) == 'unknown':
            print("unknown user id or unknown item id")
            continue
        try:
            reviews.append(js['reviewText'])
            users_id.append(str(js['reviewerID']))
            items_id.append(str(js['asin']))
            ratings.append(str(js['overall']))
        except:
            continue
    temp_data = {'user_id': pd.Series(users_id), 'item_id': pd.Series(items_id), 'rating': pd.Series(ratings),
                 'review': pd.Series(reviews)}
    data = pd.DataFrame(temp_data)
    del temp_data
    del users_id
    del items_id
    del ratings
    del reviews
    uid_List = data['user_id'].unique().tolist()
    iid_List = data['item_id'].unique().tolist()
    user_num = len(uid_List)
    item_num = len(iid_List)
    print("用户数量：", user_num)
    print("物品数量：", item_num)
    print("数据集大小：", len(data))
    print("数据集密度:", data['rating'].count() / (user_num * item_num))
    print('------')
    user_to_id = dict(zip(uid_List, range(user_num)))
    item_to_id = dict(zip(iid_List, range(item_num)))
    data['user_id'] = data['user_id'].map(lambda x: user_to_id[x])
    data['item_id'] = data['item_id'].map(lambda x: item_to_id[x])
    print(data.head())
    print("数据集转换完成")
    print('------')
    print("开始划分数据集", now())
    data_train, data_test = train_test_split(data, test_size=0.2, random_state=1234)
    uid_train = data_train['user_id'].unique().tolist()
    uid_test = data_test['user_id'].unique().tolist()
    iid_train = data_train['item_id'].unique().tolist()
    iid_test = data_test['item_id'].unique().tolist()
    user_num_train = len(uid_train)
    item_num_train = len(iid_train)
    print("训练集用户数量：", user_num_train)
    print("训练集物品数量：", item_num_train)
    print("训练集数据集大小：", len(data_train))
    print("------")
    print("查看有无缺失用户或物品")
    uid_miss = []
    iid_miss = []
    if user_num_train != user_num or item_num_train != item_num:
        for i in range(user_num):
            if i not in uid_train:
                uid_miss.append(i)
        print("缺失用户：", uid_miss)
        for i in range(item_num):
            if i not in iid_train:
                iid_miss.append(i)
        print("缺失物品：", iid_miss)
    print("------")
    print("训练集中添加缺失ID")
    uid_miss_index = []
    iid_miss_index = []
    for uid in uid_miss:
        index = data_test[data_test['user_id'] == uid].index.tolist()
        uid_miss_index.extend(index)
    data_train = pd.concat([data_train, data_test.loc[uid_miss_index]])  # 将缺失的用户添加到训练集中

    for iid in iid_miss:
        index = data_test[data_test['item_id'] == iid].index.tolist()
        iid_miss_index.extend(index)
    data_train = pd.concat([data_train, data_test.loc[iid_miss_index]])  # 将缺失的物品添加到训练集中
    all_miss_index = list(set(uid_miss_index + iid_miss_index))
    data_test = data_test.drop(all_miss_index)  # 将缺失的用户和物品从测试集中删除
    print("------")
    print("划分测试集和验证集")
    data_val, data_test = train_test_split(data_test, test_size=0.5, random_state=1234)
    uid_list_train = data_train['user_id'].unique().tolist()
    iid_list_train = data_train['item_id'].unique().tolist()
    user_num_train = len(uid_list_train)
    item_num_train = len(iid_list_train)
    print("------")
    print("训练集用户数量：", user_num_train)
    print("训练集物品数量：", item_num_train)
    print("训练集数据集大小：", data_train.shape[0])
    print("------")
    print("保存数据集")
    new_config['train_data_size'] = data_train.shape[0]

    def extract(data_dict):
        x = []
        y = []
        for i in data_dict.values:
            uid = i[0]
            iid = i[1]
            x.append([uid, iid])
            y.append(float(i[2]))
        return x, y

    x_train, y_train = extract(data_train)
    x_val, y_val = extract(data_val)
    x_test, y_test = extract(data_test)
    np.save(save_folder + '/train/Train.npy', x_train)
    np.save(save_folder + '/train/Train_rating.npy', y_train)
    np.save(save_folder + '/val/Val.npy', x_val)
    np.save(save_folder + '/val/Val_rating.npy', y_val)
    np.save(save_folder + '/test/Test.npy', x_test)
    np.save(save_folder + '/test/Test_rating.npy', y_test)
    print("------")
    print("提取评论数据")
    user_reviews_dict = {}
    item_reviews_dict = {}
    user_iid_dict = {}
    item_uid_dict = {}
    for i in data_train.values:
        single_review = clean_str(i[3].encode('ascii', 'ignore').decode('ascii'))
        if len(single_review) == 0:
            single_review = "<unk>"
        if i[0] in user_reviews_dict:
            user_reviews_dict[i[0]].append(single_review)
            user_iid_dict[i[0]].append(i[1])
        else:
            user_reviews_dict[i[0]] = [single_review]
            user_iid_dict[i[0]] = [i[1]]

        if i[1] in item_reviews_dict:
            item_reviews_dict[i[1]].append(single_review)
            item_uid_dict[i[1]].append(i[0])
        else:
            item_reviews_dict[i[1]] = [single_review]
            item_uid_dict[i[1]] = [i[0]]
    vocab, user_review2doc, item_review2doc, user_reviews_dict, item_reviews_dict = build_doc(user_reviews_dict,
                                                                                              item_reviews_dict)
    print("------")
    word_index = {}
    word_index['<unk>'] = 0
    for i, w in enumerate(vocab.keys(), 1):
        word_index[w] = i
    print(f"The vocab size: {len(word_index)}")
    print(f"Average user document length: {sum([len(i) for i in user_review2doc]) / len(user_review2doc)}")
    print(f"Average item document length: {sum([len(i) for i in item_review2doc]) / len(item_review2doc)}")

    print(now())
    u_minNum, u_maxNum, u_averageNum, u_maxSent, u_minSent, u_pReviewLen, u_pSentLen = countNum(user_reviews_dict)
    print("用户最少有{}个评论,最多有{}个评论，平均有{}个评论, " \
          "句子最大长度{},句子的最短长度{}，" \
          "设定用户评论个数为{}： 设定句子最大长度为{}".format(u_minNum, u_maxNum, u_averageNum, u_maxSent, u_minSent,
                                                              u_pReviewLen, u_pSentLen))
    new_config['u_max_r'] = u_pSentLen

    # 计算评论的数量统计信息
    i_minNum, i_maxNum, i_averageNum, i_maxSent, i_minSent, i_pReviewLen, i_pSentLen = countNum(item_reviews_dict)
    print("商品最少有{}个评论,最多有{}个评论，平均有{}个评论," \
          "句子最大长度{},句子的最短长度{}," \
          ",设定商品评论数目{}, 设定句子最大长度为{}".format(i_minNum, i_maxNum, i_averageNum, u_maxSent, i_minSent,
                                                             i_pReviewLen, i_pSentLen))
    print("最终设定句子最大长度为(取最大值)：{}".format(max(u_pSentLen, i_pSentLen)))
    new_config['i_max_r'] = i_pSentLen

    # 设置最大句子长度和最小句子长度
    maxSentLen = max(u_pSentLen, i_pSentLen)
    minSentlen = 1
    new_config['r_max_len']=maxSentLen

    # 初始化用户和商品的文本和id列表
    userReview2Index = []
    userDoc2Index = []
    user_iid_list = []
    print(f"-" * 60)
    print(f"{now()} Step4: padding all the text and id lists and save into npy.")

    # 定义填充文本的函数
    def padding_text(textList, num):
        new_textList = []
        if len(textList) >= num:
            new_textList = textList[:num]
        else:
            padding = [[0] * len(textList[0]) for _ in range(num - len(textList))]
            new_textList = textList + padding
        return new_textList

    # 定义填充id的函数
    def padding_ids(iids, num, pad_id):
        if len(iids) >= num:
            new_iids = iids[:num]
        else:
            new_iids = iids + [pad_id] * (num - len(iids))
        return new_iids

    # 定义填充文档的函数
    def padding_doc(doc):
        pDocLen = DOC_LEN
        new_doc = []
        for d in doc:
            if len(d) < pDocLen:
                d = d + [0] * (pDocLen - len(d))
            else:
                d = d[:pDocLen]
            new_doc.append(d)
        return new_doc, pDocLen

    # 处理用户数据
    for i in range(user_num):
        count_user = 0
        dataList = []
        a_count = 0

        textList = user_reviews_dict[i]
        u_iids = user_iid_dict[i]
        u_reviewList = []

        user_iid_list.append(padding_ids(u_iids, u_pReviewLen, item_num + 1))
        doc2index = [word_index[w] for w in user_review2doc[i]]

        for text in textList:
            text2index = []
            wordTokens = text.strip().split()
            if len(wordTokens) == 0:
                wordTokens = ['<unk>']
            text2index = [word_index[w] for w in wordTokens]
            if len(text2index) < maxSentLen:
                text2index = text2index + [0] * (maxSentLen - len(text2index))
            else:
                text2index = text2index[:maxSentLen]
            u_reviewList.append(text2index)

        userReview2Index.append(padding_text(u_reviewList, u_pReviewLen))
        userDoc2Index.append(doc2index)

    # 填充用户文档
    userDoc2Index, userDocLen = padding_doc(userDoc2Index)
    print(f"user document length: {userDocLen}")

    # 初始化商品的文本和id列表
    itemReview2Index = []
    itemDoc2Index = []
    item_uid_list = []

    # 处理商品数据
    for i in range(item_num):
        count_item = 0
        dataList = []
        textList = item_reviews_dict[i]
        i_uids = item_uid_dict[i]
        i_reviewList = []
        i_reviewLen = []
        item_uid_list.append(padding_ids(i_uids, i_pReviewLen, user_num + 1))
        doc2index = [word_index[w] for w in item_review2doc[i]]

        for text in textList:
            text2index = []
            wordTokens = text.strip().split()
            if len(wordTokens) == 0:
                wordTokens = ['<unk>']
            text2index = [word_index[w] for w in wordTokens]
            if len(text2index) < maxSentLen:
                text2index = text2index + [0] * (maxSentLen - len(text2index))
            else:
                text2index = text2index[:maxSentLen]
            if len(text2index) < maxSentLen:
                text2index = text2index + [0] * (maxSentLen - len(text2index))
            i_reviewList.append(text2index)
        itemReview2Index.append(padding_text(i_reviewList, i_pReviewLen))
        itemDoc2Index.append(doc2index)

    # 填充商品文档
    itemDoc2Index, itemDocLen = padding_doc(itemDoc2Index)
    print(f"item document length: {itemDocLen}")

    print("-" * 60)
    print(f"{now()} start writing npy...")

    # 保存数据为npy文件
    np.save(f"{save_folder}/train/userReview2Index.npy", userReview2Index)
    np.save(f"{save_folder}/train/user_item2id.npy", user_iid_list)
    np.save(f"{save_folder}/train/userDoc2Index.npy", userDoc2Index)

    np.save(f"{save_folder}/train/itemReview2Index.npy", itemReview2Index)
    np.save(f"{save_folder}/train/item_user2id.npy", item_uid_list)
    np.save(f"{save_folder}/train/itemDoc2Index.npy", itemDoc2Index)

    print(f"{now()} write finised")

    # 生成词向量
    print("-" * 60)
    print(f"{now()} Step5: start word embedding mapping...")
    vocab_item = sorted(word_index.items(), key=itemgetter(1))
    w2v = []
    out = 0
    if PRE_W2V_BIN_PATH:
        pre_word2v = gensim.models.KeyedVectors.load_word2vec_format(PRE_W2V_BIN_PATH)
    else:
        pre_word2v = {}
    print(f"{now()} 开始提取embedding")
    for word, key in vocab_item:
        if word in pre_word2v:
            w2v.append(pre_word2v[word])
        else:
            out += 1
            w2v.append(np.random.uniform(-1.0, 1.0, (100,)))
    print("############################")
    print(f"out of vocab: {out}")
    print(f"w2v size: {len(w2v)}")
    print("############################")
    w2vArray = np.array(w2v)
    print(w2vArray.shape)
    np.save(f"{save_folder}/train/w2v.npy", w2v)
    end_time = time.time()
    print(f"{now()} all steps finised, cost time: {end_time - start_time:.4f}s")
    pass


if __name__ == '__main__':
    filename = "/dataset/Musical_Instruments_5.json"
    pre_process(filename)
