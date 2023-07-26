# load Amazon Book datset.

# @Time   : 7/10/2022
# @Author : Zhuangzhuang He

import sys
from utils import id2num, id2dict
from collections import defaultdict
import math
import json
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="../Amazondataset/", type=str, help="Amazondataset or Yelpdataset")
parser.add_argument("--output_dir", default="output/", type=str)
parser.add_argument("--data_name", default="YelpChi", type=str, help="YelpChi, YelpZip, YelpNYC")
parser.add_argument("--specific_name", default="Res", type=str, help="Res, Hotel in YelpChi")
parser.add_argument("--meta_path", default='/output_meta_yelpResData_NRYRcleaned.txt', type=str, help="define tha path of meta, or /metadata")
parser.add_argument("--review_path", default='/output_review_yelpResData_NRYRcleaned.txt', type=str, help="define tha path of review, or reviewContent")
parser.add_argument("--do_eval", action="store_true")
parser.add_argument("--load_model", default=None, type=str)
parser.add_argument("--note", default='GCNM_Res_LN_0114_test', type=str)
parser.add_argument("--mode", default='test', type=str, help="prevents duplicate creation of multiple output folders")

# model args
parser.add_argument("--model_name", default="GCNM", type=str)
parser.add_argument("--hidden_size", default=64, type=int, help="hidden size of model")
parser.add_argument("--num_hidden_layers", default=2, type=int, help="number of filter-enhanced blocks")
parser.add_argument("--num_attention_heads", default=2, type=int)
parser.add_argument("--num_convolution_layers", default=1, type=int)
parser.add_argument("--hidden_act", default="gelu", type=str) # gelu relu
parser.add_argument("--attention_probs_dropout_prob", default=0.5, type=float)
parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)
parser.add_argument("--initializer_range", default=0.02, type=float)
parser.add_argument("--max_seq_length", default=50, type=int)
parser.add_argument("--no_filters", action="store_true", help="if no filters, filter layers transform to self-attention")

# train args
parser.add_argument("--lr", default=0.001, type=float, help="learning rate of adam")
parser.add_argument("--batch_size", default=64, type=int, help="number of batch_size")
parser.add_argument("--epochs", default=200, type=int, help="number of epochs")
parser.add_argument("--criterion", default='cross_entropy', type=str, help="criterion")
parser.add_argument("--no_cuda", action="store_true")
parser.add_argument("--log_freq", default=1, type=int, help="per epoch print res")
parser.add_argument("--full_sort", action="store_true")
parser.add_argument("--patience", default=10, type=int, help="how long to wait after last time validation loss improved")
parser.add_argument('--forget_rate', type=float, help='forget rate', default=0.1)
parser.add_argument('--num_gradual', type=int, default=2,
                    help='how many epochs for linear drop rate, can be 5, 10, 15. This parameter is equal to '
                        'Tk for R(T) in Co-teaching paper.')
parser.add_argument('--exponent', type=float, default=1,
                    help='exponent of the forget rate, can be 0.5, 1, 2. This parameter is equal to c in '
                        'Tc for R(T) in Co-teaching paper.')

parser.add_argument("--seed", default=42, type=int)
parser.add_argument("--weight_decay", default=0.0, type=float, help="weight_decay of adam")
parser.add_argument("--adam_beta1", default=0.9, type=float, help="adam first beta value")
parser.add_argument("--adam_beta2", default=0.999, type=float, help="adam second beta value")
parser.add_argument("--gpu_id", default="0", type=str, help="gpu_id")
parser.add_argument("--variance", default=5, type=float)

args = parser.parse_args()


def get_total_data(args):
    # sys.stdout = open(args.output_dir  + '/' + args.note + '/' + args.args_str + '.log', 'w')
    args = vars(args)
    """
    Import all the data and do the processing on them.
    """
    if args['specific_name'] == 'Res' or 'Hotel':
        print("'---------------START PROCESSING %s_%s---------------'"% (args['data_name'], args['specific_name']))
    else:
        print('---------------START PROCESSING %s---------------'% (args['data_name']))
        
    users_id = []
    items_id = []
    ratings = []
    reviews = []
    labels = []
    uid_iid_fea = defaultdict(list)  # user,item,feature 三元组
    uid_iid_label = defaultdict(int)  # user,item,label 三元组
    uid_iid_rating = defaultdict(int)
    num_review = defaultdict(int)  # 用户评论个数,有评论就加1
    len_name = defaultdict(int)  # 用户名字长度
    score_max_min = defaultdict(lambda: [0, 5])
    num_each_score = defaultdict(lambda: [0, 0, 0, 0, 0])  # 对每个用户的评分进行初始化[0,0,0,0,0]
    ratio_each_score = defaultdict(lambda: [0, 0, 0, 0, 0])  # 完成上一步才可以进行这一步，某个评分个数与全部求和的比值
    entropy_rating = defaultdict(float)  # 用户的评分熵
    num_pos_neg = defaultdict(lambda: [0, 0])
    ratio_pos_neg = defaultdict(lambda: [0, 0])  # 用户的正负评分比例
    num_total_vote = defaultdict(int)  # 全部的投票数
    num_help_unhelp = defaultdict(lambda: [0, 0])  # help和unhelpful分别的投票数
    avg_vote = defaultdict(lambda: [0, 0])  # 正负平均投票数
    ratio_help_unhelp = defaultdict(lambda: [0, 0])  # 正负投票比例
    fea = defaultdict(list)
    users_id_amazon = defaultdict(int)
    items_id_amazon = defaultdict(int)

    if args['data_name'] == 'YelpChi':
        """
        Yelp dataset:
        1. Date
        2. review ID
        3. reviewer ID
        4. product ID
        5. Label (N means genuine review and Y means fake reviews)
        6. Useful (count information)
        7. Funny (count information)
        8. Cool (count information)
        9. star ratings
        text feature: 6,7,8,9,len(review)
        user feature: since reviewerID in yelpchi is masked, the len(reviewerID) does not needed static.
                    num_useful,num_Funny,num_Cool,num_reviews, and somematter same with amazon
        """
        # 统计用户行为信息特征
        user_num_review = defaultdict(int)
        user_score_max_min = defaultdict(lambda: [0, 5])
        user_num_each_score = defaultdict(lambda: [0, 0, 0, 0, 0])  # 对每个用户的评分进行初始化[0,0,0,0,0]
        user_num_pos_neg = defaultdict(lambda: [0, 0])
        user_num_useful = defaultdict(int)
        user_num_funny = defaultdict(int)
        user_num_cool = defaultdict(int)
        user_ratio_each_score = defaultdict(lambda: [0, 0, 0, 0, 0])  # 完成上一步才可以进行这一步，某个评分个数与全部求和的比值
        user_entropy_rating = defaultdict(float)  # 用户的评分熵
        user_ratio_pos_neg = defaultdict(lambda: [0, 0])  # 用户的正负评分比例
        user_fea = defaultdict(list)
        # 镜像的统计物品的行为信息特征
        item_num_review = defaultdict(int)
        item_score_max_min = defaultdict(lambda: [0, 5])
        item_num_each_score = defaultdict(lambda: [0, 0, 0, 0, 0])  # 对每个用户的评分进行初始化[0,0,0,0,0]
        item_num_pos_neg = defaultdict(lambda: [0, 0])
        item_num_useful = defaultdict(int)
        item_num_funny = defaultdict(int)
        item_num_cool = defaultdict(int)
        item_ratio_each_score = defaultdict(lambda: [0, 0, 0, 0, 0])  # 完成上一步才可以进行这一步，某个评分个数与全部求和的比值
        item_entropy_rating = defaultdict(float)  # 用户的评分熵
        item_ratio_pos_neg = defaultdict(lambda: [0, 0])  # 用户的正负评分比例
        item_fea = defaultdict(list)
        # 统计文本元信息
        users_id = []
        items_id = []
        reviews_id = []
        len_review = []
        reviews = []
        labels = {}
        ratings = {}
        # 统计文本元信息与自身文本语义
        meta_path = args['data_dir'] + args['data_name'] + args['meta_path']
        review_path = args['data_dir'] + args['data_name'] + args['review_path']
        review = open(review_path)
        meta = open(meta_path)
        # 读取用户的各项行为信息以及uid iid rid
        for index,line in enumerate(meta):
            value = line.split()
            # 作为整体，用于后续编号
            reviews_id.append(value[1])
            users_id.append(value[2])
            items_id.append(value[3])
            if value[4] == 'N':
                # true review
                value[4] = 0
            else:
                value[4] = 1
            labels[index] = value[4]
            ratings[index] = int(value[8])
            # how reviews per reviewer
            # users feature
            user_num_review[value[2]] = user_num_review[value[2]] + 1
            user_num_useful[value[2]] = user_num_useful[value[2]] + int(value[5])
            user_num_funny[value[2]] = user_num_useful[value[2]] + int(value[6])
            user_num_cool[value[2]] = user_num_useful[value[2]] + int(value[7])
            # max and min score per reviewer
            if int(value[8]) > user_score_max_min[value[2]][0]:
                user_score_max_min[value[2]][0] = int(value[8])
            if int(value[8]) < user_score_max_min[value[2]][1]:
                user_score_max_min[value[2]][1] = int(value[8])
            user_num_each_score[value[2]][int(value[8]) - 1] = user_num_each_score[value[2]][int(value[8]) - 1] + 1  # 根据评分，对每个部分自加
            if int(value[8]) == 4 or int(value[8]) == 5:  # 分数为4或5为高分
                user_num_pos_neg[value[2]][0] = user_num_pos_neg[value[2]][0] + 1
            if int(value[8]) == 1 or int(value[8]) == 2:  # 分数为1或2为低分
                user_num_pos_neg[value[2]][1] = user_num_pos_neg[value[2]][1] + 1
            # items feature
            item_num_review[value[3]] = item_num_review[value[3]] + 1
            item_num_useful[value[3]] = item_num_useful[value[3]] + int(value[5])
            item_num_funny[value[3]] = item_num_useful[value[3]] + int(value[6])
            item_num_cool[value[3]] = item_num_useful[value[3]] + int(value[7])
            # max and min score per items
            if int(value[8]) > item_score_max_min[value[3]][0]:
                item_score_max_min[value[3]][0] = int(value[8])
            if int(value[8]) < item_score_max_min[value[3]][1]:
                item_score_max_min[value[3]][1] = int(value[8])
            item_num_each_score[value[3]][int(value[8]) - 1] = item_num_each_score[value[3]][int(value[8]) - 1] + 1  # 根据评分，对每个部分自加
            if int(value[8]) == 4 or int(value[8]) == 5:  # 分数为4或5为高分
                item_num_pos_neg[value[3]][0] = item_num_pos_neg[value[3]][0] + 1
            if int(value[8]) == 1 or int(value[8]) == 3:  # 分数为1或3为低分
                item_num_pos_neg[value[3]][1] = item_num_pos_neg[value[3]][1] + 1
        users_id = id2dict(users_id)
        items_id = id2dict(items_id)
        reviews_id = id2dict(reviews_id)
        reviews = {}
        len_review = {}
        for index, line in enumerate(review):
            reviews[index] = line
            len_review[index] = len(line)
        review.close()

        for key in user_num_each_score.keys():
            for i in range(5):
                user_ratio_each_score[key][i] = round(user_num_each_score[key][i] / sum(user_num_each_score[key]), 3)
                if user_ratio_each_score[key][i] != 0:
                    user_entropy_rating[key] = round(
                        user_entropy_rating[key] - user_ratio_each_score[key][i] * math.log(user_ratio_each_score[key][i]), 3)
                if i < 2:
                    if sum(user_num_pos_neg[key]) == 0:
                        continue
                    else:
                        user_ratio_pos_neg[key][i] = user_num_pos_neg[key][i] / sum(user_num_pos_neg[key])
        for key in item_num_each_score.keys():
            for i in range(5):
                item_ratio_each_score[key][i] = round(item_num_each_score[key][i] / sum(item_num_each_score[key]), 3)
                if item_ratio_each_score[key][i] != 0:
                    item_entropy_rating[key] = round(
                        item_entropy_rating[key] - item_ratio_each_score[key][i] * math.log(item_ratio_each_score[key][i]), 3)
                if i < 2:
                    if sum(item_num_pos_neg[key]) == 0:
                        continue
                    else:
                        item_ratio_pos_neg[key][i] = item_num_pos_neg[key][i] / sum(item_num_pos_neg[key])
        for key in user_num_each_score.keys():
            user_fea[key].append(user_num_review[key])
            user_fea[key].append(user_num_useful[key])
            user_fea[key].append(user_num_funny[key])
            user_fea[key].append(user_num_cool[key])
            user_fea[key].extend(user_score_max_min[key])
            user_fea[key].extend(user_num_each_score[key])
            user_fea[key].extend(user_ratio_each_score[key])
            user_fea[key].append(user_entropy_rating[key])
            user_fea[key].extend(user_num_pos_neg[key])
            user_fea[key].extend(user_ratio_pos_neg[key])
        for key in item_num_each_score.keys():
            item_fea[key].append(item_num_review[key])
            item_fea[key].append(item_num_useful[key])
            item_fea[key].append(item_num_funny[key])
            item_fea[key].append(item_num_cool[key])
            item_fea[key].extend(item_score_max_min[key])
            item_fea[key].extend(item_num_each_score[key])
            item_fea[key].extend(item_ratio_each_score[key])
            item_fea[key].append(item_entropy_rating[key])
            item_fea[key].extend(item_num_pos_neg[key])
            item_fea[key].extend(item_ratio_pos_neg[key])
        meta.close()

        uid_userfea = defaultdict(list)
        iid_itemfea = defaultdict(list)
        rid_metafea = defaultdict(list)
        uid_rid = []
        rid_iid = []
        rid_text = defaultdict(str)
        rid_userfea = defaultdict(list)
        rid_itemfea = defaultdict(list)
        rid_label = defaultdict(int)
        uidiid_rid = defaultdict(int)
        uidiid_label = defaultdict(int)
        train_uidiid_rid = defaultdict(int)
        train_uidiid_label = defaultdict(int)
        test_uidiid_rid = defaultdict(int)
        test_uidiid_label = defaultdict(int)
        
        meta = open(meta_path)
        stat_train_fake = 0 
        stat_train_true = 0 
        stat_test_fake = 0
        stat_test_true = 0
        for index, line in enumerate(meta):
            value = line.split()
            uid_userfea[users_id[value[2]]] = user_fea[value[2]]
            iid_itemfea[items_id[value[3]]] = item_fea[value[3]]
            rid_text[reviews_id[value[1]]] = reviews[index]
            rid_label[reviews_id[value[1]]] = labels[index]
            rid_userfea[reviews_id[value[1]]] = user_fea[value[2]]
            rid_itemfea[reviews_id[value[1]]] = item_fea[value[3]]
            rid_metafea[reviews_id[value[1]]] = [ratings[index], int(value[5]), int(value[6]), int(value[7]), len_review[index]]  #一些通用特征
            if int(value[0][-2:]) < 12:
                # 根据IfSpard January 1, 2012 以前作为训练集，之后作为测试集
                train_uidiid_rid[(users_id[value[2]], items_id[value[3]])] = reviews_id[value[1]]
                train_uidiid_label[(users_id[value[2]], items_id[value[3]])] = labels[index]
                if labels[index] == 0:
                    stat_train_true += 1
                else:
                    stat_train_fake += 1
            else:
                test_uidiid_rid[(users_id[value[2]], items_id[value[3]])] = reviews_id[value[1]]
                test_uidiid_label[(users_id[value[2]], items_id[value[3]])] = labels[index]
                if labels[index] == 0:
                    stat_test_true += 1
                else:
                    stat_test_fake += 1
            uid_rid.append([users_id[value[2]],   reviews_id[value[1]]])
            rid_iid.append([reviews_id[value[1]], items_id[value[3]]])
            uidiid_rid[(users_id[value[2]], items_id[value[3]])] = reviews_id[value[1]]
            uidiid_label[(users_id[value[2]], items_id[value[3]])] = labels[index]
        # 倒置字典
        # rid_uidiid = {v:k for k,v in uidiid_rid.items()}
        print("'---------------Train and Test Information---------------' \
                      \ntrain datset statistic: traindataset number %s, true %s, fake %s, ratio %s \
                      \ntest  datset statistic: testdataset  number %s, true %s, fake %s, ratio %s \
                      \nthe ratio of train to test: %s"% 
                      (len(train_uidiid_rid), stat_train_true, stat_train_fake, round(stat_train_true/(stat_train_true+stat_train_fake),3),
                      len(test_uidiid_rid), stat_test_true, stat_test_fake, round(stat_test_true/(stat_test_true+stat_test_fake),3),
                      round(len(train_uidiid_rid)/(len(train_uidiid_rid)+len(test_uidiid_rid)),3)))
        return train_uidiid_rid, train_uidiid_label, test_uidiid_rid, test_uidiid_label, uidiid_rid, uidiid_label, uid_rid, rid_iid,\
               uid_userfea,iid_itemfea, \
               rid_metafea, rid_text, rid_userfea, rid_itemfea, rid_label
    elif args['data_name'] == 'YelpNYC' or args['data_name'] == 'YelpZip':
        # 统计用户行为信息特征
        user_num_review = defaultdict(int)
        user_score_max_min = defaultdict(lambda: [0, 5])
        user_num_each_score = defaultdict(lambda: [0, 0, 0, 0, 0])  # 对每个用户的评分进行初始化[0,0,0,0,0]
        user_num_pos_neg = defaultdict(lambda: [0, 0])
        user_ratio_each_score = defaultdict(lambda: [0, 0, 0, 0, 0])  # 完成上一步才可以进行这一步，某个评分个数与全部求和的比值
        user_entropy_rating = defaultdict(float)  # 用户的评分熵
        user_ratio_pos_neg = defaultdict(lambda: [0, 0])  # 用户的正负评分比例
        user_fea = defaultdict(list)
        # 镜像的统计物品的行为信息特征
        item_num_review = defaultdict(int)
        item_score_max_min = defaultdict(lambda: [0, 5])
        item_num_each_score = defaultdict(lambda: [0, 0, 0, 0, 0])  # 对每个用户的评分进行初始化[0,0,0,0,0]
        item_num_pos_neg = defaultdict(lambda: [0, 0])
        item_ratio_each_score = defaultdict(lambda: [0, 0, 0, 0, 0])  # 完成上一步才可以进行这一步，某个评分个数与全部求和的比值
        item_entropy_rating = defaultdict(float)  # 用户的评分熵
        item_ratio_pos_neg = defaultdict(lambda: [0, 0])  # 用户的正负评分比例
        item_fea = defaultdict(list)
        # 统计文本元信息
        users_id = []
        items_id = []
        reviews_id = []
        len_review = []
        reviews = []
        labels = {}
        ratings = {}
        
        meta_path = args['data_dir'] + args['data_name'] + args['meta_path']
        review_path = args['data_dir'] + args['data_name'] + args['review_path']
        review = open(review_path)
        meta = open(meta_path)
        for index, line in enumerate(meta):
            value = line.split('\t')
            uid = int(value[0])
            iid = int(value[1])
            rating = round(float(value[2]), 1)
            
            # reviews_id.append(index)
            users_id.append(uid)
            items_id.append(iid)
            if value[3] == "1":
                # 1为真实，不是为0
                value[3] = 0
            else:
                value[3] = 1
            labels[index] = value[3]
            ratings[index] = rating
            # how reviews per reviewer
            # users feature
            user_num_review[uid] = user_num_review[uid] + 1
            # max and min score per reviewer
            if rating > user_score_max_min[uid][0]:
                user_score_max_min[uid][0] = rating
            if rating < user_score_max_min[uid][1]:
                user_score_max_min[uid][1] = rating
            user_num_each_score[uid][int(rating) - 1] = user_num_each_score[uid][int(rating) - 1] + 1  # 根据评分，对每个部分自加
            if int(rating) == 4 or int(rating) == 5:  # 分数为4或5为高分
                user_num_pos_neg[uid][0] = user_num_pos_neg[uid][0] + 1
            if int(rating) == 1 or int(rating) == 2:  # 分数为1或2为低分
                user_num_pos_neg[uid][1] = user_num_pos_neg[uid][1] + 1
            # items feature
            item_num_review[iid] = item_num_review[iid] + 1
            # max and min score per items
            if int(rating) > item_score_max_min[iid][0]:
                item_score_max_min[iid][0] = int(rating)
            if int(rating) < item_score_max_min[iid][1]:
                item_score_max_min[iid][1] = int(rating)
            item_num_each_score[iid][int(rating) - 1] = item_num_each_score[iid][int(rating) - 1] + 1  # 根据评分，对每个部分自加
            if int(rating) == 4 or int(rating) == 5:  # 分数为4或5为高分
                item_num_pos_neg[iid][0] = item_num_pos_neg[iid][0] + 1
            if int(rating) == 1 or int(rating) == 3:  # 分数为1或3为低分
                item_num_pos_neg[iid][1] = item_num_pos_neg[iid][1] + 1
        users_id = id2dict(users_id)
        items_id = id2dict(items_id)
        
        for key in user_num_each_score.keys():
            for i in range(5):
                user_ratio_each_score[key][i] = round(user_num_each_score[key][i] / sum(user_num_each_score[key]), 3)
                if user_ratio_each_score[key][i] != 0:
                    user_entropy_rating[key] = round(
                        user_entropy_rating[key] - user_ratio_each_score[key][i] * math.log(user_ratio_each_score[key][i]), 3)
                if i < 2:
                    if sum(user_num_pos_neg[key]) == 0:
                        continue
                    else:
                        user_ratio_pos_neg[key][i] = user_num_pos_neg[key][i] / sum(user_num_pos_neg[key])
        for key in item_num_each_score.keys():
            for i in range(5):
                item_ratio_each_score[key][i] = round(item_num_each_score[key][i] / sum(item_num_each_score[key]), 3)
                if item_ratio_each_score[key][i] != 0:
                    item_entropy_rating[key] = round(
                        item_entropy_rating[key] - item_ratio_each_score[key][i] * math.log(item_ratio_each_score[key][i]), 3)
                if i < 2:
                    if sum(item_num_pos_neg[key]) == 0:
                        continue
                    else:
                        item_ratio_pos_neg[key][i] = item_num_pos_neg[key][i] / sum(item_num_pos_neg[key])
        for key in user_num_each_score.keys():
            user_fea[key].append(user_num_review[key])
            user_fea[key].extend(user_score_max_min[key])
            user_fea[key].extend(user_num_each_score[key])
            user_fea[key].extend(user_ratio_each_score[key])
            user_fea[key].append(user_entropy_rating[key])
            user_fea[key].extend(user_num_pos_neg[key])
            user_fea[key].extend(user_ratio_pos_neg[key])
        for key in item_num_each_score.keys():
            item_fea[key].append(item_num_review[key])
            item_fea[key].extend(item_score_max_min[key])
            item_fea[key].extend(item_num_each_score[key])
            item_fea[key].extend(item_ratio_each_score[key])
            item_fea[key].append(item_entropy_rating[key])
            item_fea[key].extend(item_num_pos_neg[key])
            item_fea[key].extend(item_ratio_pos_neg[key])
        meta.close()
        reviews = {}
        len_review = {}
        for index, line in enumerate(review):
            value = line.split('\t')
            reviews[index] = value[3]
            len_review[index] = len(line)
        review.close()
    
        uid_userfea = defaultdict(list)
        iid_itemfea = defaultdict(list)
        rid_metafea = defaultdict(list)
        uid_rid = []
        rid_iid = []
        rid_text = defaultdict(str)
        rid_userfea = defaultdict(list)
        rid_itemfea = defaultdict(list)
        rid_label = defaultdict(int)
        uidiid_rid = defaultdict(int)
        uidiid_label = defaultdict(int)
        train_uidiid_rid = defaultdict(int)
        train_uidiid_label = defaultdict(int)
        test_uidiid_rid = defaultdict(int)
        test_uidiid_label = defaultdict(int)
        
        meta = open(meta_path)
        stat_train_fake = 0 
        stat_train_true = 0 
        stat_test_fake = 0
        stat_test_true = 0
        for index, line in enumerate(meta):
            value = line.split()
            uid = int(value[0])
            iid = int(value[1])
            rating = round(float(value[2]), 1)
            uid_userfea[users_id[uid]] = user_fea[uid]
            iid_itemfea[items_id[iid]] = item_fea[iid]
            rid_text[index] = reviews[index]
            rid_label[index] = labels[index]
            rid_userfea[index] = user_fea[uid]
            rid_itemfea[index] = item_fea[iid]
            rid_metafea[index] = [ratings[index], len_review[index]]  #一些通用特征
            if int(value[4][2:4]) < 14:
                # 根据IfSpard January 1, 2014 以前作为训练集，之后作为测试集
                train_uidiid_rid[(users_id[uid], items_id[iid])] = index
                train_uidiid_label[(users_id[uid], items_id[iid])] = labels[index]
                if labels[index] == 0:
                    stat_train_true += 1
                else:
                    stat_train_fake += 1
            else:
                test_uidiid_rid[(users_id[uid], items_id[iid])] = index
                test_uidiid_label[(users_id[uid], items_id[iid])] = labels[index]
                if labels[index] == 0:
                    stat_test_true += 1
                else:
                    stat_test_fake += 1
            uid_rid.append([users_id[uid],   index])
            rid_iid.append([index, items_id[iid]])
            uidiid_rid[(users_id[uid], items_id[iid])] = index
            uidiid_label[(users_id[uid], items_id[iid])] = labels[index]
        # 倒置字典
        # rid_uidiid = {v:k for k,v in uidiid_rid.items()}
        print("'---------------Train and Test Information---------------' \
                      \ntrain datset statistic: traindataset number %s, true %s, fake %s, ratio %s \
                      \ntest  datset statistic: testdataset  number %s, true %s, fake %s, ratio %s \
                      \nthe ratio of train to test: %s"% 
                      (len(train_uidiid_rid), stat_train_true, stat_train_fake, round(stat_train_true/(stat_train_true+stat_train_fake),3),
                      len(test_uidiid_rid), stat_test_true, stat_test_fake, round(stat_test_true/(stat_test_true+stat_test_fake),3),
                      round(len(train_uidiid_rid)/(len(train_uidiid_rid)+len(test_uidiid_rid)),3)))
        return train_uidiid_rid, train_uidiid_label, test_uidiid_rid, test_uidiid_label, uidiid_rid, uidiid_label, uid_rid, rid_iid,\
               uid_userfea,iid_itemfea, \
               rid_metafea, rid_text, rid_userfea, rid_itemfea, rid_label
    
    else:
        meta_path = args['amazon_path'] + args['meta_path']
        meta = open(meta_path, 'r', encoding='utf-8')
        index_user = 0
        index_item = 0
        for line in meta.readlines():
            dic = json.loads(line)
            if dic['helpful'][1] < 20:
                continue
            label = dic['helpful'][0] / dic['helpful'][1]
            # 投票比例大于0.7的认为是真1，小于0.3认为是假0
            if label >= 0.7:
                if "reviewerName" not in dic.keys():
                    len_name[dic['reviewerID']] = 0
                else:
                    len_name[dic['reviewerID']] = len(dic["reviewerName"])
                num_review[dic['reviewerID']] = num_review[dic['reviewerID']] + 1
                if dic['reviewerID'] not in users_id_amazon.keys():
                    users_id_amazon[dic['reviewerID']] = index_user
                    index_user += 1
                if dic['asin'] not in items_id_amazon.keys():
                    items_id_amazon[dic['asin']] = index_item
                    index_item += 1
                if int(dic['overall']) > score_max_min[dic['reviewerID']][0]:
                    score_max_min[dic['reviewerID']][0] = int(dic['overall'])
                if int(dic['overall']) < score_max_min[dic['reviewerID']][1]:
                    score_max_min[dic['reviewerID']][1] = int(dic['overall'])
                num_each_score[dic['reviewerID']][int(dic['overall']) - 1] = num_each_score[dic['reviewerID']][
                                                                                 int(dic[
                                                                                         'overall']) - 1] + 1  # 根据评分，对每个部分自加
                if int(dic['overall']) == 4 or int(dic['overall']) == 5:  # 分数为4或5为高分
                    num_pos_neg[dic['reviewerID']][0] = num_pos_neg[dic['reviewerID']][0] + 1
                if int(dic['overall']) == 1 or int(dic['overall']) == 2:  # 分数为1或2为低分
                    num_pos_neg[dic['reviewerID']][1] = num_pos_neg[dic['reviewerID']][1] + 1
                num_total_vote[dic['reviewerID']] = num_total_vote[dic['reviewerID']] + dic['helpful'][1]
                num_help_unhelp[dic['reviewerID']][0] = num_help_unhelp[dic['reviewerID']][0] + dic['helpful'][0]
                num_help_unhelp[dic['reviewerID']][1] = num_help_unhelp[dic['reviewerID']][1] + dic['helpful'][1] - \
                                                        dic['helpful'][0]
            if label <= 0.3:
                if "reviewerName" not in dic.keys():
                    len_name[dic['reviewerID']] = 0
                else:
                    len_name[dic['reviewerID']] = len(dic["reviewerName"])
                num_review[dic['reviewerID']] = num_review[dic['reviewerID']] + 1
                if dic['reviewerID'] not in users_id_amazon.keys():
                    users_id_amazon[dic['reviewerID']] = index_user
                    index_user += 1
                if dic['asin'] not in items_id_amazon.keys():
                    items_id_amazon[dic['asin']] = index_item
                    index_item += 1
                if int(dic['overall']) > score_max_min[dic['reviewerID']][0]:
                    score_max_min[dic['reviewerID']][0] = int(dic['overall'])
                if int(dic['overall']) < score_max_min[dic['reviewerID']][1]:
                    score_max_min[dic['reviewerID']][1] = int(dic['overall'])
                num_each_score[dic['reviewerID']][int(dic['overall']) - 1] = num_each_score[dic['reviewerID']][
                                                                                 int(dic[
                                                                                         'overall']) - 1] + 1  # 根据评分，对每个部分自加
                if int(dic['overall']) == 4 or int(dic['overall']) == 5:  # 分数为4或5为高分
                    num_pos_neg[dic['reviewerID']][0] = num_pos_neg[dic['reviewerID']][0] + 1
                if int(dic['overall']) == 1 or int(dic['overall']) == 2:  # 分数为1或2为低分
                    num_pos_neg[dic['reviewerID']][1] = num_pos_neg[dic['reviewerID']][1] + 1
                num_total_vote[dic['reviewerID']] = num_total_vote[dic['reviewerID']] + dic['helpful'][1]
                num_help_unhelp[dic['reviewerID']][0] = num_help_unhelp[dic['reviewerID']][0] + dic['helpful'][0]
                num_help_unhelp[dic['reviewerID']][1] = num_help_unhelp[dic['reviewerID']][1] + dic['helpful'][1] - \
                                                        dic['helpful'][0]
        for key in num_each_score.keys():
            for i in range(5):
                ratio_each_score[key][i] = round(num_each_score[key][i] / sum(num_each_score[key]), 3)
                if ratio_each_score[key][i] != 0:
                    entropy_rating[key] = round(
                        entropy_rating[key] - ratio_each_score[key][i] * math.log(ratio_each_score[key][i]), 3)
                if i < 2:
                    if sum(num_pos_neg[key]) == 0:
                        continue
                    else:
                        ratio_pos_neg[key][i] = num_pos_neg[key][i] / sum(num_pos_neg[key])
                    avg_vote[key][i] = num_help_unhelp[key][i] / num_review[key]
                    ratio_help_unhelp[key][i] = num_help_unhelp[key][i] / sum(num_help_unhelp[key])
        for key in num_each_score.keys():
            fea[key].append(num_review[key])
            fea[key].append(len_name[key])
            fea[key].extend(score_max_min[key])
            fea[key].extend(num_each_score[key])
            fea[key].extend(ratio_each_score[key])
            fea[key].append(entropy_rating[key])
            fea[key].extend(num_pos_neg[key])
            fea[key].extend(ratio_pos_neg[key])
            fea[key].append(num_total_vote[key])
            fea[key].extend(num_help_unhelp[key])
            fea[key].extend(avg_vote[key])
            fea[key].extend(ratio_help_unhelp[key])
        meta.close()
        meta = open(meta_path, 'r', encoding='utf-8')
        for line in meta.readlines():
            dic = json.loads(line)
            single_fea = []
            if dic['helpful'][1] < 20:
                continue
            label = dic['helpful'][0] / dic['helpful'][1]
            # 投票比例大于0.7的认为是真1，小于0.3认为是假0
            if label >= 0.7:
                labels.append(1)
                single_fea.append(int(dic['overall']))
                single_fea.append(len(dic['reviewText']) / 100)
                single_fea.extend(fea[dic['reviewerID']])
                uid_iid_fea[(users_id_amazon[dic['reviewerID']], items_id_amazon[dic['asin']])].extend(single_fea)
                uid_iid_label[(users_id_amazon[dic['reviewerID']], items_id_amazon[dic['asin']])] = 1
                uid_iid_rating[(users_id_amazon[dic['reviewerID']], items_id_amazon[dic['asin']])] = int(dic['overall'])
            if label <= 0.3:
                labels.append(0)
                single_fea.append(int(dic['overall']))
                single_fea.append(len(dic['reviewText']) / 100)
                single_fea.extend(fea[dic['reviewerID']])
                uid_iid_fea[(users_id_amazon[dic['reviewerID']], items_id_amazon[dic['asin']])].extend(single_fea)
                uid_iid_label[(users_id_amazon[dic['reviewerID']], items_id_amazon[dic['asin']])] = 0
                uid_iid_rating[(users_id_amazon[dic['reviewerID']], items_id_amazon[dic['asin']])] = int(dic['overall'])
        meta.close()
    print("=" * 10+"Data import completed" + "=" * 10+'\n')
    return uid_iid_fea, uid_iid_label, uid_iid_rating



