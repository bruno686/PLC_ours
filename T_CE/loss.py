import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def PLC(user, item, train_mat, y, t, drop_rate):
    loss = F.binary_cross_entropy_with_logits(y, t, reduce = False)

    loss_mul = loss * t
    ind_sorted = np.argsort(loss_mul.cpu().data).cuda()
    loss_sorted = loss[ind_sorted]

    remember_rate = 1 - drop_rate
    
    # 筛选最高的其中一部分loss，重新打上标签，并加入训练集中训练
    #全部翻转
    # highest_ind_sorted = ind_sorted[int(remember_rate*len(loss_sorted)):]
    #翻转10%
    # highest_ind_sorted = ind_sorted[int((0.9+0.1*remember_rate)*len(loss_sorted)):]
    #翻转50%
    # highest_ind_sorted = ind_sorted[int((0.5+0.5*remember_rate)*len(loss_sorted)):]
    #翻转75%
    # highest_ind_sorted = ind_sorted[int((0.25+0.75*remember_rate)*len(loss_sorted)):]
    #翻转最高的25%s
    highest_ind_sorted = ind_sorted[int((0.75+0.25*remember_rate)*len(loss_sorted)):]
    
    # 翻转loss高的部分samles的标签，只把0改成1
    if len(highest_ind_sorted) > 0:
        # 防止list为空
        t[highest_ind_sorted[t[highest_ind_sorted] == 1]] = 0
        train_mat[user[highest_ind_sorted].cpu().numpy().tolist(), item[highest_ind_sorted].cpu().numpy().tolist()] = t[highest_ind_sorted].cpu().numpy()
    
    #筛选最低的其中一部分loss，重新打上标签，本身就在训练集中，不需要再重新加入
    # lowest_ind_sorted = ind_sorted[:int((0.75+0.25*remember_rate)*len(loss_sorted))]
    # t[lowest_ind_sorted] = t[lowest_ind_sorted].logical_not().float()
    # train_mat[user[lowest_ind_sorted].cpu().numpy().tolist(), item[lowest_ind_sorted].cpu().numpy().tolist()] = t[lowest_ind_sorted].cpu().numpy()

    t = torch.tensor(train_mat[user.cpu().numpy().tolist(), item.cpu().numpy().tolist()].todense()).squeeze().cuda()
    loss_update = F.binary_cross_entropy_with_logits(y, t)

    return loss_update, train_mat


def loss_function(y, t, drop_rate):
    loss = F.binary_cross_entropy_with_logits(y, t, reduce = False)

    loss_mul = loss * t
    ind_sorted = np.argsort(loss_mul.cpu().data).cuda()
    loss_sorted = loss[ind_sorted]

    remember_rate = 1 - drop_rate
    num_remember = int(remember_rate * len(loss_sorted))

    ind_update = ind_sorted[:num_remember]

    loss_update = F.binary_cross_entropy_with_logits(y[ind_update], t[ind_update])

    return loss_update
    
def loss_function_co_teaching(y1, y2, t, drop_rate):
    loss1 = F.binary_cross_entropy_with_logits(y1, t, reduce = False)
    loss2 = F.binary_cross_entropy_with_logits(y2, t, reduce = False)
    
    loss_mul1 = loss1 * t
    ind_sorted1 = np.argsort(loss_mul1.cpu().data).cuda()
    loss_sorted1 = loss1[ind_sorted1]
    
    loss_mul2 = loss2 * t
    ind_sorted2 = np.argsort(loss_mul2.cpu().data).cuda()
    loss_sorted2 = loss2[ind_sorted2]
    
    remember_rate = 1 - drop_rate
    num_remember1 = int(remember_rate * len(loss_sorted1))
    num_remember2 = int(remember_rate * len(loss_sorted2))	

    ind_update1 = ind_sorted1[:num_remember1]
    ind_update2 = ind_sorted2[:num_remember2]

    loss_update1 = F.binary_cross_entropy_with_logits(y1[ind_update2], t[ind_update2])
    loss_update2 = F.binary_cross_entropy_with_logits(y2[ind_update1], t[ind_update1])
    
    # loss_update1 = F.binary_cross_entropy_with_logits(y1[ind_update1], t[ind_update1])
    # loss_update2 = F.binary_cross_entropy_with_logits(y2[ind_update2], t[ind_update2])
    return loss_update1, loss_update2

def loss_function_tri_teaching(y1, y2, y3, t, drop_rate):
    # 2号作为中间的辅助模型
    
    loss1 = F.binary_cross_entropy_with_logits(y1, t, reduce = False)
    loss2 = F.binary_cross_entropy_with_logits(y2, t, reduce = False)
    loss3 = F.binary_cross_entropy_with_logits(y3, t, reduce = False)
    
    loss_mul1 = loss1 * t
    ind_sorted1 = np.argsort(loss_mul1.cpu().data).cuda()
    loss_sorted1 = loss1[ind_sorted1]
    
    loss_mul2 = loss2 * t
    ind_sorted2 = np.argsort(loss_mul2.cpu().data).cuda()
    loss_sorted2 = loss2[ind_sorted2]
    
    loss_mul3 = loss3 * t
    ind_sorted3 = np.argsort(loss_mul3.cpu().data).cuda()
    loss_sorted3 = loss3[ind_sorted3]
    
    remember_rate = 1 - drop_rate
    num_remember1 = int(remember_rate * len(loss_sorted1))
    num_remember2 = int(remember_rate * len(loss_sorted2))
    num_remember3 = int(remember_rate * len(loss_sorted3))	

    ind_update1 = ind_sorted1[:num_remember1]
    ind_update2 = ind_sorted2[:num_remember2]
    ind_update3 = ind_sorted3[:num_remember3]
    
    ind_1_later = ind_sorted1[num_remember1:]
    same_1_2 = ind_1_later[(ind_1_later.view(1, -1) == ind_update2[:int(num_remember1*0.3)].view(-1, 1)).any(dim=0)]
    new_ind_1_update = torch.cat((ind_update1, same_1_2))
    
    ind_3_later  = ind_sorted3[num_remember3:]
    same_3_2 = ind_3_later[(ind_3_later.view(1, -1) == ind_update2[:int(num_remember3*0.3)].view(-1, 1)).any(dim=0)]
    new_ind_3_update = torch.cat((ind_update3, same_3_2))

    loss_update1 = F.binary_cross_entropy_with_logits(y1[ind_update3], t[ind_update3])
    loss_update2 = F.binary_cross_entropy_with_logits(y2[ind_update1], t[ind_update1])
    loss_update3 = F.binary_cross_entropy_with_logits(y3[ind_update1], t[ind_update1])
    return loss_update1, loss_update2, loss_update3