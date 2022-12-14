import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn import CrossEntropyLoss

def con_loss(features, labels):
    B, _ = features.shape
    features = F.normalize(features)
    cos_matrix = features.mm(features.t())
    pos_label_matrix = torch.stack([labels == labels[i] for i in range(B)]).float()
    neg_label_matrix = 1 - pos_label_matrix
    pos_cos_matrix = 1 - cos_matrix
    neg_cos_matrix = cos_matrix - 0.4
    neg_cos_matrix[neg_cos_matrix < 0] = 0
    loss = (pos_cos_matrix * pos_label_matrix).sum() + (neg_cos_matrix * neg_label_matrix).sum()
    loss /= (B * B)
    return loss



class PairwiseLoss(nn.Module):
    def __init__(self, num_classes, lamba=0.1, epsilon=0.1, use_gpu=True, size_average=True):
        super(PairwiseLoss, self).__init__()
        self.lamba = lamba
        self.ce = nn.CrossEntropyLoss()

    def PairwiseConfusion(self, features, targets):
        batch_size = features.size(0)
        if float(batch_size) % 2 != 0:
            raise Exception('Incorrect batch size provided')
        batch_left = features[:int(0.5 * batch_size)]
        target_left = targets[:int(0.5 * batch_size)]
        batch_right = features[int(0.5 * batch_size):]
        target_right = targets[int(0.5 * batch_size):]
        gamma = 1 - torch.eq(target_left, target_right).float()
        norm = torch.norm(batch_left - batch_right, 2, 1) * gamma
        loss = norm.sum() / float(batch_size)
        return loss

    def forward(self, inputs, targets):
        pair_loss = self.PairwiseConfusion(inputs, targets)
        ce_loss = self.ce(inputs, targets)
        loss = pair_loss * self.lamba + ce_loss
        return loss

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            if alpha:
                #                 self.alpha = t.ones(class_num, 1, requires_grad=True)
                self.alpha = torch.tensor(alpha, requires_grad=True)
        #             else:
        #                 self.alpha = t.ones(class_num, 1*alpha).cuda()
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        # input.shape = (N, C)
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)  # ??????softmax ??????
        # ---------one hot start--------------#
        class_mask = inputs.data.new(N, C).fill_(0)  # ?????????input??????shape???tensor
        class_mask = class_mask.requires_grad_()  # ??????????????? ????????????????????????
        ids = targets.view(-1, 1)  # ?????????????????????
        class_mask.data.scatter_(1, ids.data, 1.)  # ??????scatter???????????????mask
        # ---------one hot end-------------------#
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        #         alpha = self.alpha[ids.data.view(-1, 1)]
        #         alpha = self.alpha[ids.view(-1)]
        alpha = self.alpha


        probs = (P * class_mask).sum(1).view(-1, 1)
        # ???softmax * one_hot ?????????0?????????????????? ??????1???????????? shape = (5, 1), 5????????????target?????????

        log_p = probs.log()

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p  # ??????????????????
        # batch_loss??????????????????batch???loss???

        # ??????????????????batch???loss???????????????
        return batch_loss
