import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import Module


BCE_loss = nn.BCEWithLogitsLoss()

class BCE_soft(nn.BCELoss):
    def __init__(self, beta=0.95):
        super(BCE_soft, self).__init__()
        self.beta = beta
    def forward(self, input, target):
        target = self.beta * target + (1 - self.beta) * input
        return torch.sum(target)

class BCE_hard(nn.BCELoss):
    def __init__(self, beta=0.8):
        super(BCE_hard, self).__init__()
        self.beta = beta
    def forward(self, input, target):
        z = torch.round(input)
        target = self.beta * target + (1 - self.beta) * z
        return torch.sum(target)

class Per_BootstrapCrossEntropy(nn.Module):
    def __init__(self, beta=0.95, is_hard=0):
        super(Per_BootstrapCrossEntropy, self).__init__()
        self.beta = beta
        self.is_hard = is_hard
        self.eps = 1e-6

    def forward(self, logits, labels):
        # (beta * y + (1-beta) * p) * log(p)
        probs = torch.softmax(logits,0)
        loss = -torch.sum((self.beta * labels + (1 - self.beta) * probs) * torch.log(probs))
        return loss
    
    
class BootstrapCrossEntropy(nn.Module):
    def __init__(self, beta=0.95, is_hard=0):
        super(BootstrapCrossEntropy, self).__init__()
        self.beta = beta
        self.is_hard = is_hard
        self.eps = 1e-6

    def forward(self, logits, labels):
        # (beta * y + (1-beta) * p) * log(p)
        labels = F.one_hot(labels.long(), num_classes=logits.shape[-1])
        probs = F.softmax(logits, dim=-1)
        probs = torch.clip(probs, self.eps, 1 - self.eps)

        if self.is_hard:
            pred_label = F.one_hot(torch.argmax(probs, dim=-1), num_classes=logits.shape[-1])
        else:
            pred_label = probs
        loss = torch.sum((self.beta * labels + (1 - self.beta) * pred_label) * torch.log(probs), dim=-1)
        loss = torch.mean(- loss)
        return loss


class SoftBootstrappingLoss(Module):
    """
    ``Loss(t, p) = - (beta * t + (1 - beta) * p) * log(p)``

    Args:
        beta (float): bootstrap parameter. Default, 0.95
        reduce (bool): computes mean of the loss. Default, True.
        as_pseudo_label (bool): Stop gradient propagation for the term ``(1 - beta) * p``.
            Can be interpreted as pseudo-label.
    """
    def __init__(self, beta=0.95, reduce=True, as_pseudo_label=True):
        super(SoftBootstrappingLoss, self).__init__()
        self.beta = beta
        self.reduce = reduce
        self.as_pseudo_label = as_pseudo_label

    def forward(self, y_pred, y):
        # cross_entropy = - t * log(p)
        beta_xentropy = self.beta * F.cross_entropy(y_pred, y, reduction='none')

        y_pred_a = y_pred.detach() if self.as_pseudo_label else y_pred
        # second term = - (1 - beta) * p * log(p)
        bootstrap = - (1.0 - self.beta) * torch.sum(F.softmax(y_pred_a, dim=1) * F.log_softmax(y_pred, dim=1), dim=1)

        if self.reduce:
            return torch.mean(beta_xentropy + bootstrap)
        return beta_xentropy + bootstrap


class HardBootstrappingLoss(Module):
    """
    ``Loss(t, p) = - (beta * t + (1 - beta) * z) * log(p)``
    where ``z = argmax(p)``

    Args:
        beta (float): bootstrap parameter. Default, 0.95
        reduce (bool): computes mean of the loss. Default, True.

    """
    def __init__(self, beta=0.8, reduce=True):
        super(HardBootstrappingLoss, self).__init__()
        self.beta = beta
        self.reduce = reduce

    def forward(self, y_pred, y):
        # cross_entropy = - t * log(p)
        beta_xentropy = self.beta * F.cross_entropy(y_pred, y.long(), reduction='none')

        # z = argmax(p)
        z = F.softmax(y_pred.detach(), dim=1).argmax(dim=1)
        z = z.view(-1, 1)
        bootstrap = F.log_softmax(y_pred, dim=1).gather(1, z).view(-1)
        # second term = (1 - beta) * z * log(p)
        bootstrap = - (1.0 - self.beta) * bootstrap

        if self.reduce:
            return torch.mean(beta_xentropy + bootstrap)
        return beta_xentropy + bootstrap


