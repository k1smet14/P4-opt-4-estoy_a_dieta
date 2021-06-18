import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
    
        
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )
    
class F1Loss(nn.Module):
    def __init__(self, classes=9, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon
    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()

class F1_Focal_Loss(nn.Module):
    def __init__(self, f1rate=0.4, weight=None, gamma=2.0, classes=9):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.rate = f1rate
        self.f1 = F1Loss(classes=classes)
        self.focal = FocalLoss(weight=weight, gamma=gamma)

    def forward(self, preds, truth):
        f1 = self.f1(preds, truth)
        focal = self.focal(preds, truth)
        return f1*self.rate + focal*(1-self.rate)
    
class KD_loss(nn.Module):
    def __init__(self, s_loss='CE', temperature=20, alpha=0.3):
        nn.Module.__init__(self)
        self.T = temperature
        self.alpha = alpha
        self.KLDiv = nn.KLDivLoss(reduction="batchmean")
        if s_loss=='f1focal':
            self.s_loss = F1_Focal_Loss(f1rate=0.6)
        elif s_loss=='CE':
            self.s_loss = nn.CrossEntropyLoss()
        
    def forward(self, t_preds, s_preds, truth):
        dist_loss = self.KLDiv(F.log_softmax(s_preds/self.T, dim=1), F.softmax(t_preds/self.T, dim=1))
        stud_loss = self.s_loss(s_preds, truth)
        return dist_loss*(self.alpha*self.T*self.T)  +  stud_loss*(1.-self.alpha)