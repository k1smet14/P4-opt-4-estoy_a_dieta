import time
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from metric import *

def valid_fn(model, val_loader, criterion, macs, device):
    start = time.time()
    model.eval()
    with torch.no_grad():
        batch_count = len(val_loader)
        avg_loss = 0
        preds=[]
        truth=[]
        for imgs, labels in val_loader:
            imgs = imgs.to(device).float()
            labels = labels.to(device)
            
            output = model(imgs)
            loss = criterion(output, labels)
            _, pred = torch.max(output.data, 1)
            
            avg_loss += loss.item() / batch_count
            truth += labels.tolist()
            preds += pred.tolist()
        val_acc = accuracy_score(truth, preds)
        val_f1 = f1_score(truth, preds, average='macro')
        val_score =  calc_LB(macs, val_f1)
    model.train()
    eval_time = time.time()-start
    return avg_loss, val_acc, val_f1, val_score, eval_time