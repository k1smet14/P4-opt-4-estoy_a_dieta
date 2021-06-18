import os
import random
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from solution.shufflenet import shufflenet_g3_wd4
from solution.dataloader import *
from solution.loss import *
from solution.scheduler import *
from solution.evaluate import *
from solution.metric import *
from solution.utils import *


def train(params, model, train_loader, val_loader, device):
    best_val_f1 = params["best_val_f1"]
    criterion = F1_Focal_Loss(f1rate=0.6, gamma=2.0, weight=None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["LR_start"])
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=40, T_mult=2, eta_max=params["LR_max"], T_up=5, gamma=0.5)

    print("Start training..")
    for epoch in range(params["EPOCHS"]):
        epoch+=1
        avg_loss = 0
        batch_count = len(train_loader)

        for step, (imgs, labels) in enumerate(train_loader):
            start = time.time()
            imgs = imgs.to(device).float()
            labels = labels.to(device)

            output = model(imgs)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), params["clip"])
            optimizer.step()

            avg_loss+=loss.item()/batch_count
            print(f"\rEpoch:{epoch:3d}  step:{step+1:3d}/{batch_count}  time:{time.time() - start:.3f}  LR:{scheduler.get_lr()[0]:.6f}", end='')
        scheduler.step()
        if epoch % params["val_per_step"]==0:
            val_loss, val_acc, val_f1, val_score, eval_time = valid_fn(model, val_loader, criterion, macs, device)
            print(f"  loss:{avg_loss:.4f}  ({eval_time:.2f}s) vloss:{val_loss:.4f} vAcc:{val_acc:.4f} vF1:{val_f1:.4f} vScore:{val_score:.4f}")
            if best_val_f1 < val_f1:
                best_val_f1 = val_f1
        else:
            print(f"  loss:{avg_loss:.4f}")
    print("Finish training")
    print(f"best_val_f1: {best_val_f1}")
    
    
    
def evaluate(params, model, val_loader, device):
    print("eval...")
    criterion = F1_Focal_Loss(f1rate=0.6, gamma=2.0, weight=None)    
    val_loss, val_acc, val_f1, val_score, eval_time = valid_fn(model, val_loader, criterion, params["macs"], device)
    print(f"({eval_time:.2f}s) vloss:{val_loss:.4f} vAcc:{val_acc:.4f} vF1:{val_f1:.4f} vScore:{val_score:.4f}")    
    
    
    
    
if __name__ == "__main__":
    ####### hyper parameters ######
    params= {
        "SEED" : 111,
        "BATCH_SIZE" : 64,
        "EPOCHS" : 200,
        "LR_start" : 5e-5,
        "LR_max" : 2e-4,
        "clip" : 30,
        "val_per_step" : 1,
        "best_val_f1" : 0,
        "input_size" : 80,
        "data_dir" : 'input/data',
    }
    ###############################
    
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument("--customized_model", default=True, type=bool)
    parser.add_argument("--eval", default=True, type=bool)
    args = parser.parse_args()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print (f"cuda_is_available: {torch.cuda.is_available()}")
    
    torch.manual_seed(params["SEED"])
    torch.cuda.manual_seed(params["SEED"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(params["SEED"])
    random.seed(params["SEED"])
    
    train_transform = A.Compose([
        A.Resize(params["input_size"], params["input_size"]),
        A.RandomRotate90(),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(rotate_limit=0, p=0.6),
        ToTensorV2()
    ])
    test_transform = A.Compose([
        A.Resize(params["input_size"], params["input_size"]),
        ToTensorV2()
    ])
    train_dataset = TrainSet(data_dir=params["data_dir"], mode='train', transform=train_transform)
    val_dataset = TrainSet(data_dir=params["data_dir"], mode='val', transform=test_transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=params["BATCH_SIZE"], shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=params["BATCH_SIZE"], shuffle=False, num_workers=2)
    
    if args.customized_model:
        model = shufflenet_g3_wd4(num_classes=9, pretrained=False, lastConv=True, is_custom=True)
        model.load_state_dict(torch.load('pretrained/ShuffleNet_final.pt')['model'])
    else:
        model = shufflenet_g3_wd4(num_classes=9, pretrained=True, lastConv=False)
    model.to(device)
    params["macs"], par = calc_macs(model, (3,params["input_size"],params["input_size"]), return_params=True)
    print('Model MACs : ',params["macs"])
    print('Model Params : ',par)
    
    
    if args.eval:
        evaluate(params, model, val_loader, device)
    else:
        train(params, model, train_loader, val_loader, device)