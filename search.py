import pickle
import torch.nn as nn
from src.model import Model
from src.utils.macs import calc_macs
import optuna
from tqdm import tqdm
from optuna_utils import *
from dataloader import *
from train import *
from evalution import *


def objective(trial, device):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device use : {device}")
    max_f1 = 0
    epochs = 20
    criterion = nn.CrossEntropyLoss()
    resize = trial.suggest_int("resize", 32, 224, step=4)
    train_loader, val_loader = get_loader(resize)
    architecture = sample_model(trial)
    model = Model(architecture, verbose=True).to(device)
    macs = calc_macs(model, (3, resize, resize))

    if macs > 13000000:
        print("model too large")
        return max_f1
    optimizer_name = trial.suggest_categorical("optim", ["SGD", "Adam"])
    if optimizer_name == "SGD":
        lr = 1e-2
    else:
        lr = 1e-3
    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3)
    for epoch in tqdm(range(epochs)):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        f1score, valid_acc, val_loss = evaluate(model, val_loader, criterion, scheduler, device)
        if max_f1 < f1score:
            max_f1 = f1score
    print(f"macs : {macs}, f1 : {max_f1}, resize : {resize}")
    return max_f1, macs


if __name__ == "__main__":
    study = optuna.create_study(
        directions=["maximize", "minimize"],
        sampler=optuna.samplers.MOTPESampler(),
    )
    study.optimize(objective, n_trials=200)
    with open("optuna_study.pkl", 'wb') as f:
        pickle.dump(study, f)
