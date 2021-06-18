"""Baseline train
- Author: Junghoon Kim
- Contact: placidus36@gmail.com
"""

import argparse
from datetime import datetime
import os
import yaml
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim

from src.dataloader import create_dataloader
from src.loss import CustomCriterion
from src.model import Model
from src.trainer import TorchTrainer
from src.utils.common import get_label_counts, read_yaml
from src.utils.macs import calc_macs
from src.utils.torch_utils import check_runtime, model_info
from src.modules.loss import F1_Focal_Loss
from util import load_model_partial

def train(
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    log_dir: str,
    fp16: bool,
    device: torch.device,
    is_timm : bool = False,
    timm_name : str = '',
    start_ch :int = 6,
    s2d : int = 0,
    num_classes : int = 9,
    cifar: str = '',
    is_only_test: bool = False,
    limit:int = 1e10,
    is_tune: bool = False,
) -> Tuple[float, float, float]:
    """Train."""
    # save model_config, data_config
    with open(os.path.join(log_dir, 'data.yml'), 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    with open(os.path.join(log_dir, 'model.yml'), 'w') as f:
        yaml.dump(model_config, f, default_flow_style=False)

    model_instance = Model(model_config, verbose=True,is_tune = is_tune,is_timm=is_timm,timm_name=timm_name,s2d=s2d,start_ch=start_ch,num_classes=num_classes)
    model_path = os.path.join(log_dir, "best.pt")
    print(f"Model save path: {model_path}")
    if os.path.isfile(model_path):
        model_instance.model.load_state_dict(torch.load(model_path, map_location=device))
    elif cifar != '':
        load_model_partial(model_instance.model,device,saved_dir=cifar,file_name="best.pt")

    print(model_instance.model)
    model_instance.model.to(device)

    # Create dataloader
    train_dl, val_dl, test_dl = create_dataloader(data_config)

    # Calc macs
    macs = calc_macs(model_instance.model, (3, data_config["IMG_SIZE"], data_config["IMG_SIZE"]))
    print(f"macs: {macs}")
    if macs > limit:
        return 0,0, macs
    
    optimizer = optim.AdamW(model_instance.model.parameters(), lr=data_config["INIT_LR"])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=20, verbose=True)
    
    criterion = F1_Focal_Loss(f1rate=0.6,classes=num_classes)
    # Amp loss scaler
    scaler = (
        torch.cuda.amp.GradScaler() if fp16 and device != torch.device("cpu") else None
    )

    # Create trainer
    trainer = TorchTrainer(
        model=model_instance.model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        model_path=model_path,
        verbose=1,
    )
    if not is_only_test:
        best_acc, best_f1 = trainer.train(
            train_dataloader=train_dl,
            n_epoch=data_config["EPOCHS"],
            val_dataloader=val_dl if val_dl else test_dl,
        )

    # evaluate model with test set
    model_instance.model.load_state_dict(torch.load(model_path))
    test_loss, test_f1, test_acc = trainer.test(
        model=model_instance.model, test_dataloader=val_dl if val_dl else test_dl
    )
    return test_f1, test_acc, macs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--model", default="configs/model/mobilenetv3.yaml", type=str, help="model config"
    )
    parser.add_argument(
        "--data", default="configs/data/taco.yaml", type=str, help="data config"
    )
    parser.add_argument(
        "--is_timm",action = 'store_true'
    )
    parser.add_argument(
        "--timm_name", type=str, default = 'nf_regnet_b1'
    )
    parser.add_argument(
        "--start_ch", type=int, default = 12
    )
    parser.add_argument(
        "--s2d", type=int, default = 0
    )
    parser.add_argument(
        "--classes", type=int, default = 9
    )
    parser.add_argument(
        "--cifar", type=str, default = ''
    )
    parser.add_argument(
        "--is_only_test", action='store_true'
    )
    parser.add_argument(
        "--log_dir", type=str, default = ''
    )
    parser.add_argument(
        "--is_tune", action='store_true'
    )
    args = parser.parse_args()

    model_config = read_yaml(cfg=args.model)
    data_config = read_yaml(cfg=args.data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.log_dir == '':
        log_dir = os.path.join("exp", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = args.log_dir

    test_loss, test_f1, test_acc = train(
        model_config=model_config,
        data_config=data_config,
        log_dir=log_dir,
        fp16=data_config["FP16"],
        device=device,
        is_timm = args.is_timm,
        timm_name = args.timm_name,
        start_ch = args.start_ch,
        s2d=args.s2d,
        num_classes = args.classes,
        cifar = args.cifar,
        is_only_test = args.is_only_test,
        is_tune = args.is_tune
    )
    print("log_dir = ",log_dir)
