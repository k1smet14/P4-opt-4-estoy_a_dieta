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

from src.scheduler import CosineAnnealingWarmupRestarts
from imgclsmob.pytorch.pytorchcv.model_provider import get_model as ptcv_get_model
from shufflenetv2_2.shufflenet_v2 import Network
from mobilenetv2.models.imagenet.mobilenetv2 import MobileNetV2
from mobilenetv3.mobilenetv3 import MobileNetV3, mobilenetv3_small
import timm
import random
import numpy as np

from adamp import AdamP


def train(
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    log_dir: str,
    fp16: bool,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train."""
    seed = 90
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # save model_config, data_config
    with open(os.path.join(log_dir, 'data.yml'), 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    with open(os.path.join(log_dir, 'model.yml'), 'w') as f:
        yaml.dump(model_config, f, default_flow_style=False)

    model_instance = Model(model_config, verbose=True)
    # model_instance.model = timm.create_model('tf_efficientnetv2_l_in21ft1k', pretrained=True, num_classes=9)
    # model_instance.model = ptcv_get_model('dicenet_wd5', pretrained=True)
    model_instance.model = ptcv_get_model('shufflenet_g3_wd4', pretrained=False, use_se=False)
    model_instance.model.features.final_pool = nn.AdaptiveAvgPool2d(1)
    # model_instance.model.output.conv2 = nn.Conv2d(1024, 9, 1, 1)
    
    model_instance.model.output = nn.Linear(240, 9)
    # model_instance.model = mobilenetv3_small(num_classes=9, width_mult=0.75, teacher=Trues)
    # # # model_instance.model.load_state_dict(torch.load('./shufflenetv2_2/shufflenet_v2_x0.25.pth', map_location=device), strict=False)
    # file_path = './mobilenetv3/pretrained/mobilenetv3-small-0.75-86c972c3.pth'
    # pretrained_state = torch.load(file_path)
    # model_dict = model_instance.model.state_dict()
    # pretrained_state, _ = match_state_dict(model_dict, pretrained_state)
    # model_dict.update(pretrained_state)
    # model_instance.model.load_state_dict(model_dict)

    # model_instance.model.load_state_dict(torch.load(file_path, map_location=device))

    model_path = os.path.join(log_dir, "best.pt")
    print(f"Model save path: {model_path}")
    if os.path.isfile(model_path):
        print('NO!')
        model_instance.model.load_state_dict(torch.load(model_path, map_location=device))
    model_instance.model.to(device)

    # Create dataloader
    train_dl, val_dl, test_dl = create_dataloader(data_config)

    # Calc macs
    macs = calc_macs(model_instance.model, (3, data_config["IMG_SIZE"], data_config["IMG_SIZE"]))
    print(f"macs: {macs}")
    # if macs > 1000000:
    #     return 0, 0, macs
    # Create optimizer, scheduler, criterion
    # optimizer = torch.optim.SGD(
        # model_instance.model.parameters(), lr=data_config["INIT_LR"], momentum=0.9)
    # optimizer = torch.optim.AdamW(
    #     model_instance.model.parameters(), lr=data_config["INIT_LR"])
    optimizer = AdamP(
        model_instance.model.parameters(), lr=data_config["INIT_LR"], betas=(0.9, 0.999), weight_decay=1e-1)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer=optimizer,
    #     max_lr=data_config["INIT_LR"],
    #     steps_per_epoch=len(train_dl),
    #     epochs=data_config["EPOCHS"],
    #     pct_start=0.05,
    # )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer,
    #     T_0=950,
    #     eta_min=1e-5,
    # )
    scheduler = CosineAnnealingWarmupRestarts(optimizer, 4750, 1, 0.001, 0.0, 100, 0.5)

    criterion = CustomCriterion(
        samples_per_cls=get_label_counts(data_config["DATA_PATH"])
        if data_config["DATASET"] == "TACO"
        else None,
        device=device,
    )
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
    best_acc, best_f1 = trainer.train(
        train_dataloader=train_dl,
        n_epoch=data_config["EPOCHS"],
        val_dataloader=val_dl if val_dl else test_dl,
    )

    # evaluate model with test set
    # model_instance.model.load_state_dict(torch.load(model_path))
    # test_loss, test_f1, test_acc = trainer.test(
    #     model=model_instance.model, test_dataloader=val_dl if val_dl else test_dl
    # )
    return best_f1, best_acc, macs


def train_kd(
    model_config: Dict[str, Any],
    data_config: Dict[str, Any],
    log_dir: str,
    fp16: bool,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Train."""
    seed = 90
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # save model_config, data_config
    with open(os.path.join(log_dir, 'data.yml'), 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    with open(os.path.join(log_dir, 'model.yml'), 'w') as f:
        yaml.dump(model_config, f, default_flow_style=False)

    model_instance = Model(model_config, verbose=True)
    # model_instance.model = timm.create_model('tf_efficientnetv2_s_in21ft1k', pretrained=True)
    # model_instance.model.classifier.out_features = 9
    # model_instance.model = mobilenetv3_small(num_classes=9, width_mult=0.75, teacher=False)
    # model_instance.model.output.conv2 = nn.Conv2d(1024, 9, kernel_size=(1,1), stride=(1,1))
    model_instance.model = ptcv_get_model('dicenet_wd5', pretrained=True)
    model_instance.model.features.final_pool = nn.AdaptiveAvgPool2d(1)
    model_instance.model.output.conv2 = nn.Conv2d(1024, 9, 1, 1)
    model_instance.model.load_state_dict(torch.load('./result/dicenet_wd5_img64_new/2021-06-13_23-10-16/best.pt', map_location=device))
    # file_path = './mobilenetv3/pretrained/mobilenetv3-small-0.75-86c972c3.pth'
    # pretrained_state = torch.load(file_path)
    # model_dict = model_instance.model.state_dict()
    # pretrained_state, _ = match_state_dict(model_dict, pretrained_state)
    # model_dict.update(pretrained_state)
    # model_instance.model.load_state_dict(model_dict)
    # model_instance.model.to(device)

    teacher_model = ptcv_get_model('dicenet_wd5', pretrained=True)
    teacher_model.features.final_pool = nn.AdaptiveAvgPool2d(1)
    teacher_model.output.conv2 = nn.Conv2d(1024, 9, 1, 1)
    teacher_model.load_state_dict(torch.load('./result/dicenet_wd5_img224_new/2021-06-14_00-29-00/best.pt'))
    teacher_model.to(device)

    model_path = os.path.join(log_dir, "best.pt")
    print(f"Model save path: {model_path}")
    if os.path.isfile(model_path):
        print('NO!')
        model_instance.model.load_state_dict(torch.load(model_path, map_location=device))
    model_instance.model.to(device)

    # Create dataloader
    train_dl, val_dl, test_dl = create_dataloader(data_config)

    # Calc macs
    macs = calc_macs(model_instance.model, (3, data_config["IMG_SIZE"], data_config["IMG_SIZE"]))
    print(f"macs: {macs}")

    # Create optimizer, scheduler, criterion
    # optimizer = torch.optim.SGD(
    #     model_instance.model.parameters(), lr=data_config["INIT_LR"], momentum=0.9
    # )
    # optimizer = torch.optim.AdamW(
    #     model_instance.model.parameters(), lr=data_config["INIT_LR"])
    optimizer = AdamP(
        model_instance.model.parameters(), lr=data_config["INIT_LR"])
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer=optimizer,
    #     max_lr=data_config["INIT_LR"],
    #     steps_per_epoch=len(train_dl),
    #     epochs=data_config["EPOCHS"],
    #     pct_start=0.05,
    # )
    scheduler = CosineAnnealingWarmupRestarts(optimizer, 2375, 1, 0.001, 0.0, 200, 0.5)

    criterion = CustomCriterion(
        samples_per_cls=get_label_counts(data_config["DATA_PATH"])
        if data_config["DATASET"] == "TACO"
        else None,
        device=device,
    )
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
        t_model=teacher_model,
    )
    best_acc, best_f1 = trainer.train_kd(
        train_dataloader=train_dl,
        n_epoch=data_config["EPOCHS"],
        val_dataloader=val_dl if val_dl else test_dl,
    )

    # evaluate model with test set
    # model_instance.model.load_state_dict(torch.load(model_path))
    # test_loss, test_f1, test_acc = trainer.test(
    #     model=model_instance.model, test_dataloader=val_dl if val_dl else test_dl
    # )
    return best_f1, best_acc


def match_state_dict(
	state_dict_a,
	state_dict_b,
):
	""" Filters state_dict_b to contain only states that are present in state_dict_a.

	Matching happens according to two criteria:
	    - Is the key present in state_dict_a?
	    - Does the state with the same key in state_dict_a have the same shape?

	Returns
	    (matched_state_dict, unmatched_state_dict)

	    States in matched_state_dict contains states from state_dict_b that are also
	    in state_dict_a and unmatched_state_dict contains states that have no
	    corresponding state in state_dict_a.

		In addition: state_dict_b = matched_state_dict U unmatched_state_dict.
	"""
	matched_state_dict = {
		key: state
		for (key, state) in state_dict_b.items()
		if key in state_dict_a and state.shape == state_dict_a[key].shape
	}
	unmatched_state_dict = {
		key: state
		for (key, state) in state_dict_b.items()
		if key not in matched_state_dict
	}
	return matched_state_dict, unmatched_state_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--model", default="configs/model/baseline.yaml", type=str, help="model config"
    )
    parser.add_argument(
        "--data", default="configs/data/taco.yaml", type=str, help="data config"
    )
    args = parser.parse_args()

    model_config = read_yaml(cfg=args.model)
    data_config = read_yaml(cfg=args.data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = os.path.join("result/shufflenet_g3_wd4_fca", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    best_f1, best_acc, macs = train(
        model_config=model_config,
        data_config=data_config,
        log_dir=log_dir,
        fp16=data_config["FP16"],
        device=device,
    )


# 