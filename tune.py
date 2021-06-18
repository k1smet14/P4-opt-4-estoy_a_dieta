"""Tune Model.

- Author: Junghoon Kim, Jongsun Shin
- Contact: placidus36@gmail.com, shinn1897@makinarocks.ai
"""
import argparse
import copy
import optuna
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from src.dataloader import create_dataloader
from src.model import Model
from src.utils.torch_utils import model_info
from src.utils.common import read_yaml
from src.utils.macs import calc_macs
from src.trainer import TorchTrainer
from typing import Any, Dict, List, Tuple, Union
from train import train
import wandb


def search_hyperparam(trial: optuna.trial.Trial) -> Dict[str, Any]:
    """Search hyperparam from user-specified search space."""
    #epochs = trial.suggest_int("epochs", low=50, high=200, step=50)
    img_size = trial.suggest_categorical("img_size", [112, 128, 224, 256])
    n_select = trial.suggest_int("n_select", low=0, high=6, step=2)
    #batch_size = trial.suggest_int("batch_size", low=16, high=64, step=16)
    return {
        "EPOCHS": 20,
        "IMG_SIZE": img_size,
        "n_select": n_select,
        "BATCH_SIZE": 64
    }


def get_block_list(path: str):
    nas_yaml = read_yaml(path)
    max_block = nas_yaml['max_blocks']
    
    config = dict()
    block_config = [dict() for _ in range(nas_yaml['max_blocks'])]
    for block in nas_yaml['blocks'].keys():
        start = nas_yaml['blocks'][block]['start']
        for i in range(start,max_block):
            n_dict = dict()
            for k in nas_yaml['blocks'][block].keys():
                if k=='start': continue
                n_dict[k] = nas_yaml['blocks'][block][k][i-start]
                
            block_config[i][block] = n_dict
            
    for i in range(max_block):
        cate = block_config[i].keys()
        block_config[i]['category'] = cate
        block_config[i]['common'] = dict()    
        block_config[i]['common']['activation'] = nas_yaml['common']['activation']
        for k in ['repeat','out_channels','stride']:
            block_config[i]['common'][k] = nas_yaml['common'][k][i]
            
    if 'SpaceToChannel' in nas_yaml:
        config['SpaceToChannel'] = nas_yaml['SpaceToChannel']
        
    config['Classifier'] = nas_yaml['Classifier']
    config['Blocks'] = block_config
    return config

    
def search_block(trial: optuna.trial.Trial, block: dict,idx: int):
    common = block['common']
    repeat = trial.suggest_int(f"m{idx}/repeat", common['repeat'][0], common['repeat'][1], step=1)
    out_ch = trial.suggest_int(f"m{idx}/out_channels", common['out_channels'][0], common['out_channels'][1], step=common['out_channels'][2])
    
    if common['stride'] == -1:
        stride = trial.suggest_int(f"m{idx}/stride", low=1, high=2)
    else:
        stride = common['stride']
        
    m = trial.suggest_categorical(f"m{idx}",block['category'])
    info = block[m]
    m_args = []
    if m == "Conv" or m == 'DWConv':
        kernel = trial.suggest_int(f"m{idx}/kernel_size", low=info['kernel'][0], high=info['kernel'][1], step=info['kernel'][2])
        activation = trial.suggest_categorical(f"m{idx}/activation", common['activation'])
        if m == "Conv":
            m_args = [out_ch, kernel, stride, None, 1, activation]
        else:
            m_args = [out_ch, kernel, stride, None, activation]
    elif m == "InvertedResidualv2":
        c = trial.suggest_int(f"m{idx}/v2_c", low=info['c'][0], high=info['c'][1], step=info['c'][2])
        t = trial.suggest_int(f"m{idx}/v2_t", low=info['t'][0], high=info['t'][1])
        m_args =[c,t, stride]
    elif m == "InvertedResidualv3":
        kernel = trial.suggest_int(f"m{idx}/kernel_size", low=info['kernel'][0], high=info['kernel'][1], step=info['kernel'][2])

        t = round(trial.suggest_int(f"m{idx}/v3_t", low=info['t'][0], high=info['t'][1],step=info['t'][2]), 1)
        c = trial.suggest_int(f"m{idx}/v3_c", low=info['c'][0], high=info['c'][1], step=info['c'][2])
        se = trial.suggest_categorical(f"m{idx}/v3_se", info['se'])
        hs = trial.suggest_categorical(f"m{idx}/v3_hs", info['hs'])
        m_args = [kernel,t,c,se,hs,stride]
    elif m == "InvertedResidual":
        multi = trial.suggest_int(f"m{idx}/multi",low=info['multi'][0], high=info['multi'][1],step=info['multi'][2])
        if multi:
            kernel = [3,5,7]
        else:
            kernel = trial.suggest_int(f"m{idx}/kernel_size", low=info['kernel'][0], high=info['kernel'][1], step=info['kernel'][2])

        exp_ratio = trial.suggest_float(f"m{idx}/exp_ratio", low=info['exp_ratio'][0], high=info['exp_ratio'][1],step=info['exp_ratio'][2])
        se_ratio = trial.suggest_float(f"m{idx}/se_ratio", low=info['se_ratio'][0], high=info['se_ratio'][1],step=info['se_ratio'][2])
        activation = trial.suggest_categorical(f"m{idx}/activation", common['activation'])
        m_args = [kernel, stride, out_ch, exp_ratio, se_ratio, activation]
    elif m == "MuxInvertedResidual":
        stride = 1
        kernel = trial.suggest_int(f"m{idx}/kernel_size", low=info['kernel'][0], high=info['kernel'][1], step=info['kernel'][2])
        multi = trial.suggest_int(f"m{idx}/multi",low=info['multi'][0], high=info['multi'][1],step=info['multi'][2])
        if multi:
            scales = [-2,0,2]
        else:
            scales = 0
        exp_ratio = trial.suggest_float(f"m{idx}/exp_ratio", low=info['exp_ratio'][0], high=info['exp_ratio'][1],step=info['exp_ratio'][2])
        se_ratio = trial.suggest_float(f"m{idx}/se_ratio", low=info['se_ratio'][0], high=info['se_ratio'][1],step=info['se_ratio'][2])
        activation = trial.suggest_categorical(f"m{idx}/activation", common['activation'])
        m_args = [kernel, stride, exp_ratio, se_ratio, scales, activation]
    else:
        return None
    return [repeat,m,m_args]


def search_model(trial: optuna.trial.Trial, args) -> List[Any]:
    model = []
    config = get_block_list(args.nas_config)
    
    if 'SpaceToChannel' in config.keys() and config['SpaceToChannel']['apply']:
        r = config['SpaceToChannel']['repeat']
        m0_repeat =  trial.suggest_int("m0/repeat", r[0], r[1], step=1)
        if m0_repeat >0:
            model.append([m0_repeat,"SpaceToChannel",""])
    
    for i, block in enumerate(config['Blocks']):
        searched = search_block(trial, block,i+1)
        if searched is not None:
            model.append(searched)
    
        
    dim = config['Classifier']['last_dim']
    num_classes = config['Classifier']['num_classes']
    last_dim = trial.suggest_int("last_dim", low=dim[0],high=dim[1],step=dim[2])
    
    # We can setup fixed structure as well
    model.append([1, "Conv", [last_dim, 1, 1]])
    model.append([1, "GlobalAvgPool", []])
    model.append([1, "FixedConv", [num_classes, 1, 1, None, 1, None]])
    return model
    
    
def objective(trial: optuna.trial.Trial, device,name,args) -> Tuple[float, int, float]:
    """Optuna objective.

    Args:
        trial
    Returns:
        float: score1(e.g. accuracy)
        int: score2(e.g. params)
    """
    MODEL_CONFIG = read_yaml(cfg=args.model_config)
    DATA_CONFIG = read_yaml(cfg=args.data_config)
    
    model_config = copy.deepcopy(MODEL_CONFIG)
    data_config = copy.deepcopy(DATA_CONFIG)

    # hyperparams: EPOCHS, IMG_SIZE, n_select, BATCH_SIZE
    hyperparams = search_hyperparam(trial)

    model_config["input_size"] = [hyperparams["IMG_SIZE"], hyperparams["IMG_SIZE"]]
    model_config["backbone"] = search_model(trial, args)

    data_config["AUG_TRAIN_PARAMS"]["n_select"] = hyperparams["n_select"]
    data_config["BATCH_SIZE"] = hyperparams["BATCH_SIZE"]
    data_config["EPOCHS"] = hyperparams["EPOCHS"]
    data_config["IMG_SIZE"] = hyperparams["IMG_SIZE"]

    log_dir = os.path.join("exp", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    # model_config, data_config
    try:
        best_f1, best_acc, macs = train(
        model_config=model_config,
        data_config=data_config,
        log_dir=log_dir,
        fp16=data_config["FP16"],
        device=device,
        limit=args.limit,
        is_tune = True
        )

        summary = wandb.init(project=name,
                            name=log_dir,
                            reinit=True,
                            job_type="logging")


        if best_f1 > 0.837:
            ret_f1 = 0.5*(1 - best_f1 / 0.837)
        else:
            ret_f1 = (1 - best_f1 / 0.837)
        
        ret_mac = float(macs) / 13860000
        score = ret_f1 + ret_mac
        summary.log({'best_f1':best_f1,"best_acc":best_acc,"macs":macs,"score":score})
        wandb.run.save()
        return score
    except:
        return 10000


def tune(args, storage: Union[str, None] = None, study_name: str = "pstage_automl"):
    gpu_id = args.gpu
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    elif 0 <= gpu_id < torch.cuda.device_count():
        device = torch.device(f"cuda:{gpu_id}")
    sampler = optuna.samplers.TPESampler(n_startup_trials=20) #optuna.samplers.MOTPESampler(n_startup_trials=20)
    if storage is not None:
        rdb_storage = optuna.storages.RDBStorage(url=storage)
    else:
        rdb_storage = None

    study = optuna.create_study(
        directions=["minimize"],
        storage=rdb_storage,
        study_name=study_name,
        sampler=sampler,
        load_if_exists=True
    )
    study.optimize(lambda trial: objective(trial, device, study_name,args), n_trials=500)

    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    complete_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trials:")
    best_trials = study.best_trials

    ## trials that satisfies Pareto Fronts
    for tr in best_trials:
        print(f"  value1:{tr.values[0]}, value2:{tr.values[1]}")
        for key, value in tr.params.items():
            print(f"    {key}:{value}")

    best_trial = get_best_trial_with_condition(study)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna tuner.")
    parser.add_argument("--gpu", default=0, type=int, help="GPU id to use")
    parser.add_argument("--storage", default="", type=str, help="RDB Storage URL for optuna.")
    parser.add_argument("--model_config", default="configs/model/mobilenetv3.yaml", type=str, help="Configuration of model")
    parser.add_argument("--data_config", default="configs/data/taco.yaml", type=str, help="Configuration of data")
    parser.add_argument("--nas_config", default="configs/NAS_block/base.yaml", type=str, help="Configuration of nas search blocks")    
    parser.add_argument("--study-name", default="pstage_automl", type=str, help="Optuna study name.")
    parser.add_argument("--limit", default=5000000, type=int, help="Limit of MACs")
    args = parser.parse_args()
    tune(args, storage=None if args.storage == "" else args.storage, study_name=args.study_name)
