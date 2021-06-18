from ptflops import get_model_complexity_info
import torch
import torch.nn as nn


def calc_macs(model, input_shape, return_params=False):
    macs, params = get_model_complexity_info(
        model=model,
        input_res=input_shape,
        as_strings=False,
        print_per_layer_stat=False,
        verbose=False,
        ignore_modules=[nn.ReLU, nn.PReLU, nn.ELU, nn.LeakyReLU, nn.ReLU6],
    )
    if return_params:
        return macs, params
    else:
        return macs


def calc_LB(macs, val_f1, st_macs = 13878369, st_f1 = 0.8367):
    mac_term = macs/st_macs
    f1_term = 1-(val_f1/st_f1)
    if(val_f1 >= st_f1):
        f1_term/=2
    elif(val_f1 < 0.5):
        f1_term = 1.0
    return mac_term + f1_term