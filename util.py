import os

import numpy as np
import torch

def save_model(model, saved_dir="model", file_name="default.pt"):
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    check_point = {'model' : model.state_dict()}
    path = os.path.join(saved_dir, file_name)
    torch.save(check_point, path)
    print("save success")
    
def load_model(model, device, saved_dir="model", file_name="default.pt"):
    path = os.path.join(saved_dir, file_name)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(state_dict=checkpoint['model'])
    print("load success")
    
def load_model_partial(model,device,saved_dir="model",file_name="default.pt"):
    path = os.path.join(saved_dir, file_name)
    checkpoint = torch.load(path, map_location=device)
    model_dict = model.state_dict()
    load_ck = dict()
    for c_k,c_v in checkpoint['model'].items():
            if c_k in model_dict.keys() and model_dict[c_k].shape == c_v.shape:
                load_ck[c_k]=c_v
    print(load_ck.keys())

    model_dict.update(load_ck)
    model.load_state_dict(state_dict=model_dict)
    print("load success")