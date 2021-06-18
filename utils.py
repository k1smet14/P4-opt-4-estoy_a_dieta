# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
import os
import glob
import torch
import numpy as np


def save_model(model, saved_dir="model", file_name="default.pt"):
    #os.makedirs(saved_dir, exist_ok=True)
    check_point = {'model' : model.state_dict()}
    path = os.path.join(saved_dir, file_name)
    torch.save(check_point, path)
    
def load_model(model, device, saved_dir="model", file_name="default.pt", log=True):
    path = os.path.join(saved_dir, file_name)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(state_dict=checkpoint['model'])
    if log:
        print("load success")
    
    
def calculate_parameter(model, print_param=False, return_value=False):
    n_param = 0
    n_conv = 0
    for p_idx,(param_name,param) in enumerate(model.named_parameters()):
        if param.requires_grad:
            param_numpy = param.detach().cpu().numpy() # to numpy array 
            n_param += len(param_numpy.reshape(-1))
            if print_param==True:
                print ("[%d] name:[%s] shape:[%s]."%(p_idx,param_name,param_numpy.shape))
            if "conv" in param_name: n_conv+=1
    if return_value:
        return n_param
    else:
        print("-"*50+f"\nTotal number of parameters: [{n_param:,d}]\n"+"-"*50)
        print(f"Total number of Conv layer : {n_conv}")
    
    
from torch.utils.data import WeightedRandomSampler

def get_class_weight(label):
    label_unique, count = np.unique(label, return_counts=True)
    return [1-(c*2)/sum(count) for c in count]

def get_weighted_sampler(label):
    class_w = get_class_weight(label)
    exam_w = [class_w[e] for e in label]
    return WeightedRandomSampler(exam_w, len(label))

    
    
import json
import requests
import os
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

def submit(file_path = '', desc=""):
    url = urlparse('http://ec2-13-124-161-225.ap-northeast-2.compute.amazonaws.com:8000/api/v1/competition/28/presigned_url/?description=&hyperparameters={%22training%22:{},%22inference%22:{}}')
    qs = dict(parse_qsl(url.query))
    qs['description'] = desc
    parts = url._replace(query=urlencode(qs))
    url = urlunparse(parts)

    print(url)
    headers = {
        'Authorization': 'Bearer 0f527e16e65386933b5320164e9f30523c13251c'
    }
    res = requests.get(url, headers=headers)
    print(res.text)
    data = json.loads(res.text)
    
    submit_url = data['url']
    body = {
        'key':'app/Competitions/000028/Users/{}/Submissions/{}/output.csv'.format(str(data['submission']['user']).zfill(8),str(data['submission']['local_id']).zfill(4)),
        'x-amz-algorithm':data['fields']['x-amz-algorithm'],
        'x-amz-credential':data['fields']['x-amz-credential'],
        'x-amz-date':data['fields']['x-amz-date'],
        'policy':data['fields']['policy'],
        'x-amz-signature':data['fields']['x-amz-signature']
    }
    requests.post(url=submit_url, data=body, files={'file': open(file_path, 'rb')})
