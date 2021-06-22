"""Utils for model compression.

- Author: wlaud1001
- Email: wlaud1001@snu.ac.kr
- Reference:
    https://github.com/j-marple-dev/model_compression
"""

from multiprocessing import Pool
import random
from typing import Tuple
import numpy as np
import torch


def get_rand_bbox_coord(
    w: int, h: int, len_ratio: float
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Get a coordinate of random box."""
    size_hole_w = int(len_ratio * w)
    size_hole_h = int(len_ratio * h)
    x = random.randint(0, w)  # [0, w]
    y = random.randint(0, h)  # [0, h]

    x0 = max(0, x - size_hole_w // 2)
    y0 = max(0, y - size_hole_h // 2)
    x1 = min(w, x + size_hole_w // 2)
    y1 = min(h, y + size_hole_h // 2)
    return (x0, y0), (x1, y1)

def weights_for_balanced_classes(subset, nclasses):                        
    count = [0] * nclasses                            
    for i in subset:                                                         
        count[i[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weightget_rand_bbox_coord

def cutmix(input, target, beta, cutmix_prob, r):
    if beta > 0 and r < cutmix_prob:
        # generate mixed sample
        lam = np.random.beta(beta, beta)
        rand_index = torch.randperm(input.size()[0]).cuda()
        target_a = target
        target_b = target[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
        # compute output
        return input, lam, target_a, target_b

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2