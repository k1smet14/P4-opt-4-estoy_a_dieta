import cv2
import os
import glob
from natsort import natsorted
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A

label2idx = {'Battery':0, 'Clothing':1, 'Glass':2, 'Metal':3, 'Paper':4, 'Paperpack':5, 'Plastic':6, 'Plasticbag':7, 'Styrofoam':8}

def getDataInfo(data_dir, mode):
    if mode in ['test']:
        dataPath = natsorted(glob.glob(os.path.join(data_dir,'test/NoLabel/*.jpg')))
        return dataPath
    elif mode in ['train', 'val']:
        dataPath = natsorted(glob.glob(os.path.join(data_dir,mode,'*/*.jpg')))  
    labels = []
    for path in dataPath:
        labels.append(label2idx[path.split('/')[3]])
    return dataPath, labels


class TrainSet(Dataset):
    def __init__(self, data_dir, mode='train', transform=None):
        self.imgsPath, self.labels = getDataInfo(data_dir, mode=mode)
        self.transform = transform
        
    def __len__(self):
        return len(self.imgsPath)
            
    def __getitem__(self, idx):
        img = cv2.imread(self.imgsPath[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
        label = int(self.labels[idx])
        if self.transform is not None:
            img = self.transform(image=img)['image']
        return img, label
    
    def get_dataset_labels(self,):
        return self.labels
    
class TestSet(Dataset):
    def __init__(self, data_dir, transform=None):
        self.imgsPath = getDataInfo(data_dir, mode='test')
        self.transform = transform
        
    def __len__(self):
        return len(self.imgsPath)
            
    def __getitem__(self, idx):
        img = cv2.imread(self.imgsPath[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(image=img)['image']
            
        fname = self.imgsPath[idx].split('/')[4]
        return img, fname