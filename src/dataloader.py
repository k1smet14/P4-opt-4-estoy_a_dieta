"""Tune Model.

- Author: Junghoon Kim, Jongkuk Lim, Jimyeong Kim
- Contact: placidus36@gmail.com, lim.jeikei@gmail.com, wlaud1001@snu.ac.kr
- Reference
    https://github.com/j-marple-dev/model_compression
"""
import glob
import os
from typing import Any, Dict, List, Tuple, Union

import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import ImageFolder, VisionDataset
import yaml
import numpy as np

from src.utils.data import weights_for_balanced_classes
from src.utils.torch_utils import split_dataset_index
from src.augmentation.policies import get_transforms

import cv2
import PIL.Image as Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from tqdm import tqdm

def create_dataloader(
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Simple dataloader.

    Args:
        cfg: yaml file path or dictionary type of the data.

    Returns:
        train_loader
        valid_loader
        test_loader
    """
    # Data Setup
    train_dataset, val_dataset, test_dataset = get_dataset(
        data_path=config["DATA_PATH"],
        dataset_name=config["DATASET"],
        img_size=config["IMG_SIZE"],
        val_ratio=config["VAL_RATIO"],
        transform_train=config["AUG_TRAIN"],
        transform_test=config["AUG_TEST"],
        transform_train_params=config["AUG_TRAIN_PARAMS"],
        transform_test_params=config.get("AUG_TEST_PARAMS"),
    )

    return get_dataloader(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=config["BATCH_SIZE"],
    )


def get_dataset(
    data_path: str = "./save/data",
    dataset_name: str = "CIFAR10",
    img_size: float = 32,
    val_ratio: float=0.2,
    transform_train: str = "simple_augment_train",
    transform_test: str = "simple_augment_test",
    transform_train_params: Dict[str, int] = None,
    transform_test_params: Dict[str, int] = None,
):
    """Get dataset for training and testing."""
    if not transform_train_params:
        transform_train_params = dict()
    if not transform_test_params:
        transform_test_params = dict()

    # preprocessing policies
    transform_train = getattr(
        __import__("src.augmentation.policies", fromlist=[""]),
        transform_train,
    )(dataset=dataset_name, img_size=img_size, **transform_train_params)
    transform_test = getattr(
        __import__("src.augmentation.policies", fromlist=[""]),
        transform_test,
    )(dataset=dataset_name, img_size=img_size, **transform_test_params)

    label_weights = None
    # pytorch dataset
    if dataset_name == "TACO":
        train_path = os.path.join(data_path, "train")
        val_path = os.path.join(data_path, "val")
        test_path = os.path.join(data_path, "test")

        # train_dataset = ImageFolder(root=train_path, transform=transform_train)
        val_dataset = ImageFolder(root=val_path, transform=transform_test)
        test_dataset = ImageFolder(root=test_path, transform=transform_test)
        train_dataset = MyDataset(data_dir=train_path, transform=get_transforms()['train'])
        # train_dataset = KD_TrainSet(data_dir=train_path, t_transf=get_transforms()['val'], s_transf=get_transforms()['train'])
        # val_dataset = KD_TrainSet(data_dir=val_path, transform=transform_test)
        # test_dataset = KD_Trainset(data_dir=test_path, transform=transform_test)

        # transform = get_transforms()
        # train_dataset.set_transform(transform['train'])
        # val_dataset.set_transform(transform['val'])

    else:
        Dataset = getattr(
            __import__("torchvision.datasets", fromlist=[""]), dataset_name
        )
        train_dataset = Dataset(
            root=data_path, train=True, download=True, transform=transform_train
        )
        # from train dataset, train: 80%, val: 20%
        train_length = int(len(train_dataset) * (1.0-val_ratio))
        train_dataset, val_dataset = random_split(
            train_dataset, [train_length, len(train_dataset) - train_length]
        )
        test_dataset = Dataset(
            root=data_path, train=False, download=False, transform=transform_test
        )
    return train_dataset, val_dataset, test_dataset


def get_dataloader(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get dataloader for training and testing."""

    train_loader = DataLoader(
        dataset=train_dataset,
        pin_memory=(torch.cuda.is_available()),
        shuffle=True,
        batch_size=batch_size,
        num_workers=4,
        drop_last=True
    )
    valid_loader = DataLoader(
        dataset=val_dataset,
        pin_memory=(torch.cuda.is_available()),
        shuffle=False,
        batch_size=batch_size,
        num_workers=4
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        pin_memory=(torch.cuda.is_available()),
        shuffle=False,
        batch_size=batch_size,
        num_workers=4
    )
    return train_loader, valid_loader, test_loader


class MyDataset(Dataset):
    num_classes = 9
    image_paths = []
    images = []
    img_labels = []
    classes = {
    'Battery' : 0,
    'Clothing' : 1,
    'Glass' : 2,
    'Metal' : 3,
    'Paper' : 4,
    'Paperpack' : 5,
    'Plastic' : 6,
    'Plasticbag' : 7,
    'Styrofoam' : 8,
    }
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform
        self.setup()


    def set_transform(self, transform):
        self.transform = transform
    
    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            train_path = os.path.join(self.data_dir, profile)
            file_names = os.listdir(train_path)
            if profiles != ['NoLabel']:
                label = self.classes[profile]
            for file_name in file_names:
                img_path = os.path.join(self.data_dir, profile, file_name)
                if os.path.exists(img_path):
                    # image = Image.open(img_path)
                    # image = cv2.imread(img_path)
                    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    # self.images.append(image)
                    self.image_paths.append(img_path)
                    if profiles != ['NoLabel']:
                        self.img_labels.append(label)

        # for fname in tqdm(self.image_paths, desc="Loading files in RAM"):
        #     with open(fname, "rb") as f:
        #         self.images.append(f.read())
        

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # image /= 255.0
        # image = self.images[index]

        # image = self.images[index]
        # image = np.fromstring(self.images[index], dtype = np.uint8)
        # image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        img_label = self.img_labels[index]
        if self.transform is not None:
            image_transform = self.transform(image=image)['image']
        return image_transform, img_label

    def __len__(self):
        return len(self.image_paths)




## 커스텀 데이터셋
## 같은 이미지를 두가지 사이즈로 반환
class KD_TrainSet(Dataset):
    classes = {
    'Battery' : 0,
    'Clothing' : 1,
    'Glass' : 2,
    'Metal' : 3,
    'Paper' : 4,
    'Paperpack' : 5,
    'Plastic' : 6,
    'Plasticbag' : 7,
    'Styrofoam' : 8,
    }

    def __init__(self, data_dir, mode='train', t_transf=None, s_transf=None):
        self.imgsPath, self.labels = getDataInfo(data_dir, mode=mode)
        self.t_transf = t_transf
        self.s_transf = s_transf
 
    def __len__(self):
        return len(self.imgsPath)
            
    def __getitem__(self, idx):
        img = cv2.imread(self.imgsPath[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        label = int(self.labels[idx])
        t_img = self.t_transf(image=img)['image']
        s_img = self.s_transf(image=img)['image']
        return t_img, s_img, label
    
    def get_dataset_labels(self,):
        return self.img_labels



def getDataInfo(data_dir, mode):
    classes = {
    'Battery' : 0,
    'Clothing' : 1,
    'Glass' : 2,
    'Metal' : 3,
    'Paper' : 4,
    'Paperpack' : 5,
    'Plastic' : 6,
    'Plasticbag' : 7,
    'Styrofoam' : 8,
    }
    image_paths = []
    img_labels = []
    profiles = os.listdir(data_dir)
    for profile in profiles:
        train_path = os.path.join(data_dir, profile)
        file_names = os.listdir(train_path)
        # if profiles != ['NoLabel']:
        if mode == 'train':
            label = classes[profile]
        for file_name in file_names:
            img_path = os.path.join(data_dir, profile, file_name)
            if os.path.exists(img_path):
                image_paths.append(img_path)
                # if profiles != ['NoLabel']:
                if mode == 'train':
                    img_labels.append(label)
    return image_paths, img_labels
