"""PyTorch transforms for data augmentation.

- Author: wlaud1001
- Email: wlaud1001@snu.ac.kr
- Reference:
    https://github.com/j-marple-dev/model_compression
"""

import torchvision.transforms as transforms

from src.augmentation.methods import RandAugmentation, SequentialAugmentation
from src.augmentation.transforms import FILLCOLOR, SquarePad, SquarePad2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import numpy as np

DATASET_NORMALIZE_INFO = {
    "CIFAR10": {"MEAN": (0.4914, 0.4822, 0.4465), "STD": (0.2470, 0.2435, 0.2616)},
    "CIFAR100": {"MEAN": (0.5071, 0.4865, 0.4409), "STD": (0.2673, 0.2564, 0.2762)},
    "IMAGENET": {"MEAN": (0.485, 0.456, 0.406), "STD": (0.229, 0.224, 0.225)},
    "TACO": {"MEAN": (0.485, 0.456, 0.406), "STD": (0.229, 0.224, 0.225)},
}


def simple_augment_train(
    dataset: str = "TACO", img_size: float = 32, n_select: int = 0, level: int = 0, n_level: int = 0,
) -> transforms.Compose:
    """Simple data augmentation rule for training CIFAR100."""
    return transforms.Compose(
        [
            # SquarePad(),
            transforms.Resize((img_size, img_size)),
            # transforms.RandomResizedCrop(
            #     size=img_size, ratio=(0.75, 1.0, 1.3333333333333333)
            # ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                DATASET_NORMALIZE_INFO[dataset]["MEAN"],
                DATASET_NORMALIZE_INFO[dataset]["STD"],
            ),
            
        ]
    )

# def simple_augment_train(
#     dataset: str = "TACO", img_size: float = 32, n_select: int = 0, level: int = 0, n_level: int = 0,
# ):
#     """Simple data augmentation rule for training CIFAR100."""
#     train_transform = A.Compose(
#         [
#             A.Resize(img_size, img_size),
#             # transforms.RandomResizedCrop(
#             #     size=img_size, ratio=(0.75, 1.0, 1.3333333333333333)
#             # ),
#             # A.RandomHorizontalFlip(),
#             A.HorizontalFlip(p=0.5),
#             A.ShiftScaleRotate(p=0.6),
#             ToTensorV2(),
#             A.Normalize(
#                 DATASET_NORMALIZE_INFO[dataset]["MEAN"],
#                 DATASET_NORMALIZE_INFO[dataset]["STD"],
#             ),
#         ]
#     )
#     # return train_transform
#     return lambda img:train_transform(image=np.array(img))
    


def simple_augment_test(
    dataset: str = "CIFAR10", img_size: float = 32
) -> transforms.Compose:
    """Simple data augmentation rule for testing CIFAR100."""
    return transforms.Compose(
        [
            # SquarePad(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                DATASET_NORMALIZE_INFO[dataset]["MEAN"],
                DATASET_NORMALIZE_INFO[dataset]["STD"],
            ),
        ]
    )


def randaugment_train(
    dataset: str = "CIFAR10",
    img_size: float = 32,
    n_select: int = 2,
    level: int = 14,
    n_level: int = 31,
) -> transforms.Compose:
    """Random augmentation policy for training CIFAR100."""
    operators = [
        "Identity",
        "AutoContrast",
        "Equalize",
        "Rotate",
        "Solarize",
        "Color",
        "Posterize",
        "Contrast",
        "Brightness",
        "Sharpness",
        "ShearX",
        "ShearY",
        "TranslateX",
        "TranslateY",
    ]
    return transforms.Compose(
        [
            # SquarePad(),
            transforms.Resize((img_size, img_size)),
            RandAugmentation(operators, n_select, level, n_level),
            transforms.RandomHorizontalFlip(),
            # SequentialAugmentation([("Cutout", 0.8, 9)]),
            transforms.ToTensor(),
            transforms.Normalize(
                DATASET_NORMALIZE_INFO[dataset]["MEAN"],
                DATASET_NORMALIZE_INFO[dataset]["STD"],
            ),
        ]
    )

def get_transforms(need=('train', 'val')):
    transformations = {}
    if 'train' in need:
        transformations['train'] = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.6),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),
            ToTensorV2(p=1.0),    
        ], p=1.0)
    if 'val' in need:
        transformations['val'] = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.6),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.0)
    return transformations