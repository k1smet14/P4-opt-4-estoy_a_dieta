import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def get_loader(resize, train_mini=False):
    transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    if train_mini:
        train_dataset = ImageFolder("/opt/ml/input/data/train_mini", transform=transform)
    else:
        train_dataset = ImageFolder("/opt/ml/input/data/train", transform=transform)
    val_dataset = ImageFolder("/opt/ml/input/data/val", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=True)
    return train_loader, val_loader