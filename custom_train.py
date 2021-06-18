import torch
import torchvision
from src.net.custommobile import Custom_mobile
from src.augmentation.policies import simple_augment_test,simple_augment_train
import torch.optim as optim
from src.modules.loss import F1_Focal_Loss, F1_CE_Loss
from src.trainer import TorchTrainer
from util import load_model, load_model_partial
from src.utils.common import get_label_counts, read_yaml
from src.dataloader import create_dataloader
import argparse


def train(args):
    load_data =args.dataset

    if load_data == "CIFAR100":
        num_classes = 100
        transform_train = simple_augment_train("CIFAR100",img_size=64)
        transform_test = simple_augment_test("CIFAR100",img_size=64)

        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                download=True, transform=transform_train)

        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                download=True, transform=transform_test)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                                shuffle=True, num_workers=10)
        testloader = torch.utils.data.DataLoader(testset, batch_size=512,
                                                shuffle=True, num_workers=4)
    elif load_data == "CIFAR10":
        num_classes = 10
        transform_train = simple_augment_train("CIFAR10",img_size=64)
        transform_test = simple_augment_test("CIFAR10",img_size=64)

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform_train)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform_test)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=512,
                                                shuffle=True, num_workers=10)
        testloader = torch.utils.data.DataLoader(testset, batch_size=512,
                                                shuffle=True, num_workers=4)
    else:
        num_classes = 9
        data_config = read_yaml(cfg='configs/data/taco.yaml')
        trainloader, testloader, _ = create_dataloader(data_config)


    outs = [4,8,[8,8],[8,8,8,8],[32,32,32,32],128,512]
    stride = [2,2,2]
    kernels = [[3,3],[5,5,5,5],[5,5,5,5]]
    factor = [2,[2,2],[2,2,2],[2,2,2]]
    model = Custom_mobile(3,stride,kernels,outs,factor,num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.AdamW(model.parameters(), lr=1e-2)

    criterion = F1_Focal_Loss(f1rate=0.6,classes=num_classes)
        # Amp loss scaler
    scaler = (
            torch.cuda.amp.GradScaler() if device != torch.device("cpu") else None
        )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=20, verbose=True)

    if args.weight_path !='':
        load_model_partial(model,device,saved_dir=args.weight_path,file_name="best.pt")
    model = model.to(device)

    trainer = TorchTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        model_path=args.model_path,
        verbose=1,
    )
    best_acc, best_f1 = trainer.train(
                train_dataloader=trainloader,
                n_epoch=200,
                val_dataloader=testloader
            )
    print(best_acc, best_f1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna tuner.")
    parser.add_argument("--weight_path", default="exp/custom_cfiar10_to_100", type=str, help="Folder of pretrained weight")
    parser.add_argument("--model_path", default="exp/custom_cifar10_to_100_taco", type=str, help="Folder of your model")
    parser.add_argument("--dataset", default="taco", type=str, help="Train dataset")
    args = parser.parse_args()
    train(args)