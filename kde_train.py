import timm
import torch

from src.net.custommobile import Custom_mobile
import torch.optim as optim
from src.modules.loss import F1_Focal_Loss,F1_CE_Loss
from src.trainer import TorchKDTrainer
from util import load_model, load_model_partial
from src.utils.common import get_label_counts, read_yaml
from src.dataloader import create_dataloader, create_multi_dataloader
import argparse


def train(args):
    data_config = read_yaml(cfg=args.data_config)
    # Create dataloader
    train_dl, val_dl, test_dl = create_dataloader(data_config)
    num_classes=9
    outs = [4,8,[8,8],[8,8,8,8],[32,32,32,32],128,512]
    stride = [2,2,2]
    kernels = [[3,3],[5,5,5,5],[5,5,5,5]]
    factor = [2,[2,2],[2,2,2],[2,2,2]]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    s_model = Custom_mobile(3,stride,kernels,outs,factor,num_classes)

    load_model_partial(s_model,device,saved_dir=args.sweight_path,file_name="best.pt")
    s_model.to(device)


    t_model = timm.create_model('tf_mobilenetv3_large_100',num_classes=num_classes,pretrained=True,in_chans=3)
    t_model.load_state_dict(torch.load(args.tweight_path)['model'])
    t_model.to(device)


    optimizer =  optim.Adam(s_model.parameters(), lr=1e-2,weight_decay=1e-5)
    criterion =  F1_Focal_Loss(f1rate=0.6,classes=num_classes) # Main criterion for student

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=20, verbose=True)

    # Create trainer
    trainer = TorchKDTrainer(
        t_model=t_model,
        s_model =s_model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        model_path=args.model_path,
        verbose=1,
        kd_method = args.kd_method
    )
    best_acc, best_f1 = trainer.train(
                train_dataloader=train_dl,
                n_epoch=200,
                val_dataloader=val_dl
            )

    test_loss, test_f1, test_acc = trainer.test(
        model=s_model, test_dataloader=val_dl)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna tuner.")
    parser.add_argument("--data_config", default="configs/dta/custom_taco.yaml", type=str, help="Configuration of data")
    parser.add_argument("--tweight_path", default="exp/custom_cfiar10_to_100", type=str, help="Folder of Teacher's pretrained weight")
    parser.add_argument("--sweight_path", default="exp/custom_cfiar10_to_100", type=str, help="Folder of Student's pretrained weight")
    parser.add_argument("--model_path", default="exp/kde_training", type=str, help="Folder of your model")
    parser.add_argument('--kd_method',
                        default='true_kd',
                        const='true_kd',
                        nargs='?',
                        choices=['true_kd', 'kd','skd'],
                        help="Kde method, Please select in ['true_kd', 'kd','skd']")
    args = parser.parse_args()
    train(args)