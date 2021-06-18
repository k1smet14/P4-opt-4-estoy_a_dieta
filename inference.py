"""Example code for submit.
- Author: Junghoon Kim
- Contact: placidus36@gmail.com
"""
import json
import argparse
import torch
import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from src.model import Model
from src.augmentation.policies import simple_augment_test
from src.utils.common import read_yaml
from src.utils.inference_utils import run_model
from src.net.custommobile import Custom_mobile
from util import load_model, load_model_partial

CLASSES = ['Battery', 'Clothing', 'Glass', 'Metal', 'Paper', 'Paperpack', 'Plastic', 'Plasticbag', 'Styrofoam']

class CustomImageFolder(ImageFolder):
    """ImageFolder with filename."""

    def __getitem__(self, index):
        img_gt = super(CustomImageFolder, self).__getitem__(index)
        fdir = self.imgs[index][0]
        fname = fdir.rsplit(os.path.sep, 1)[-1]
        return (img_gt + (fname,))

def get_dataloader(img_root: str, data_config: str) -> DataLoader:
    """Get dataloader.

    Note:
	Don't forget to set normalization.
    """
    # Load yaml
    data_config = read_yaml(data_config)

    transform_test_args = data_confg["AUG_TEST_PARAMS"] if data_config.get("AUG_TEST_PARAMS") else None
    # Transformation for test
    transform_test = getattr(
        __import__("src.augmentation.policies", fromlist=[""]),
        data_config["AUG_TEST"],
    )(dataset=data_config["DATASET"], img_size=data_config["IMG_SIZE"])

    dataset = CustomImageFolder(root=img_root, transform=transform_test)
    dataloader = DataLoader(
	dataset=dataset,
	batch_size=1,
	num_workers=4
    )
    return dataloader

@torch.no_grad()
def inference(model, dataloader, dst_path: str):
    result = {}
    model = model.to(device)
    model.eval()
    submission_csv = {}
    for img, _, fname in dataloader:
        img = img.to(device)
        pred, enc_data = run_model(model, img)
        pred = torch.argmax(pred)
        submission_csv[fname[0]] = CLASSES[int(pred.detach())]

    result["macs"] = enc_data
    result["submission"] = submission_csv
    j = json.dumps(result, indent=4)
    save_path = os.path.join(dst_path, 'submission.csv')
    with open(save_path, 'w') as outfile:
        json.dump(result, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit.")
    parser.add_argument(
        "--dst", default=".", type=str, help="destination path for submit"
    )
    parser.add_argument(
        "--weight",  type=str, help="model weight path"
    )
    parser.add_argument(
        "--model_config",  default="configs/model/mobilenetv3.yaml", type=str, help="model config path"
    )
    parser.add_argument(
        "--data_config",  type=str, default="configs/data/custom_taco.yaml", help="dataconfig used for training."
    )
    parser.add_argument(
	"--img_root", type=str, default='../input/data/test',help="image folder root. e.g) 'data/test'"
    )
    parser.add_argument(
        "--is_timm",action = 'store_true'
    )
    parser.add_argument(
        "--timm_name", type=str, default = 'nf_regnet_b1'
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    # prepare datalaoder
    dataloader = get_dataloader(img_root=args.img_root, data_config=args.data_config)

    # prepare model
    # model_instance = Model(args.model_config, verbose=True,is_timm=args.is_timm,timm_name=args.timm_name)
    # model_instance.model.load_state_dict(torch.load(args.weight, map_location=torch.device('cpu')))
    num_classes = 9
    outs = [4,8,[8,8],[8,8,8,8],[32,32,32,32],128,512]
    stride = [2,2,2]
    kernels = [[3,3],[5,5,5,5],[5,5,5,5]]
    factor = [2,[2,2],[2,2,2],[2,2,2]]
    model = Custom_mobile(3,stride,kernels,outs,factor,num_classes)
    # model_path='exp/custom_cifar10_to_100_taco_f1kde_large_custom'
    load_model_partial(model,device,saved_dir="exp/custom_cifar10_to_100_taco_f1kde_large_custom_nocutout_adam",file_name="best.pt")
    model.to(device)


    # inference
    # inference(model_instance.model, dataloader, args.dst)
    inference(model, dataloader, args.dst)
