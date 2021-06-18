"""Model parser and model.

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

from typing import Dict, List, Type, Union

import torch
import torch.nn as nn
import yaml
import timm

from src.modules import ModuleGenerator
from src.net.muxnet import MUXNet, MUXNet2, MUXNet3

from src.net.custommobile import Custom_mobile
class SpaceToChannel(nn.Module):

    def __init__(self, downscale_factor=2):
        super().__init__()
        self.bs = downscale_factor

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x


class Model(nn.Module):
    """Base model class."""

    def __init__(
        self,
        cfg: Union[str, Dict[str, Type]] = "./model_configs/show_case.yaml",
        verbose: bool = False,
        is_timm: bool = False,
        timm_name: str = 'nf_regnet_b1',
        start_ch: int = 12,
        s2d:int =0,
        num_classes:int = 9,
        is_tune: bool = False
    ) -> None:
        """Parse model from the model config file.

        Args:
            cfg: yaml file path or dictionary type of the model.
            verbose: print the model parsing information.
        """
        super().__init__()
        self.model = None

        in_ch = 3
        if s2d!=0:
            s2ds = []
            for i in range(s2d):
                in_ch*=4
                s2ds.append(SpaceToChannel(2))
            self.model = nn.Sequential(*s2ds)

        if is_timm:
            if timm_name=='mux':
                if self.model is not None:
                     self.model = nn.Sequential(self.model,MUXNet(in_ch,start_ch,num_classes))
                else:
                    self.model = MUXNet(in_ch,start_ch,num_classes)
            elif timm_name=='mux2':
                if self.model is not None:
                     self.model = nn.Sequential(self.model,MUXNet2(in_ch,start_ch,num_classes))
                else:
                    self.model = MUXNet2(in_ch,start_ch, num_classes)
            elif timm_name=='mux3':
                if self.model is not None:
                     self.model = nn.Sequential(self.model,MUXNet3(in_ch,start_ch,num_classes))
                else:
                    self.model = MUXNet3(in_ch,start_ch,num_classes)
            else:
                if self.model is not None:
                    self.model = nn.Sequential(self.model,timm.create_model(timm_name,num_classes=num_classes,pretrained=True,in_chans=in_ch))
                else:
                    self.model = timm.create_model(timm_name,num_classes=num_classes,pretrained=True,in_chans=in_ch)
            
        
        elif is_tune:
            self.model_parser = ModelParser(cfg=cfg, verbose=verbose)
            self.model = self.model_parser.model
        else:
            outs = [4,8,[8,8],[8,8,8,8],[32,32,32,32],128,512]
            stride = [2,2,2]
            kernels = [[3,3],[5,5,5,5],[5,5,5,5]]
            factor = [2,[2,2],[2,2,2],[2,2,2]]

            self.model = Custom_mobile(3,stride,kernels,outs,factor,num_classes)
        
        


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return self.forward_one(x)

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """Forward onetime."""

        return self.model(x)


class ModelParser:
    """Generate PyTorch model from the model yaml file."""

    def __init__(
        self,
        cfg: Union[str, Dict[str, Type]] = "./model_configs/show_case.yaml",
        verbose: bool = False,
    ) -> None:
        """Generate PyTorch model from the model yaml file.

        Args:
            cfg: model config file or dict values read from the model config file.
            verbose: print the parsed model information.
        """

        self.verbose = verbose
        if isinstance(cfg, dict):
            self.cfg = cfg
        else:
            with open(cfg) as f:
                self.cfg = yaml.load(f, Loader=yaml.FullLoader)

        self.in_channel = self.cfg["input_channel"]

        self.depth_multiply = self.cfg["depth_multiple"]
        self.width_multiply = self.cfg["width_multiple"]

        # error: Incompatible types in assignment (expression has type "Type[Any]",
        # variable has type "List[Union[int, str, float]]")
        self.model_cfg: List[Union[int, str, float]] = self.cfg["backbone"]  # type: ignore

        self.model = self._parse_model()

    def log(self, msg: str):
        """Log."""
        if self.verbose:
            print(msg)

    def _parse_model(self) -> nn.Sequential:
        """Parse model."""
        layers: List[nn.Module] = []
        log: str = (
            f"{'idx':>3} | {'n':>3} | {'params':>10} "
            f"| {'module':>15} | {'arguments':>20} | {'in_channel':>12} | {'out_channel':>13}"
        )
        self.log(log)
        self.log(len(log) * "-")  # type: ignore

        in_channel = self.in_channel
        for i, (repeat, module, args) in enumerate(self.model_cfg):  # type: ignore
            repeat = (
                max(round(repeat * self.depth_multiply), 1) if repeat > 1 else repeat
            )

            module_generator = ModuleGenerator(module, in_channel)(  # type: ignore
                *args,
                width_multiply=self.width_multiply,
            )
            m = module_generator(repeat=repeat)

            layers.append(m)
            in_channel = module_generator.out_channel

            log = (
                f"{i:3d} | {repeat:3d} | "
                f"{m.n_params:10,d} | {m.type:>15} | {str(args):>20} | "
                f"{str(module_generator.in_channel):>12}"
                f"{str(module_generator.out_channel):>13}"
            )

            self.log(log)

        parsed_model = nn.Sequential(*layers)
        n_param = sum([x.numel() for x in parsed_model.parameters()])
        n_grad = sum([x.numel() for x in parsed_model.parameters() if x.requires_grad])
        # error: Incompatible return value type (got "Tuple[Sequential, List[int]]",
        # expected "Tuple[Module, List[Optional[int]]]")
        self.log(
            f"Model Summary: {len(list(parsed_model.modules())):,d} "
            f"layers, {n_param:,d} parameters, {n_grad:,d} gradients"
        )

        return parsed_model
