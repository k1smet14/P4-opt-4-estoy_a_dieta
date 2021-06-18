from typing import List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modules.activations import HardSwish,HardSigmoid

def autopad(
    kernel_size: Union[int, List[int]], padding: Union[int, None] = None
) -> Union[int, List[int]]:
    """Auto padding calculation for pad='same' in TensorFlow."""
    # Pad to 'same'
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    return padding or [x // 2 for x in kernel_size]

class DWConv(nn.Module):
    """Depthwise convolution with batch normalization and activation."""

    def __init__(
        self,
        in_channel: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[int, None] = None,
        activation: Union[str, None] = "ReLU",
    ) -> None:
        """Depthwise convolution with batch normalization and activation.

        Args:
            in_channel: input channels.
            out_channels: output channels.
            kernel_size: kernel size.
            stride: stride.
            padding: input padding. If None is given, autopad is applied
                which is identical to padding='SAME' in TensorFlow.
            activation: activation name. If None is given, nn.Identity is applied
                which is no activation.
        """
        super().__init__()
        # error: Argument "padding" to "Conv2d" has incompatible type "Union[int, List[int]]";
        # expected "Union[int, Tuple[int, int]]"
        self.conv = nn.Conv2d(
            in_channel,
            out_channels,
            kernel_size,
            stride,
            padding=autopad(kernel_size, padding),  # type: ignore
            groups=math.gcd(in_channel, out_channels),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels) #nn.GroupNorm(max(out_channels//32,1),out_channels)
        self.act = nn.ReLU()#Activation(activation)()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return self.act(self.bn(self.conv(x)))


class DWConvSeparableConv(nn.Module):
    """Depthwise convolution with batch normalization and activation."""

    def __init__(
        self,
        in_channel: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[int, None] = None,
        activation: Union[str, None] = "ReLU",
        sq_factor: int = 2,
    ) -> None:
        """Depthwise convolution with batch normalization and activation.

        Args:
            in_channel: input channels.
            out_channels: output channels.
            kernel_size: kernel size.
            stride: stride.
            padding: input padding. If None is given, autopad is applied
                which is identical to padding='SAME' in TensorFlow.
            activation: activation name. If None is given, nn.Identity is applied
                which is no activation.
        """
        super().__init__()
        # error: Argument "padding" to "Conv2d" has incompatible type "Union[int, List[int]]";
        # expected "Union[int, Tuple[int, int]]"
        self.dw = DWConv(in_channel,out_channels,kernel_size,stride)
        self.se = SqueezeExcitation(out_channels,sq_factor)
        self.pw = nn.Conv2d(out_channels,out_channels,kernel_size=(1,1),bias=False)
        self.bn = nn.BatchNorm2d(out_channels) #nn.GroupNorm(max(out_channels//32,1),out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        return self.bn(self.pw(self.se(self.dw(x))))

class InvertedResidualv3(nn.Module):
    """Inverted Residual block MobilenetV3.
    Reference:
        https://github.com/d-li14/mobilenetv3.pytorch/blob/master/mobilenetv3.py
    """

    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs,sq_factor=4):
        super().__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                HardSwish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SqueezeExcitation(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                HardSwish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    (kernel_size - 1) // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                 nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SqueezeExcitation(hidden_dim) if use_se else nn.Identity(),
                HardSwish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x: torch.Tensor):
        """Forward."""
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

def make_divisible(v: float, divisor: int = 8, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels: int, squeeze_factor: int = 4):
        super().__init__()
        squeeze_channels = make_divisible(input_channels // squeeze_factor, 8)
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.hardsigmoid = HardSigmoid()

    def _scale(self, input: torch.Tensor, inplace: bool) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        return self.hardsigmoid(scale)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scale = self._scale(input, True)
        return scale * input


class Custom_mobile(nn.Module):
    def __init__(self,in_chs,stride,kernels,outs,factor,num_classes):
        super().__init__()
        self.stem = DWConv(in_chs,outs[0],3,stride=2)
        
        blocks = []
        blocks.append(DWConvSeparableConv(outs[0],outs[1],kernel_size=(3,3),stride=2,sq_factor=factor[0]))
        in_chs = outs[1]

        for i,(s,ff,kernel,out) in enumerate(zip(stride,factor[1:],kernels,outs[2:-2])):
            blocks.append(nn.Sequential(*[InvertedResidualv3(in_chs if j==0 else out[j-1],
                                                                                int(in_chs*f) if j==0 else int(out[j-1]*f),
                                                                                o,k,s if j==0 else 1, False if i==0 else True, False if i==0 else True)  for j,(f,o,k) in enumerate(zip(ff,out,kernel))]))
            in_chs = out[-1]
        blocks.append(nn.Sequential(nn.Conv2d(in_chs,outs[-2],1,1),
                                                                nn.BatchNorm2d(outs[-2]),
                                                               HardSwish()))
        in_chs = outs[-2]

        self.blocks = nn.Sequential(*blocks)
        self.classifier =nn.Sequential(nn.Conv2d(in_chs,outs[-1],1,1),
                                                                nn.BatchNorm2d(outs[-1]),
                                                                HardSwish(),
                                                                nn.AdaptiveAvgPool2d(1),
                                                                nn.Conv2d(outs[-1],num_classes,1,1,bias=True))
    
    def forward(self,x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.classifier(x).squeeze(3).squeeze(2)
        return x
