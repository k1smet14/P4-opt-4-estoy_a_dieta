"""Inverted Residual v3 block.

Reference:
    https://github.com/d-li14/mobilenetv3.pytorch/blob/master/mobilenetv3.py
- Author: Junghoon Kim
- Contact: placidus36@gmail.com
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from src.modules.activations import HardSigmoid, HardSwish
from src.modules.base_generator import GeneratorAbstract
from src.utils.torch_utils import make_divisible

from torch.nn.parameter import Parameter

class InvertedResidualv3(nn.Module):
    """Inverted Residual block MobilenetV3.
    Reference:
        https://github.com/d-li14/mobilenetv3.pytorch/blob/master/mobilenetv3.py
    """

    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
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
                # SqueezeExcitation(hidden_dim) if use_se else nn.Identity(),
                # sa_layer(hidden_dim) if use_se else nn.Identity(),
                eca_layer(hidden_dim, kernel_size),
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
                # SqueezeExcitation(hidden_dim) if use_se else nn.Identity(),
                # sa_layer(hidden_dim) if use_se else nn.Identity(),
                eca_layer(hidden_dim, kernel_size),
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


class InvertedResidualv3Generator(GeneratorAbstract):
    """Bottleneck block generator."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        """Get out channel size."""
        return self._get_divisible_channel(self.args[2] * self.width_multiply)

    @property
    def base_module(self) -> nn.Module:
        """Returns module class from src.common_modules based on the class name."""
        return getattr(__import__("src.modules", fromlist=[""]), self.name)

    def __call__(self, repeat: int = 1):
        """call method.

        InvertedResidualv3 args consists,
        repeat(=n), [kernel, exp_ratio, out, SE, NL, s] //
        note original notation from paper is [exp_size, out, SE, NL, s]
        """
        module = []
        k, t, _, se, hs, s = self.args  # c is equivalent as self.out_channel
        inp, oup = self.in_channel, self.out_channel
        for i in range(repeat):
            stride = s if i == 0 else 1
            exp_size = self._get_divisible_channel(inp * t)
            module.append(
                self.base_module(
                    inp=inp,
                    hidden_dim=exp_size,
                    oup=oup,
                    kernel_size=k,
                    stride=stride,
                    use_se=se,
                    use_hs=hs,
                )
            )
            inp = oup
        return self._get_module(module)


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)



class sa_layer(nn.Module):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=64):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * groups), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * groups), 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.reshape(b * self.groups, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = torch.cat([xn, xs], dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out