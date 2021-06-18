import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


                                                                            ### Activation ###
class HSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0
    
    
    
    
                                                                            ### convolution ###
def conv1x1(in_channels,
            out_channels,
            stride=1,
            groups=1,
            bias=False):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        groups=groups,
        bias=bias)


def conv3x3(in_channels,
            out_channels,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=False):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias)


def depthwise_conv3x3(channels,
                      stride=1,
                      padding=1,
                      dilation=1,
                      bias=False):
    return nn.Conv2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=channels,
        bias=bias)



                                                                            ### layer block ###
    
def channel_shuffle(x, groups):
    batch, channels, height, width = x.size()
    # assert (channels % groups == 0)
    channels_per_group = channels // groups
    x = x.view(batch, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch, channels, height, width)
    return x


class ChannelShuffle(nn.Module):
    def __init__(self,
                 channels,
                 groups):
        super(ChannelShuffle, self).__init__()
        # assert (channels % groups == 0)
        if channels % groups != 0:
            raise ValueError('channels must be divisible by groups')
        self.groups = groups

    def forward(self, x):
        return channel_shuffle(x, self.groups)

    def __repr__(self):
        s = "{name}(groups={groups})"
        return s.format(
            name=self.__class__.__name__,
            groups=self.groups)
    
    
class ShuffleUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 groups,
                 downsample,
                 dropout_rate,
                 ignore_group):
        super(ShuffleUnit, self).__init__()
        self.downsample = downsample
        mid_channels = out_channels // 4

        if downsample:
            out_channels -= in_channels

        self.compress_conv1 = conv1x1(in_channels=in_channels,out_channels=mid_channels,groups=(1 if ignore_group else groups))
        self.compress_bn1 = nn.BatchNorm2d(num_features=mid_channels)
        self.c_shuffle = ChannelShuffle(channels=mid_channels,groups=groups)
        self.dw_conv2 = depthwise_conv3x3(channels=mid_channels,stride=(2 if self.downsample else 1))
        self.dw_bn2 = nn.BatchNorm2d(num_features=mid_channels)
        self.expand_conv3 = conv1x1(in_channels=mid_channels,out_channels=out_channels,groups=groups)
        self.expand_bn3 = nn.BatchNorm2d(num_features=out_channels)
        if downsample:
            self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        elif dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate, inplace=True)
        self.dropout_rate = dropout_rate
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.compress_conv1(x)
        x = self.compress_bn1(x)
        x = self.activ(x)
        x = self.c_shuffle(x)
        x = self.dw_conv2(x)
        x = self.dw_bn2(x)
        x = self.expand_conv3(x)
        x = self.expand_bn3(x)
        if self.downsample:
            identity = self.avgpool(identity)
            x = torch.cat((x, identity), dim=1)
        else:
            if self.dropout_rate > 0.0:
                x = self.dropout(x)
            x = x + identity
        x = self.activ(x)
        return x
    
    
class ShuffleInitBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(ShuffleInitBlock, self).__init__()

        self.conv = conv3x3(in_channels=in_channels,out_channels=out_channels,stride=2)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activ = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        x = self.pool(x)
        return x

                                                                            ### Model Define ###
class ShuffleNet(nn.Module):
    def __init__(self,
                 channels,
                 init_block_channels,
                 groups,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000,
                 dropout_rate=0.0,
                 lastConv=False):
        super(ShuffleNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        self.lastConv = lastConv
        self.features = nn.Sequential()
        self.features.add_module("init_block", ShuffleInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                downsample = (j == 0)
                ignore_group = (i == 0) and (j == 0)
                stage.add_module("unit{}".format(j + 1),
                                 ShuffleUnit(in_channels=in_channels,
                                             out_channels=out_channels,
                                             groups=groups,
                                             downsample=downsample,
                                             dropout_rate=dropout_rate,
                                             ignore_group=ignore_group)
                                )
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_pool", nn.AdaptiveAvgPool2d(1))

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features.init_block(x)
        x = self.features.stage1(x)
        x = self.features.stage2(x)
        x = self.features.stage3(x)
        
        x = self.features.final_pool(x)
        if self.lastConv:
            x = self.output(x)
            x = x.view(x.size(0), -1)
        else:
            x = x.view(x.size(0), -1)
            x = self.output(x)
        return x


                                                                            ### Load Function ###
def get_shufflenet(groups,
                   width_scale,
                   model_name=None,
                   pretrained=False,
                   dropout_rate=0.0,
                   num_classes=1000,
                   lastConv=True,
                   is_custom=False,
                   root=os.path.join("~", ".torch", "models"),
                   **kwargs):

    init_block_channels = 24
    if is_custom:
        layers = [2, 5, 2]
    else:
        layers = [4, 8, 4]
    channels_per_layers = [240, 480, 960]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
        init_block_channels = int(init_block_channels * width_scale)

    net = ShuffleNet(channels=channels,init_block_channels=init_block_channels,groups=groups,lastConv=lastConv,dropout_rate=dropout_rate,**kwargs)

    if pretrained:
        pretrained_state = torch.load('pretrained/shufflenet_g3_wd4.pth')
        model_dict = net.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_dict}
        net.load_state_dict(pretrained_state)

    if lastConv:
        classifier = nn.Conv2d(240, num_classes, kernel_size=(1,1), stride=(1,1), bias=False)
    else:
        classifier = nn.Linear(in_features=240, out_features=num_classes, bias=True)
        
    if dropout_rate == 0.0:
        net.output = classifier
    else:
        net.output = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            classifier
        )
    
    return net


def shufflenet_g3_wd4(**kwargs):
    return get_shufflenet(groups=3, width_scale=0.25, model_name="shufflenet_g3_wd4", **kwargs)
