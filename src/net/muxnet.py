import torch.nn as nn
import torch
from src.modules.activations import Swish, HardSigmoid, HardSwish
from src.modules.base_generator import GeneratorAbstract
_BN_ARGS_PT= {'momentum':0.1, 'eps': 1e-05}



def _get_padding(kernel_size, stride=1, dilation=1, **_):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

def _split_channels(num_chan, num_groups):
    split = [num_chan // num_groups for _ in range(num_groups)]
    split[0] += num_chan - sum(split)
    return split


def conv2d_pad(in_chs, out_chs, kernel_size, **kwargs):
    padding = kwargs.pop('padding', '')
    kwargs.setdefault('bias', False)
    if isinstance(padding, str):
        # for any string padding, the padding will be calculated for you, one of three ways
        padding = padding.lower()
        if padding == 'same':
            # TF compatible 'SAME' padding, has a performance and GPU memory allocation impact
            if _is_static_pad(kernel_size, **kwargs):
                # static case, no extra overhead
                padding = _get_padding(kernel_size, **kwargs)
                return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)
            else:
                # dynamic padding
                return Conv2dSame(in_chs, out_chs, kernel_size, **kwargs)
        elif padding == 'valid':
            # 'VALID' padding, same as padding=0
            return nn.Conv2d(in_chs, out_chs, kernel_size, padding=0, **kwargs)
        else:
            # Default to PyTorch style 'same'-ish symmetric padding
            padding = _get_padding(kernel_size, **kwargs)
            return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)
    else:
        # padding was specified as a number or pair
        return nn.Conv2d(in_chs, out_chs, kernel_size, padding=padding, **kwargs)



def select_conv2d(in_chs, out_chs, kernel_size, **kwargs):
    scale_size = kwargs.pop('scales', 0)
    if isinstance(kernel_size, list) or isinstance(scale_size, list):
        # assert 'groups' not in kwargs  # only use 'depthwise' bool arg
        # We're going to use only lists for defining the MixedConv2d kernel groups,
        # ints, tuples, other iterables will continue to pass to normal conv and specify h, w.
        if isinstance(scale_size, list):
            return MuxConv(in_chs, out_chs, kernel_size, scale_size=scale_size, **kwargs)
        else:
            return MixedConv2d(in_chs, out_chs, kernel_size, **kwargs)
    else:
        depthwise = kwargs.pop('depthwise', False)
        groups = out_chs if depthwise else kwargs.pop('groups', 1)
        return conv2d_pad(in_chs, out_chs, kernel_size, groups=groups, **kwargs)

class ChannelToSpace(nn.Module):

    def __init__(self, upscale_factor=2):
        super().__init__()
        self.bs = upscale_factor

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.bs, self.bs, C // (self.bs ** 2), H, W)  # (N, bs, bs, C//bs^2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # (N, C//bs^2, H, bs, W, bs)
        x = x.view(N, C // (self.bs ** 2), H * self.bs, W * self.bs)  # (N, C//bs^2, H * bs, W * bs)
        return x


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



class MuxConv(nn.Module):
    """ MuxConv
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, padding='', scale_size=0, groups=1, depthwise=False, **kwargs):
        super(MuxConv, self).__init__()

        scale_size = scale_size if isinstance(scale_size, list) else [scale_size]
        assert len(set(scale_size)) > 1, "use regular convolution for faster inference"

        num_groups = len(scale_size)
        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size] * num_groups
        groups = groups if isinstance(groups, list) else [groups] * num_groups

        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)

        convs = []
        for k, in_ch, out_ch, scale, _group in zip(kernel_size, in_splits, out_splits, scale_size, groups):
            # padding = (k - 1) // 2
            if scale < 0:  # space-to-channel -> learn -> channel-to-space
                # if depthwise:
                scale_r = scale**2
                _group = in_ch * scale_r
                convs.append(
                    nn.Sequential(
                        SpaceToChannel(-scale),
                        conv2d_pad(
                            in_ch * scale_r, out_ch * scale_r, k, stride=stride,
                            padding=padding, dilation=1, groups=_group, **kwargs),
                        ChannelToSpace(-scale),
                    )
                )
            elif scale > 0:  # channel-to-space -> learn -> space-to-channel
                # if depthwise:
                scale_r = scale**2
                _group = in_ch // scale_r
                convs.append(
                    nn.Sequential(
                        ChannelToSpace(scale),
                        conv2d_pad(
                            in_ch // scale_r, out_ch // scale_r, k, stride=stride,
                            padding=padding, dilation=1, groups=_group, **kwargs),
                        SpaceToChannel(scale),
                    )
                )
            else:
                # if depthwise:
                _group = out_ch
                convs.append(
                    conv2d_pad(
                        in_ch, out_ch, k, stride=stride,
                        padding=padding, dilation=1, groups=_group, **kwargs))

        self.convs = nn.ModuleList(convs)
        self.splits = in_splits
        self.scale_size = scale_size

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = []
        for spx, conv in zip(x_split, self.convs):
            x_out.append(conv(spx))
        x = torch.cat(x_out, 1)
        return x

class ChannelShuffle(nn.Module):
    # FIXME haven't used yet
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def __repr__(self):
        return '%s(groups=%d)' % (self.__class__.__name__, self.groups)

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        assert C % g == 0, "Incompatible group size {} for input channel {}".format(
            g, C
        )
        return (
            x.view(N, g, int(C / g), H, W)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(N, C, H, W)
        )


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, reduce_chs=None, act_fn=nn.ReLU(inplace=True), gate_fn=nn.Sigmoid()):
        super(SqueezeExcite, self).__init__()
        self.act_fn = act_fn
        self.gate_fn = gate_fn
        reduced_chs = reduce_chs or in_chs
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        # NOTE adaptiveavgpool can be used here, but seems to cause issues with NVIDIA AMP performance
        x_se = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
        x_se = self.conv_reduce(x_se)
        x_se = self.act_fn(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class SplitBlock(nn.Module):
    def __init__(self, ratio):
        super(SplitBlock, self).__init__()
        self.ratio = ratio

    def __repr__(self):
        return '%s(ratio=%.2f)' % (self.__class__.__name__, self.ratio)

    def forward(self, x):
        c = int(x.size(1) * self.ratio)
        return x[:, :c, :, :], x[:, c:, :, :]


class MuxInvertedResidual(nn.Module):
    """ Inverted residual block w/ Channel Shuffling w/ optional SE"""

    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, pad_type='', act_fn=nn.ReLU(inplace=True), noskip=False,
                 exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1,
                 se_ratio=0., se_reduce_mid=False, se_gate_fn=nn.Sigmoid(),
                 shuffle_type=None, bn_args=_BN_ARGS_PT, drop_connect_rate=0.,
                 split_ratio=0.75, shuffle_groups=2, dw_group_factor=1,
                 scales=0):
        super(MuxInvertedResidual, self).__init__()

        assert in_chs == out_chs, "should only be used when input channels == output channels"
        assert stride < 2, "should NOT be used to down-sample"

        self.split = SplitBlock(split_ratio)
        in_chs = int(in_chs * split_ratio)
        out_chs = int(out_chs * split_ratio)
        mid_chs = int(in_chs * exp_ratio)

        self.has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.act_fn = act_fn
        self.drop_connect_rate = drop_connect_rate

        # Point-wise expansion
        self.conv_pw = select_conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type)
        self.bn1 = nn.BatchNorm2d(mid_chs, **bn_args)

        # Depth-wise/group-wise convolution
        self.conv_dw = select_conv2d(
            mid_chs, mid_chs, dw_kernel_size, stride=stride, padding=pad_type,
            groups=mid_chs // dw_group_factor,
            scales=scales
        )
        self.bn2 = nn.BatchNorm2d(mid_chs, **bn_args)

        # Squeeze-and-excitation
        if self.has_se:
            se_base_chs = mid_chs if se_reduce_mid else in_chs
            self.se = SqueezeExcite(
                mid_chs, reduce_chs=max(1, int(se_base_chs * se_ratio)), act_fn=act_fn, gate_fn=se_gate_fn)

        # Point-wise linear projection
        self.conv_pwl = select_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn3 = nn.BatchNorm2d(out_chs, **bn_args)

        self.shuffle = ChannelShuffle(groups=shuffle_groups)

    def forward(self, x):

        x, x1 = self.split(x)

        residual = x
        
        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act_fn(x)

        # Depth-wise convolution
        
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act_fn(x)

        # Squeeze-and-excitation
        if self.has_se:
            x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            # if self.drop_connect_rate > 0.:
            #     x = drop_connect(x, self.training, self.drop_connect_rate)
            x += residual

        x = torch.cat([x, x1], dim=1)
        x = self.shuffle(x)

        return x


class InvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE"""

    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, pad_type='', act_fn=nn.ReLU(inplace=True), noskip=False,
                 exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1,
                 se_ratio=0., se_reduce_mid=False, se_gate_fn=nn.Sigmoid(),
                 shuffle_type=None, bn_args=_BN_ARGS_PT, drop_connect_rate=0.):
        super(InvertedResidual, self).__init__()
        mid_chs = int(in_chs * exp_ratio)
        self.has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.act_fn = act_fn
        self.drop_connect_rate = drop_connect_rate

        # Point-wise expansion
        self.conv_pw = select_conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type)
        self.bn1 = nn.BatchNorm2d(mid_chs, **bn_args)

        self.shuffle_type = shuffle_type
        if shuffle_type is not None and isinstance(exp_kernel_size, list):
            self.shuffle = ChannelShuffle(len(exp_kernel_size))

        # Depth-wise convolution
        self.conv_dw = select_conv2d(
            mid_chs, mid_chs, dw_kernel_size, stride=stride, padding=pad_type, depthwise=True)
        self.bn2 = nn.BatchNorm2d(mid_chs, **bn_args)

        # Squeeze-and-excitation
        if self.has_se:
            se_base_chs = mid_chs if se_reduce_mid else in_chs
            self.se = SqueezeExcite(
                mid_chs, reduce_chs=max(1, int(se_base_chs * se_ratio)), act_fn=act_fn, gate_fn=se_gate_fn)

        # Point-wise linear projection
        self.conv_pwl = select_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn3 = nn.BatchNorm2d(out_chs, **bn_args)

    def forward(self, x):
        residual = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act_fn(x)

        # FIXME haven't tried this yet
        # for channel shuffle when using groups with pointwise convs as per FBNet variants
        if self.shuffle_type == "mid":
            x = self.shuffle(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act_fn(x)

        # Squeeze-and-excitation
        if self.has_se:
            x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        return x



class SwishAutoFn(torch.autograd.Function):
        """ Memory Efficient Swish
        From: https://blog.ceshine.net/post/pytorch-memory-swish/
        """
        @staticmethod
        def forward(ctx, x):
            result = x.mul(torch.sigmoid(x))
            ctx.save_for_backward(x)
            return result

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_variables[0]
            sigmoid_x = torch.sigmoid(x)
            return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))


def swish(x, inplace=False):
   return SwishAutoFn.apply(x)

def _round_channels(channels, multiplier=1.0, divisor=8, channel_min=None):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return channels

    channels *= multiplier
    channel_min = channel_min or divisor
    new_channels = max(
        int(channels + divisor / 2) // divisor * divisor,
        channel_min)
    # Make sure that round down does not go down by more than 10%.
    if new_channels < 0.9 * channels:
        new_channels += divisor
    return new_channels

class DepthwiseSeparableConv(nn.Module):
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks with an expansion
    factor of 1.0. This is an alternative to having a IR with an optional first pw conv.
    """
    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, pad_type='', act_fn=nn.ReLU(inplace=True), noskip=False,
                 pw_kernel_size=1, pw_act=False,
                 se_ratio=0., se_gate_fn=nn.Sigmoid(),
                 bn_args=_BN_ARGS_PT, drop_connect_rate=0.):
        super(DepthwiseSeparableConv, self).__init__()
        assert stride in [1, 2]
        self.has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act  # activation after point-wise conv
        self.act_fn = act_fn
        self.drop_connect_rate = drop_connect_rate

        self.conv_dw = select_conv2d(
            in_chs, in_chs, dw_kernel_size, stride=stride, padding=pad_type, depthwise=True)
        self.bn1 = nn.BatchNorm2d(in_chs, **bn_args)

        # Squeeze-and-excitation
        if self.has_se:
            self.se = SqueezeExcite(
                in_chs, reduce_chs=max(1, int(in_chs * se_ratio)), act_fn=act_fn, gate_fn=se_gate_fn)

        self.conv_pw = select_conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = nn.BatchNorm2d(out_chs, **bn_args)

    def forward(self, x):
        residual = x

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act_fn(x)

        if self.has_se:
            x = self.se(x)

        x = self.conv_pw(x)
        x = self.bn2(x)
        if self.has_pw_act:
            x = self.act_fn(x)

        if self.has_residual:
            x += residual
        return x

class MixedConv2d(nn.Module):
    """ Mixed Grouped Convolution
    Based on MDConv and GroupedConv in MixNet impl:
      https://github.com/tensorflow/tpu/blob/master/models/official/mnasnet/mixnet/custom_layers.py
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding='', dilated=False, depthwise=False, **kwargs):
        super(MixedConv2d, self).__init__()

        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)
        in_splits = _split_channels(in_channels, num_groups)
        out_splits = _split_channels(out_channels, num_groups)
        if depthwise:
            conv_groups = out_splits
        else:
            groups = kwargs.pop('groups', 1)
            if groups > 1:
                conv_groups = _split_channels(groups, num_groups)
            else:
                conv_groups = [1] * num_groups

        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            d = 1
            # FIXME make compat with non-square kernel/dilations/strides
            if stride == 1 and dilated:
                d, k = (k - 1) // 2, 3
            # conv_groups = out_ch if depthwise else kwargs.pop('groups', 1)
            # use add_module to keep key space clean
            self.add_module(
                str(idx),
                conv2d_pad(
                    in_ch, out_ch, k, stride=stride,
                    padding=padding, dilation=d, groups=conv_groups[idx], **kwargs)
            )
        self.splits = in_splits

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = [c(x) for x, c in zip(x_split, self._modules.values())]
        x = torch.cat(x_out, 1)
        return x

class MUXNet(nn.Module):
    def __init__(self,input_ch,start_ch,n_classes,bn_args=_BN_ARGS_PT):
        super(MUXNet,self).__init__()
        stem_size = _round_channels(start_ch, 1.0, 8, None)
        self.stem = nn.Sequential(select_conv2d(input_ch, stem_size, 3, stride=2, padding=''),
                                nn.BatchNorm2d(stem_size, **bn_args),
                                DepthwiseSeparableConv(stem_size,stem_size,3,1,
                                                                                                            act_fn=nn.ReLU(inplace=True),
                                                                                                            bn_args=bn_args,
                                                                                                            se_ratio=0.25))
        ch = stem_size
        self.block1 = nn.Sequential(InvertedResidual(ch,ch,3,2,exp_ratio=4.0,se_ratio=0.75),
                                                            MuxInvertedResidual(ch,ch,exp_ratio=4.0,se_ratio=0.75,dw_group_factor=2,scales=[-2,0,2]),
                                                            MuxInvertedResidual(ch,ch,exp_ratio=4.0,se_ratio=0.75,dw_group_factor=2,scales=[-2,0,2])
                                                                                                            )
        ch*=2
        self.block2 = nn.Sequential(InvertedResidual(ch//2,ch,[3,5,7],2,exp_ratio=4.0,se_ratio=0.75,act_fn=Swish(inplace=True)),
                                                            MuxInvertedResidual(ch,ch,exp_ratio=6.0,se_ratio=0.75,dw_group_factor=2,scales=[-2,0,2],act_fn=Swish(inplace=True)),
                                                            MuxInvertedResidual(ch,ch,exp_ratio=6.0,se_ratio=0.75,dw_group_factor=2,scales=[-2,0,2],act_fn=Swish(inplace=True)))
        # ch*=2                                                                                                    
        # self.block3 = nn.Sequential(InvertedResidual(ch//2,ch,[3,5,7],2,exp_ratio=4.0,se_ratio=0.75,act_fn=Swish(inplace=True)),
        #                                                     MuxInvertedResidual(ch,ch,exp_ratio=6.0,se_ratio=0.75,dw_group_factor=2,scales=[-2,0,2],act_fn=Swish(inplace=True)),
        #                                                     MuxInvertedResidual(ch,ch,exp_ratio=6.0,se_ratio=0.75,dw_group_factor=2,scales=[-2,0,2],act_fn=Swish(inplace=True))
        #                                                                                                     )
        ch*=2
        self.block3 = InvertedResidual(ch//2,ch,3,1,exp_ratio=6.0,se_ratio=0.75,act_fn=Swish(inplace=True))

        self.classifier = nn.Sequential(nn.Conv2d(ch,n_classes,1,1),
                                                                nn.BatchNorm2d(n_classes,momentum=bn_args['momentum'],eps=bn_args['eps']),
                                                                Swish(inplace=True),
                                                                nn.AdaptiveAvgPool2d(1))
    def forward(self,x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # x = self.block4(x)
        x = self.classifier(x).squeeze(3).squeeze(2)
        return x

class MUXNet2(nn.Module):
    def __init__(self,input_ch,start_ch,n_classes,bn_args=_BN_ARGS_PT):
        super(MUXNet2,self).__init__()
        stem_size = _round_channels(start_ch, 1.0, 8, None)
        self.stem = nn.Sequential(DepthwiseSeparableConv(input_ch, stem_size, 3, stride=2,act_fn=nn.ReLU(inplace=True),
                                                                                                            bn_args=bn_args,
                                                                                                            se_ratio=0.25),
                                DepthwiseSeparableConv(stem_size,stem_size,3,1,
                                                                                                            act_fn=nn.ReLU(inplace=True),
                                                                                                            bn_args=bn_args,
                                                                                                            se_ratio=0.25))
        ch = stem_size

        self.block1 = nn.Sequential(InvertedResidual(ch,ch,3,2,exp_ratio=4.0,se_ratio=0.75),
                                                            MuxInvertedResidual(ch,ch,exp_ratio=4.0,se_ratio=0.75,dw_group_factor=2,scales=[-2,0,2]),
                                                            MuxInvertedResidual(ch,ch,exp_ratio=4.0,se_ratio=0.75,dw_group_factor=2,scales=[-2,0,2])
                                                                                                            )
        ch*=2
        self.block2 = nn.Sequential(InvertedResidual(ch//2,ch,[3,5,7],2,exp_ratio=4.0,se_ratio=0.75,act_fn=Swish(inplace=True)),
                                                            MuxInvertedResidual(ch,ch,exp_ratio=4.0,se_ratio=0.75,dw_group_factor=2,scales=[-2,0,2],act_fn=Swish(inplace=True)),
                                                            MuxInvertedResidual(ch,ch,exp_ratio=4.0,se_ratio=0.75,dw_group_factor=2,scales=[-2,0,2],act_fn=Swish(inplace=True)))
        ch*=2                                                                                                    
        self.block3 = nn.Sequential(InvertedResidual(ch//2,ch,[3,5,7],2,exp_ratio=4.0,se_ratio=0.75,act_fn=Swish(inplace=True)),
                                                            MuxInvertedResidual(ch,ch,exp_ratio=4.0,se_ratio=0.75,dw_group_factor=2,scales=[-2,0,2],act_fn=Swish(inplace=True)),
                                                            MuxInvertedResidual(ch,ch,exp_ratio=4.0,se_ratio=0.75,dw_group_factor=2,scales=[-2,0,2],act_fn=Swish(inplace=True))
                                                                                                            )
        ch*=2
        self.block4 = InvertedResidual(ch//2,ch,3,1,exp_ratio=4.0,se_ratio=0.75,act_fn=Swish(inplace=True))

        self.classifier = nn.Sequential(nn.Conv2d(ch,ch,1,1),
                                                                nn.BatchNorm2d(ch,momentum=bn_args['momentum'],eps=bn_args['eps']),
                                                                Swish(inplace=True),
                                                                nn.Conv2d(ch,n_classes,1,1),
                                                                nn.AdaptiveAvgPool2d(1))
    def forward(self,x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x).squeeze(3).squeeze(2)
        return x


class MUXNet3(nn.Module):
    def __init__(self,input_ch,start_ch,n_classes,bn_args=_BN_ARGS_PT):
        super(MUXNet3,self).__init__()
        stem_size = _round_channels(start_ch, 1.0, 8, None)
        self.stem = nn.Sequential(DepthwiseSeparableConv(input_ch, stem_size, 3, stride=2,act_fn=nn.ReLU(inplace=True),
                                                                                                            bn_args=bn_args,
                                                                                                            se_ratio=0.25),
                                DepthwiseSeparableConv(stem_size,stem_size,3,1,
                                                                                                            act_fn=nn.ReLU(inplace=True),
                                                                                                            bn_args=bn_args,
                                                                                                            se_ratio=0.25))
        ch = stem_size

        self.block1 = nn.Sequential(InvertedResidual(ch,ch,3,2,exp_ratio=4.0,se_ratio=0.75),
                                                            MuxInvertedResidual(ch,ch,exp_ratio=4.0,se_ratio=0.75,dw_group_factor=2,scales=[-2,0,2]),
                                                            MuxInvertedResidual(ch,ch,exp_ratio=4.0,se_ratio=0.75,dw_group_factor=2,scales=[-2,0,2]),
                                                            MuxInvertedResidual(ch,ch,exp_ratio=4.0,se_ratio=0.75,dw_group_factor=2,scales=[-2,0,2])
                                                                                                            )
        ch*=2
        self.block2 = nn.Sequential(InvertedResidual(ch//2,ch,[3,5,7],2,exp_ratio=4.0,se_ratio=0.75,act_fn=Swish(inplace=True)),
                                                            MuxInvertedResidual(ch,ch,exp_ratio=4.0,se_ratio=0.75,dw_group_factor=2,scales=[-2,0,2],act_fn=Swish(inplace=True)),
                                                            MuxInvertedResidual(ch,ch,exp_ratio=4.0,se_ratio=0.75,dw_group_factor=2,scales=[-2,0,2],act_fn=Swish(inplace=True)),
                                                            MuxInvertedResidual(ch,ch,exp_ratio=4.0,se_ratio=0.75,dw_group_factor=2,scales=[-2,0,2],act_fn=Swish(inplace=True)))
        ch*=2                                                                                                    
        self.block3 = nn.Sequential(InvertedResidual(ch//2,ch,[3,5,7],2,exp_ratio=4.0,se_ratio=0.75,act_fn=Swish(inplace=True)),
                                                            MuxInvertedResidual(ch,ch,exp_ratio=4.0,se_ratio=0.75,dw_group_factor=2,scales=[-2,0,2],act_fn=Swish(inplace=True)),
                                                            MuxInvertedResidual(ch,ch,exp_ratio=4.0,se_ratio=0.75,dw_group_factor=2,scales=[-2,0,2],act_fn=Swish(inplace=True)),
                                                            MuxInvertedResidual(ch,ch,exp_ratio=4.0,se_ratio=0.75,dw_group_factor=2,scales=[-2,0,2],act_fn=Swish(inplace=True))
                                                                                                            )
        ch*=2
        self.block4 = nn.Sequential(InvertedResidual(ch//2,ch,3,1,exp_ratio=4.0,se_ratio=0.75,act_fn=Swish(inplace=True)),
                                                            MuxInvertedResidual(ch,ch,exp_ratio=4.0,se_ratio=0.75,dw_group_factor=2,scales=[-2,0,2],act_fn=Swish(inplace=True)),
                                                            MuxInvertedResidual(ch,ch,exp_ratio=4.0,se_ratio=0.75,dw_group_factor=2,scales=[-2,0,2],act_fn=Swish(inplace=True)),
                                                            MuxInvertedResidual(ch,ch,exp_ratio=4.0,se_ratio=0.75,dw_group_factor=2,scales=[-2,0,2],act_fn=Swish(inplace=True))
                                                                                                            )
                                                            
        ch*=2
        self.block5 = InvertedResidual(ch//2,ch,3,1,exp_ratio=4.0,se_ratio=0.75,act_fn=Swish(inplace=True))
        self.classifier = nn.Sequential(nn.Conv2d(ch,ch,1,1),
                                                                nn.BatchNorm2d(ch,momentum=bn_args['momentum'],eps=bn_args['eps']),
                                                                Swish(inplace=True),
                                                                nn.Conv2d(ch,n_classes,1,1),
                                                                nn.AdaptiveAvgPool2d(1))
    def forward(self,x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.classifier(x).squeeze(3).squeeze(2)
        return x

