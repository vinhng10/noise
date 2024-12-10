from typing import Callable, List, Optional, OrderedDict, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def sub_filter_start_end(kernel_size, sub_kernel_size):
    center = kernel_size // 2
    dev = sub_kernel_size // 2
    start, end = center - dev, center + dev + 1
    assert end - start == sub_kernel_size
    return start, end


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, "invalid kernel size: %s" % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), "kernel size should be either `int` or `tuple`"
    assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2


def build_activation(act_func, inplace=True):
    if act_func == "relu":
        return nn.ReLU(inplace=inplace)
    elif act_func == "relu6":
        return nn.ReLU6(inplace=inplace)
    elif act_func == "tanh":
        return nn.Tanh()
    elif act_func == "sigmoid":
        return nn.Sigmoid()
    elif act_func == "h_swish":
        return Hswish(inplace=inplace)
    elif act_func == "h_sigmoid":
        return Hsigmoid(inplace=inplace)
    elif act_func is None or act_func == "none":
        return None
    else:
        raise ValueError("do not support: %s" % act_func)


def val2list(val, repeat_time=1):
    if isinstance(val, list) or isinstance(val, np.ndarray):
        return val
    elif isinstance(val, tuple):
        return list(val)
    else:
        return [val for _ in range(repeat_time)]


def make_divisible(v, divisor, min_val=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_val:
    :return:
    """
    if min_val is None:
        min_val = divisor
    new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0

    def __repr__(self):
        return "Hswish()"


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0

    def __repr__(self):
        return "Hsigmoid()"


class DynamicSeparableConv2d(nn.Module):
    # Haotian: official version uses KERNEL_TRANSFORM_MODE=None,
    # but the ckpt requires it to be 1
    KERNEL_TRANSFORM_MODE = 1

    def __init__(
        self,
        max_in_channels,
        kernel_size_list,
        stride=1,
        dilation=1,
        conv_layer: Callable[..., nn.Module] = nn.Conv2d,
    ):
        super().__init__()

        self.max_in_channels = max_in_channels
        self.kernel_size_list = kernel_size_list
        self.stride = stride
        self.dilation = dilation

        if conv_layer is nn.Conv2d:
            self.conv_fn = F.conv2d
        elif conv_layer is nn.ConvTranspose2d:
            self.conv_fn = F.conv_transpose2d
        else:
            raise ValueError(f"Unsupported conv layer: {conv_layer}")

        self.conv = conv_layer(
            self.max_in_channels,
            self.max_in_channels,
            max(self.kernel_size_list),
            self.stride,
            groups=self.max_in_channels,
            bias=False,
        )

        self._ks_set = list(set(self.kernel_size_list))
        self._ks_set.sort()  # e.g., [3, 5, 7]
        if self.KERNEL_TRANSFORM_MODE is not None:
            # register scaling parameters
            # 7to5_matrix, 5to3_matrix
            scale_params = {}
            for i in range(len(self._ks_set) - 1):
                ks_small = self._ks_set[i]
                ks_larger = self._ks_set[i + 1]
                param_name = "%dto%d" % (ks_larger, ks_small)
                scale_params["%s_matrix" % param_name] = nn.Parameter(
                    torch.eye(ks_small**2)
                )
            for name, param in scale_params.items():
                self.register_parameter(name, param)

        self.active_kernel_size = max(self.kernel_size_list)

    def get_active_filter(self, in_channel, kernel_size):
        out_channel = in_channel
        max_kernel_size = max(self.kernel_size_list)

        start, end = sub_filter_start_end(max_kernel_size, kernel_size)
        filters = self.conv.weight[:out_channel, :in_channel, start:end, start:end]
        if self.KERNEL_TRANSFORM_MODE is not None and kernel_size < max_kernel_size:
            start_filter = self.conv.weight[
                :out_channel, :in_channel, :, :
            ]  # start with max kernel
            for i in range(len(self._ks_set) - 1, 0, -1):
                src_ks = self._ks_set[i]
                if src_ks <= kernel_size:
                    break
                target_ks = self._ks_set[i - 1]
                start, end = sub_filter_start_end(src_ks, target_ks)
                _input_filter = start_filter[:, :, start:end, start:end]
                _input_filter = _input_filter.contiguous()
                _input_filter = _input_filter.view(
                    _input_filter.size(0), _input_filter.size(1), -1
                )
                _input_filter = _input_filter.view(-1, _input_filter.size(2))
                _input_filter = F.linear(
                    _input_filter,
                    self.__getattr__("%dto%d_matrix" % (src_ks, target_ks)),
                )
                _input_filter = _input_filter.view(
                    filters.size(0), filters.size(1), target_ks**2
                )
                _input_filter = _input_filter.view(
                    filters.size(0), filters.size(1), target_ks, target_ks
                )
                start_filter = _input_filter
            filters = start_filter
        return filters

    def forward(self, x, kernel_size=None):
        if kernel_size is None:
            kernel_size = self.active_kernel_size
        in_channel = x.size(1)

        filters = self.get_active_filter(in_channel, kernel_size).contiguous()

        padding = get_same_padding(kernel_size)
        y = self.conv_fn(
            x, filters, None, self.stride, padding, self.dilation, in_channel
        )
        return y


class DynamicPointConv2d(nn.Module):
    def __init__(
        self, max_in_channels, max_out_channels, kernel_size=1, stride=1, dilation=1
    ):
        super().__init__()

        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

        self.conv = nn.Conv2d(
            self.max_in_channels,
            self.max_out_channels,
            self.kernel_size,
            stride=self.stride,
            bias=False,
        )

        self.active_out_channel = self.max_out_channels

    def forward(self, x, out_channel=None):
        if out_channel is None:
            out_channel = self.active_out_channel
        in_channel = x.size(1)
        filters = self.conv.weight[:out_channel, :in_channel, :, :].contiguous()

        padding = get_same_padding(self.kernel_size)
        y = F.conv2d(x, filters, None, self.stride, padding, self.dilation, 1)
        return y


class DynamicBatchNorm2d(nn.Module):
    SET_RUNNING_STATISTICS = False

    def __init__(self, max_feature_dim):
        super().__init__()

        self.max_feature_dim = max_feature_dim
        self.bn = nn.BatchNorm2d(self.max_feature_dim)

    @staticmethod
    def bn_forward(x, bn: nn.BatchNorm2d, feature_dim):
        if bn.num_features == feature_dim or DynamicBatchNorm2d.SET_RUNNING_STATISTICS:
            return bn(x)
        else:
            exponential_average_factor = 0.0

            if bn.training and bn.track_running_stats:
                # TODO: if statement only here to tell the jit to skip emitting this when it is None
                if bn.num_batches_tracked is not None:
                    bn.num_batches_tracked += 1
                    if bn.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(bn.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = bn.momentum
            return F.batch_norm(
                x,
                bn.running_mean[:feature_dim],
                bn.running_var[:feature_dim],
                bn.weight[:feature_dim],
                bn.bias[:feature_dim],
                bn.training or not bn.track_running_stats,
                exponential_average_factor,
                bn.eps,
            )

    def forward(self, x):
        feature_dim = x.size(1)
        y = self.bn_forward(x, self.bn, feature_dim)
        return y


class DynamicConv2dNormActivation(nn.Module):
    def __init__(
        self,
        in_channel_list,
        out_channel_list,
        kernel_size=3,
        stride=1,
        dilation=1,
        use_bn=True,
        act_func="relu6",
    ):
        super().__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.use_bn = use_bn
        self.act_func = act_func

        self.conv = DynamicPointConv2d(
            max_in_channels=max(self.in_channel_list),
            max_out_channels=max(self.out_channel_list),
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
        )
        if self.use_bn:
            self.bn = DynamicBatchNorm2d(max(self.out_channel_list))
        self.act = build_activation(self.act_func, inplace=True)

        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x):
        self.conv.active_out_channel = self.active_out_channel

        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act(x)
        return x


class DynamicInvertedConv2d(nn.Module):
    def __init__(
        self,
        in_channel_list: List[int],
        out_channel_list: List[int],
        kernel_size_list: Union[int, List[int]],
        expand_ratio_list: Union[int, List[int]],
        stride: int = 1,
        activation: str = "relu6",
        dw_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        self.in_channel_list = in_channel_list
        self.out_channel_list = out_channel_list

        self.kernel_size_list = val2list(kernel_size_list, 1)
        self.expand_ratio_list = val2list(expand_ratio_list, 1)

        self.stride = stride
        self.activation = activation

        max_middle_channel = round(
            max(self.in_channel_list) * max(self.expand_ratio_list)
        )
        if max(self.expand_ratio_list) == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            DynamicPointConv2d(
                                max(self.in_channel_list), max_middle_channel
                            ),
                        ),
                        ("bn", DynamicBatchNorm2d(max_middle_channel)),
                        ("act", build_activation(self.activation, inplace=True)),
                    ]
                )
            )

        if dw_layer is None:
            self.depth_conv = None
        else:
            self.depth_conv = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            DynamicSeparableConv2d(
                                max_middle_channel,
                                self.kernel_size_list,
                                self.stride,
                                conv_layer=dw_layer,
                            ),
                        ),
                        ("bn", DynamicBatchNorm2d(max_middle_channel)),
                        ("act", build_activation(self.activation, inplace=True)),
                    ]
                )
            )

        self.point_linear = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv",
                        DynamicPointConv2d(
                            max_middle_channel, max(self.out_channel_list)
                        ),
                    ),
                    ("bn", DynamicBatchNorm2d(max(self.out_channel_list))),
                ]
            )
        )

        self.active_kernel_size = max(self.kernel_size_list)
        self.active_expand_ratio = max(self.expand_ratio_list)
        self.active_out_channel = max(self.out_channel_list)

    def forward(self, x: Tensor) -> Tensor:
        in_channel = x.size(1)

        if self.inverted_bottleneck is not None:
            self.inverted_bottleneck.conv.active_out_channel = make_divisible(
                round(in_channel * self.active_expand_ratio), 8
            )

        self.depth_conv.conv.active_kernel_size = self.active_kernel_size
        self.point_linear.conv.active_out_channel = self.active_out_channel

        if self.inverted_bottleneck is not None:
            x = self.inverted_bottleneck(x)
        if self.depth_conv is not None:
            x = self.depth_conv(x)
        x = self.point_linear(x)
        return x


class DynamicInvertedResidual(nn.Module):

    def __init__(
        self,
        mobile_inverted_conv: Optional[nn.Module],
        shortcut: Optional[nn.Module],
    ):
        super().__init__()

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.mobile_inverted_conv is None:
            ...
        elif self.shortcut is None:
            x = self.mobile_inverted_conv(x)
        else:
            x = self.mobile_inverted_conv(x) + self.shortcut(x)
        return x


class OFAMCUNets(nn.Module):
    def __init__(
        self,
        n_classes=1000,
        bn_param=(0.1, 1e-3),
        dropout_rate=0.1,
        base_stage_width=None,
        base_depth=(2, 2, 2, 2, 2),
        width_mult_list=1.0,
        ks_list=3,
        expand_ratio_list=6,
        depth_list=4,
        no_mix_layer=False,
        fuse_blk1=False,
        se_stages=None,
    ):

        self.width_mult_list = val2list(width_mult_list, 1)
        self.ks_list = val2list(ks_list, 1)
        self.expand_ratio_list = val2list(expand_ratio_list, 1)
        self.depth_list = val2list(depth_list, 1)
        self.base_stage_width = base_stage_width
        self.base_depth = base_depth

        self.width_mult_list.sort()
        self.ks_list.sort()
        self.expand_ratio_list.sort()
        self.depth_list.sort()

        if base_stage_width == "google":
            base_stage_width = [32, 16, 24, 32, 64, 96, 160, 320, 1280]
        elif base_stage_width == "proxyless":
            # ProxylessNAS Stage Width
            base_stage_width = [32, 16, 24, 40, 80, 96, 192, 320, 1280]
        elif base_stage_width == "mcunet384":
            base_stage_width = [32, 16, 24, 40, 80, 96, 192, 320, 384]

        input_channel = [
            make_divisible(base_stage_width[0] * width_mult, 8)
            for width_mult in self.width_mult_list
        ]
        first_block_width = [
            make_divisible(base_stage_width[1] * width_mult, 8)
            for width_mult in self.width_mult_list
        ]
        last_channel = [
            (
                make_divisible(base_stage_width[-1] * width_mult, 8)
                if width_mult > 1.0
                else base_stage_width[-1]
            )
            for width_mult in self.width_mult_list
        ]

        # first conv layer
        if len(input_channel) == 1:
            first_conv = ConvLayer(
                3,
                max(input_channel),
                kernel_size=3,
                stride=2,
                use_bn=True,
                act_func="relu6",
                ops_order="weight_bn_act",
            )
        else:
            first_conv = DynamicConv2dNormActivation(
                in_channel_list=val2list(3, len(input_channel)),
                out_channel_list=input_channel,
                kernel_size=3,
                stride=2,
                act_func="relu6",
            )

        # first block
        if len(first_block_width) == 1:
            first_block_conv = DynamicInvertedConv2d(
                in_channel_list=max(input_channel),
                out_channel_list=max(first_block_width),
                kernel_size_list=3,
                expand_ratio_list=1,
                stride=1,
                activation="relu6",
                dw_layer=None,
            )
        else:
            first_block_conv = DynamicInvertedConv2d(
                in_channel_list=input_channel,
                out_channel_list=first_block_width,
                kernel_size_list=3,
                expand_ratio_list=1,
                stride=1,
                activation="relu6",
                dw_layer=None,
            )
        first_block = DynamicInvertedResidual(first_block_conv, None)

        input_channel = first_block_width

        # inverted residual blocks
        self.block_group_info = []
        blocks = [first_block]
        _block_index = 1

        stride_stages = [2, 2, 2, 1, 2, 1]
        if depth_list is None:
            n_block_list = [2, 3, 4, 3, 3, 1]
            self.depth_list = [4, 4]
            print("Use MobileNetV2 Depth Setting")
        else:
            n_block_list = [
                max(self.depth_list) + self.base_depth[i] for i in range(5)
            ] + [1]

        width_list = []
        for base_width in base_stage_width[2:-1]:
            width = [
                make_divisible(base_width * width_mult, 8)
                for width_mult in self.width_mult_list
            ]
            width_list.append(width)

        if se_stages is None:
            se_stages = [False] * (len(base_stage_width) - 3)
            assert len(se_stages) == len(width_list)

        for width, n_block, s, use_se in zip(
            width_list, n_block_list, stride_stages, se_stages
        ):
            self.block_group_info.append([_block_index + i for i in range(n_block)])
            _block_index += n_block

            output_channel = width
            for i in range(n_block):
                if i == 0:
                    stride = s
                else:
                    stride = 1

                mobile_inverted_conv = DynamicInvertedConv2d(
                    in_channel_list=val2list(input_channel, 1),
                    out_channel_list=val2list(output_channel, 1),
                    kernel_size_list=ks_list,
                    expand_ratio_list=expand_ratio_list,
                    stride=stride,
                    act_func="relu6",
                    use_se=use_se[i] if isinstance(use_se, list) else use_se,
                )

                if stride == 1 and input_channel == output_channel:
                    shortcut = IdentityLayer(input_channel, input_channel)
                else:
                    shortcut = None

                mb_inverted_block = DynamicInvertedResidual(
                    mobile_inverted_conv, shortcut
                )

                blocks.append(mb_inverted_block)
                input_channel = output_channel

        super(OFAMCUNets, self).__init__(
            first_conv, blocks, feature_mix_layer, classifier
        )

        # set bn param
        self.set_bn_param(momentum=bn_param[0], eps=bn_param[1])

        # runtime_depth
        self.runtime_depth = [len(block_idx) for block_idx in self.block_group_info]

    def forward(self, x):
        # first conv
        x = self.first_conv(x)
        # first block
        x = self.blocks[0](x)

        # blocks
        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                x = self.blocks[idx](x)

        # feature_mix_layer
        if self.feature_mix_layer is not None:
            x = self.feature_mix_layer(x)
        x = x.mean(3).mean(2)

        x = self.classifier(x)
        return x

    @property
    def module_str(self):
        _str = self.first_conv.module_str + "\n"
        _str += self.blocks[0].module_str + "\n"

        for stage_id, block_idx in enumerate(self.block_group_info):
            depth = self.runtime_depth[stage_id]
            active_idx = block_idx[:depth]
            for idx in active_idx:
                _str += self.blocks[idx].module_str + "\n"
        if self.feature_mix_layer is not None:
            _str += self.feature_mix_layer.module_str + "\n"
        _str += self.classifier.module_str + "\n"
        return _str

    @property
    def config(self):
        return {
            "name": OFAMCUNets.__name__,
            "bn": self.get_bn_param(),
            "first_conv": self.first_conv.config,
            "blocks": [block.config for block in self.blocks],
            "feature_mix_layer": (
                None
                if self.feature_mix_layer is None
                else self.feature_mix_layer.config
            ),
            "classifier": self.classifier.config,
        }

    """ Width Related Methods """

    def re_organize_middle_weights(self, expand_ratio_stage=0):
        if len(self.width_mult_list) > 1:
            print(
                " * WARNING: sorting is not implemented right for multiple width-mult"
            )

        for block in self.blocks[1:]:
            block.mobile_inverted_conv.re_organize_middle_weights(expand_ratio_stage)
