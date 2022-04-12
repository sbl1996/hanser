import math

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomNormal

from hanser.models.layers import Conv2d, Identity
from hanser.models.attention import SELayer
from hanser.models.modules import GlobalAvgPool, Dropout, DropPath
from hanser.models.utils import init_layer_ascending_drop_path

width_divisor = 8
less_se_channels = True
zero_last_bn_gamma = True


def py2_round(x):
    return math.floor(x + 0.5) if x >= 0.0 else math.ceil(x - 0.5)


def get_divisible_by(num, divisible_by=8, min_val=None):
    ret = int(num)
    if min_val is None:
        min_val = divisible_by
    if divisible_by > 0 and num % divisible_by != 0:
        ret = int((py2_round(num / divisible_by) or 1) * divisible_by)
        if ret < 0.95 * num:
            ret += divisible_by
    if ret < min_val:
        ret = min_val
    return ret


class InvertedResidual(Layer):
    def __init__(self, in_channels, channels, out_channels, kernel_size, stride, act='relu', with_se=True, drop_path=0.0):
        super().__init__()
        self.with_se = with_se
        self.use_res_connect = stride == 1 and in_channels == out_channels

        if in_channels != channels:
            self.expand = Conv2d(in_channels, channels, kernel_size=1,
                                 norm='bn', act=act)
        else:
            self.expand = Identity()

        self.dwconv = Conv2d(channels, channels, kernel_size, stride, groups=channels,
                             norm='bn', act=act)

        if self.with_se:
            se_channels = int(in_channels // 4) if less_se_channels else int(channels // 4)
            if less_se_channels:
                self.se = SELayer(channels, se_channels=se_channels, act=act, gating_fn='hsigmoid',
                                  min_se_channels=8, divisible=8)

        self.project = Conv2d(channels, out_channels, kernel_size=1,
                              norm='bn', gamma_init='zeros' if zero_last_bn_gamma and self.use_res_connect else 'ones')
        self.drop_path = DropPath(drop_path) if drop_path and self.use_res_connect else Identity()

    def call(self, x):
        identity = x
        x = self.expand(x)
        x = self.dwconv(x)
        if self.with_se:
            x = self.se(x)
        x = self.project(x)
        if self.use_res_connect:
            x = self.drop_path(x)
            x += identity
        return x


class FBNetV3(Model):

    def __init__(self, setting, num_classes=1000, dropout=0, drop_path=0):
        super().__init__()
        in_channels = setting['init_channels']
        last_channels = setting['last_channels']

        # Original code has bias=True
        self.stem = Conv2d(3, in_channels, kernel_size=3, stride=2,
                           norm='bn', act='hswish')

        for i, stage_setting in enumerate(setting['stages']):
            stage = []
            for k, c, s, n, e, se, nl in stage_setting:
                mid_channels = get_divisible_by(in_channels * e, 8)
                out_channels = c
                if k == 1:
                    stage.append(Conv2d(
                        in_channels, out_channels, kernel_size=1, stride=s, norm='bn', act=nl))
                else:
                    stage.append(InvertedResidual(
                        in_channels, mid_channels, out_channels, k, s, nl, se, drop_path))
                    in_channels = out_channels

                    mid_channels = get_divisible_by(in_channels * e, 8)
                    for j in range(n-1):
                        stage.append(InvertedResidual(
                            in_channels, mid_channels, out_channels, k, 1, nl, se))
                name = f"stage{i+1}"
                setattr(self, name, Sequential(stage))

        if drop_path:
            init_layer_ascending_drop_path(self, drop_path)

        self.last_pw = Conv2d(in_channels, in_channels * 6, kernel_size=1,
                              norm='bn', act='hswish')
        self.avgpool = GlobalAvgPool(keep_dim=True)
        # Original code has bias=False
        self.last_fc = Conv2d(in_channels * 6, last_channels, kernel_size=1, act='hswish')
        self.dropout = Dropout(dropout) if dropout else None
        self.fc = Conv2d(last_channels, num_classes, kernel_size=1,
                         kernel_init=RandomNormal(stddev=0.01), bias_init='zeros')

    def call(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)

        x = self.last_pw(x)
        x = self.avgpool(x)
        x = self.last_fc(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        x = tf.squeeze(x, axis=(1, 2))
        return x


def convert_def(d):
    setting = {
        "input_size": d['input_size'],
        "init_channels": d['blocks'][0][0][1],
    }
    i = setting['init_channels']
    stages = []
    for s in d['blocks'][1:-1]:
        stage = []
        for b in s:
            c, s, n, e = b[1:5]
            if b[0] == 'skip':
                if i == c:
                    continue
                else:
                    k = 1
            else:
                k = int(b[0][4])
            se = 'sehsig' in b[0]
            nl = 'hswish' if b[0].endswith('hs') else 'relu'
            stage.append([k, c, s, n, e, se, nl])
            i = c
        stages.append(stage)
    setting['stages'] = stages
    setting['last_channels'] = d['blocks'][-1][0][1]
    return setting


def fbnet_v3_a(**kwargs):
    return FBNetV3(convert_def(FBNetV3_A_def), **kwargs)


def fbnet_v3_c(**kwargs):
    return FBNetV3(convert_def(FBNetV3_C_def), **kwargs)


def fbnet_v3_e(**kwargs):
    return FBNetV3(convert_def(FBNetV3_E_def), **kwargs)


# From https://github.com/facebookresearch/mobile-vision/blob/main/mobile_cv/arch/fbnet_v2/fbnet_modeldef_cls_fbnetv3.py

FBNetV3_A_def = {
    "input_size": 224,
    "blocks": [
        [["conv_k3_hs", 16, 2, 1]],
        [["ir_k3_hs", 16, 1, 2, 1]],
        [
            ["ir_k5_hs", 24, 2, 1, 4],
            ["ir_k5_hs", 24, 1, 3, 2],
        ],
        [
            ["ir_k5_sehsig_hs", 40, 2, 1, 5],
            ["ir_k5_sehsig_hs", 40, 1, 4, 3],
        ],
        [
            ["ir_k5_hs", 72, 2, 1, 5], # p1
            ["ir_k3_hs", 72, 1, 4, 3],
            ["ir_k3_sehsig_hs", 120, 1, 1, 5],
            ["ir_k5_sehsig_hs", 120, 1, 5, 3],
        ],
        [
            ["ir_k3_sehsig_hs", 184, 2, 1, 6], # p0
            ["ir_k5_sehsig_hs", 184, 1, 5, 4],
            ["ir_k5_sehsig_hs", 224, 1, 1, 6],
        ],
        [["ir_pool_hs", 1984, 1, 6]],
    ],
}

FBNetV3_B_def = {
    "input_size": 248,
    "blocks": [
        [["conv_k3_hs", 16, 2, 1]],
        [["ir_k3_hs", 16, 1, 2, 1]],
        [
            ["ir_k5_hs", 24, 2, 1, 4],
            ["ir_k5_hs", 24, 1, 3, 2],
        ],
        [
            ["ir_k5_sehsig_hs", 40, 2, 1, 5],
            ["ir_k5_sehsig_hs", 40, 1, 4, 3],
        ],
        [
            ["ir_k5_hs", 72, 2, 1, 5], # p1
            ["ir_k3_hs", 72, 1, 4, 3],
            ["ir_k3_sehsig_hs", 120, 1, 1, 5],
            ["ir_k5_sehsig_hs", 120, 1, 5, 3],
        ],
        [
            ["ir_k3_sehsig_hs", 184, 2, 1, 6], # p0
            ["ir_k5_sehsig_hs", 184, 1, 5, 4],
            ["ir_k5_sehsig_hs", 224, 1, 1, 6],
        ],
        [["ir_pool_hs", 1984, 1, 1, 6]],
    ],
}

FBNetV3_C_def = {
    "input_size": 248,
    "blocks": [
        [["conv_k3_hs", 16, 2, 1]],
        [["ir_k3_hs", 16, 1, 2, 1]],
        [
            ["ir_k5_hs", 24, 2, 1, 5],
            ["ir_k3_hs", 24, 1, 4, 3],
        ],
        [
            ["ir_k5_sehsig_hs", 48, 2, 1, 5],
            ["ir_k5_sehsig_hs", 48, 1, 4, 2],
        ],
        [
            ["ir_k5_hs", 88, 2, 1, 4], # p1
            ["ir_k3_hs", 88, 1, 4, 3],
            ["ir_k3_sehsig_hs", 120, 1, 1, 4],
            ["ir_k5_sehsig_hs", 120, 1, 5, 3],
        ],
        [
            ["ir_k5_sehsig_hs", 216, 2, 1, 5], # p1
            ["ir_k5_sehsig_hs", 216, 1, 5, 5],
            ["ir_k5_sehsig_hs", 216, 1, 1, 6],
        ],
        [["ir_pool_hs", 1984, 1, 1, 6]],
    ],
}

FBNetV3_D_def = {
    "input_size": 248,
    "blocks": [
        [["conv_k3_hs", 24, 2, 1]],
        [["ir_k3_hs", 16, 1, 2, 1]],
        [
            ["ir_k3_hs", 24, 2, 1, 5],
            ["ir_k3_hs", 24, 1, 5, 2],
        ],
        [
            ["ir_k5_sehsig_hs", 40, 2, 1, 4],
            ["ir_k3_sehsig_hs", 40, 1, 4, 3],
        ],
        [
            ["ir_k3_hs", 72, 2, 1, 5],
            ["ir_k3_hs", 72, 1, 4, 3],
            ["ir_k3_sehsig_hs", 128, 1, 1, 5],
            ["ir_k5_sehsig_hs", 128, 1, 6, 3],
        ],
        [
            ["ir_k3_sehsig_hs", 208, 2, 1, 6],
            ["ir_k5_sehsig_hs", 208, 1, 5, 5],
            ["ir_k5_sehsig_hs", 240, 1, 1, 6],
        ],
        [["ir_pool_hs", 1984, 1, 1, 6]],
    ],
}


FBNetV3_E_def = {
    "input_size": 264,
    "blocks": [
        [["conv_k3_hs", 24, 2, 1]],
        [["ir_k3_hs", 16, 1, 3, 1]],
        [
            ["ir_k5_hs", 24, 2, 1, 4],
            ["ir_k5_hs", 24, 1, 4, 2],
        ],
        [
            ["ir_k5_sehsig_hs", 48, 2, 1, 4],
            ["ir_k5_sehsig_hs", 48, 1, 4, 3],
        ],
        [
            ["ir_k5_hs", 80, 2, 1, 5], # p1
            ["ir_k3_hs", 80, 1, 4, 3],
            ["ir_k3_sehsig_hs", 128, 1, 1, 5],
            ["ir_k5_sehsig_hs", 128, 1, 7, 3],
        ],
        [
            ["ir_k3_sehsig_hs", 216, 2, 1, 6],
            ["ir_k5_sehsig_hs", 216, 1, 5, 5],
            ["ir_k5_sehsig_hs", 240, 1, 1, 6],
        ],
        [["ir_pool_hs", 1984, 1, 1, 6]],
    ],
}


FBNetV3_F_def = {
    "input_size": 272,
    "blocks": [
        [["conv_k3_hs", 24, 2, 1]],
        [["ir_k3_hs", 24, 1, 3, 1]],
        [
            ["ir_k5_hs", 32, 2, 1, 4],
            ["ir_k5_hs", 32, 1, 4, 2],
        ],
        [
            ["ir_k5_sehsig_hs", 56, 2, 1, 4],
            ["ir_k5_sehsig_hs", 56, 1, 4, 3],
        ],
        [
            ["ir_k5_hs", 88, 2, 1, 5],
            ["ir_k3_hs", 88, 1, 4, 3],
            ["ir_k3_sehsig_hs", 144, 1, 1, 5],
            ["ir_k5_sehsig_hs", 144, 1, 8, 3],
        ],
        [
            ["ir_k3_sehsig_hs", 248, 2, 1, 6], # p0
            ["ir_k5_sehsig_hs", 248, 1, 6, 5],
            ["ir_k5_sehsig_hs", 272, 1, 1, 6],
        ],
        [["ir_pool_hs", 1984, 1, 1, 6]],
    ],
}