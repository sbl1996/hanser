# import tensorflow as tf
# from tensorflow.keras.layers import Layer
#
# class FPN(Layer):
#
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  num_outs):
#         super(FPN, self).__init__()
#         assert isinstance(in_channels, list)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.num_ins = len(in_channels)
#         self.num_outs = num_outs
#
#         if end_level == -1:
#             self.backbone_end_level = self.num_ins
#             assert num_outs >= self.num_ins - start_level
#         else:
#             # if end_level < inputs, no extra level is allowed
#             self.backbone_end_level = end_level
#             assert end_level <= len(in_channels)
#             assert num_outs == end_level - start_level
#         self.start_level = start_level
#         self.end_level = end_level
#         self.lateral_convs = nn.ModuleList()
#
#         for i in range(self.start_level, self.backbone_end_level):
#             l_conv = ConvModule(
#                 in_channels[i],
#                 out_channels,
#                 1,
#                 conv_cfg=conv_cfg,
#                 norm_cfg=norm_cfg,
#                 activation=activation,
#                 inplace=False)
#
#             self.lateral_convs.append(l_conv)
#         self.init_weights()
#
#     def forward(self, inputs):
#         assert len(inputs) == len(self.in_channels)
#
#         # build laterals
#         laterals = [
#             lateral_conv(inputs[i + self.start_level])
#             for i, lateral_conv in enumerate(self.lateral_convs)
#         ]
#
#         # build top-down path
#         used_backbone_levels = len(laterals)
#         for i in range(used_backbone_levels - 1, 0, -1):
#             laterals[i - 1] += F.interpolate(
#                 laterals[i], scale_factor=2, mode='bilinear')
#
#         # build outputs
#         outs = [
#             # self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
#             laterals[i] for i in range(used_backbone_levels)
#         ]
#         return tuple(outs)
#
#
# class PAN(FPN):
#     """Path Aggregation Network for Instance Segmentation.
#     This is an implementation of the `PAN in Path Aggregation Network
#     <https://arxiv.org/abs/1803.01534>`_.
#     Args:
#         in_channels (List[int]): Number of input channels per scale.
#         out_channels (int): Number of output channels (used at each scale)
#         num_outs (int): Number of output scales.
#         start_level (int): Index of the start input backbone level used to
#             build the feature pyramid. Default: 0.
#         end_level (int): Index of the end input backbone level (exclusive) to
#             build the feature pyramid. Default: -1, which means the last level.
#         add_extra_convs (bool): Whether to add conv layers on top of the
#             original feature maps. Default: False.
#         extra_convs_on_inputs (bool): Whether to apply extra conv on
#             the original feature from the backbone. Default: False.
#         relu_before_extra_convs (bool): Whether to apply relu before the extra
#             conv. Default: False.
#         no_norm_on_lateral (bool): Whether to apply norm on lateral.
#             Default: False.
#         conv_cfg (dict): Config dict for convolution layer. Default: None.
#         norm_cfg (dict): Config dict for normalization layer. Default: None.
#         act_cfg (str): Config dict for activation layer in ConvModule.
#             Default: None.
#     """
#
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  num_outs,
#                  start_level=0,
#                  end_level=-1,
#                  conv_cfg=None,
#                  norm_cfg=None,
#                  activation=None):
#         super(PAN,
#               self).__init__(in_channels, out_channels, num_outs, start_level,
#                              end_level, conv_cfg, norm_cfg, activation)
#         self.init_weights()
#
#     def forward(self, inputs):
#         """Forward function."""
#         assert len(inputs) == len(self.in_channels)
#
#         # build laterals
#         laterals = [
#             lateral_conv(inputs[i + self.start_level])
#             for i, lateral_conv in enumerate(self.lateral_convs)
#         ]
#
#         # build top-down path
#         used_backbone_levels = len(laterals)
#         for i in range(used_backbone_levels - 1, 0, -1):
#             laterals[i - 1] += F.interpolate(
#                 laterals[i], scale_factor=2, mode='bilinear')
#
#         # build outputs
#         # part 1: from original levels
#         inter_outs = [
#             laterals[i] for i in range(used_backbone_levels)
#         ]
#
#         # part 2: add bottom-up path
#         for i in range(0, used_backbone_levels - 1):
#             inter_outs[i + 1] += F.interpolate(inter_outs[i], scale_factor=0.5, mode='bilinear')
#
#         outs = []
#         outs.append(inter_outs[0])
#         outs.extend([
#             inter_outs[i] for i in range(1, used_backbone_levels)
#         ])
#         return tuple(outs)