from hanser.models.detection.retinanet import RetinaNet, RetinaNetBiFPN


class ATSS(RetinaNet):

    def __init__(self, backbone, num_classes, backbone_indices=(1, 2, 3), feat_channels=256,
                 stacked_convs=4, norm='bn', num_extra_convs=2, centerness=True):
        super().__init__(backbone, 1, num_classes, backbone_indices,
                         feat_channels, extra_convs_on='output', num_extra_convs=num_extra_convs,
                         stacked_convs=stacked_convs, norm=norm, centerness=centerness)


class ATSSBiFPN(RetinaNetBiFPN):

    def __init__(self, backbone, num_classes, backbone_indices=(1, 2, 3),
                 feat_channels=160, fpn_repeats=6, num_extra_levels=2, seperable_conv=False,
                 fpn_act='def', stacked_convs=4, norm='bn', centerness=True):
        super().__init__(backbone, 1, num_classes, backbone_indices,
                         feat_channels, fpn_repeats, num_extra_levels, seperable_conv,
                         fpn_act, stacked_convs=stacked_convs, norm=norm, centerness=centerness)