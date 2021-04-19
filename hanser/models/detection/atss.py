from hanser.models.detection.retinanet import RetinaNet


class ATSS(RetinaNet):

    def __init__(self, backbone, num_classes, backbone_indices=(1, 2, 3), feat_channels=256,
                 stacked_convs=4, norm='bn', num_extra_convs=2):
        super().__init__(backbone, 1, num_classes, backbone_indices,
                         feat_channels, extra_convs_on='output', num_extra_convs=num_extra_convs,
                         stacked_convs=stacked_convs, norm=norm, centerness=True)
