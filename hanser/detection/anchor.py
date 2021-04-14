from typing import List

import numpy as np
import tensorflow as tf
from hanser.ops import _pair, _meshgrid


class AnchorGenerator(object):
    """Standard anchor generator for 2D anchor-based detectors.

    Args:
        strides (List[int]): Strides of anchors
            in multiple feature levels in order (w, h).
        ratios (List[float]): The list of ratios between the height and width
            of anchors in a single level.
        scales (List[float] | None): Anchor scales for anchors in a single level.
            It cannot be set at the same time if `octave_base_scale` and
            `scales_per_octave` are set.
        scale_major (bool): Whether to multiply scales first when generating
            base anchors. If true, the anchors in the same row will have the
            same scales. By default it is True in V2.0
        octave_base_scale (int): The base scale of octave.
        scales_per_octave (int): Number of scales for each octave.
            `octave_base_scale` and `scales_per_octave` are usually used in
            retinanet and the `scales` should be None when they are set.
        center_offset (float): The offset of center in proportion to anchors'
            width and height. By default it is 0 in V2.0.

    Examples:
        >>> from hanser.detection.anchor import AnchorGenerator
        >>> self = AnchorGenerator([16], [1.], [1.])
        >>> all_anchors = self.grid_anchors([(2, 2)], device='cpu')
        >>> print(all_anchors)
        [tensor([[-4.5000, -4.5000,  4.5000,  4.5000],
                [11.5000, -4.5000, 20.5000,  4.5000],
                [-4.5000, 11.5000,  4.5000, 20.5000],
                [11.5000, 11.5000, 20.5000, 20.5000]])]
        >>> self = AnchorGenerator([16, 32], [1.], [1.])
        >>> all_anchors = self.grid_anchors([(2, 2), (1, 1)], device='cpu')
        >>> print(all_anchors)
        [tensor([[-4.5000, -4.5000,  4.5000,  4.5000],
                [11.5000, -4.5000, 20.5000,  4.5000],
                [-4.5000, 11.5000,  4.5000, 20.5000],
                [11.5000, 11.5000, 20.5000, 20.5000]]), \
        tensor([[-9., -9., 9., 9.]])]
    """

    # noinspection PyTypeChecker
    def __init__(self,
                 strides,
                 ratios,
                 scales=None,
                 scale_major=True,
                 octave_base_scale=None,
                 scales_per_octave=None,
                 center_offset=0.5):

        self.strides = [_pair(stride) for stride in strides]
        self.base_sizes = [min(stride) for stride in self.strides]

        # calculate scales of anchors
        assert ((octave_base_scale is not None
                and scales_per_octave is not None) ^ (scales is not None)), \
            'scales and octave_base_scale with scales_per_octave cannot' \
            ' be set at the same time'
        if scales is not None:
            self.scales = np.array(scales, dtype=np.float32)
        elif octave_base_scale is not None and scales_per_octave is not None:
            octave_scales = np.array(
                [2**(i / scales_per_octave) for i in range(scales_per_octave)])
            scales = octave_scales * octave_base_scale
            self.scales = scales
        else:
            raise ValueError('Either scales or octave_base_scale with '
                             'scales_per_octave should be set')

        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.ratios = np.array(ratios, dtype=np.float32)
        self.scale_major = scale_major
        self.center_offset = center_offset
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        """list[int]: total number of base anchors in a feature grid"""
        return [base_anchors.shape[0] for base_anchors in self.base_anchors]

    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)

    def gen_base_anchors(self):
        """Generate base anchors.

        Returns:
            List[tf.Tensor]: Base anchors of a feature grid in multiple \
                feature levels.
        """
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            base_anchors = self.gen_single_level_base_anchors(
                base_size,
                scales=self.scales,
                ratios=self.ratios,
            )
            base_anchors = tf.constant(base_anchors, dtype=tf.float32)
            multi_level_base_anchors.append(base_anchors)
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self,
                                      base_size,
                                      scales,
                                      ratios,
                                      center=None):
        """Generate base anchors of a single level.

        Args:
            base_size (int | float): Basic size of an anchor.
            scales: Scales of the anchor.
            ratios: The ratio between between the height
                and width of anchors in a single level.
            center (tuple[float], optional): The center of the base anchor
                related to a single feature grid. Defaults to None.

        Returns:
            torch.Tensor: Anchors in a single-level feature maps.
        """
        h = w = base_size
        if center is None:
            y_center = self.center_offset * h
            x_center = self.center_offset * w
        else:
            y_center, x_center = center

        h_ratios = np.sqrt(ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            hs = (h * h_ratios[:, None] * scales[None, :]).reshape(-1)
            ws = (w * w_ratios[:, None] * scales[None, :]).reshape(-1)
        else:
            hs = (h * scales[:, None] * h_ratios[None, :]).reshape(-1)
            ws = (w * scales[:, None] * w_ratios[None, :]).reshape(-1)

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            y_center - 0.5 * hs, x_center - 0.5 * ws, y_center + 0.5 * hs, x_center + 0.5 * ws]
        base_anchors = np.stack(base_anchors, axis=-1)
        return base_anchors


    def grid_anchors(self, featmap_sizes):
        """Generate grid anchors in multiple feature levels.

        Args:
            featmap_sizes (list[tuple|list]): List of feature map sizes in
                multiple feature levels.

        Return:
            list[tf.Tensor]: Anchors in multiple feature levels. \
                The sizes of each tensor should be [N, 4], where \
                N = width * height * num_base_anchors, width and height \
                are the sizes of the corresponding feature level, \
                num_base_anchors is the number of anchors for that level.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_anchors(
                self.base_anchors[i],
                featmap_sizes[i],
                self.strides[i])
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_anchors(self,
                                  base_anchors,
                                  featmap_size,
                                  stride=(16, 16)):
        """Generate grid anchors of a single level.

        Note:
            This function is usually called by method ``self.grid_anchors``.

        Args:
            base_anchors (tf.Tensor): The base anchors of a feature grid.
            featmap_size (tf.Tensor|tuple[int]): Size of the feature maps.
            stride (tuple[int], optional): Stride of the feature map in order
                (w, h). Defaults to (16, 16).

        Returns:
            tf.Tensor: Anchors in the overall feature maps.
        """
        feat_h, feat_w = featmap_size[0], featmap_size[1]
        shift_y = tf.range(0, feat_h) * stride[0]
        shift_x = tf.range(0, feat_w) * stride[1]

        shift_yy, shift_xx = _meshgrid(shift_y, shift_x, row_major=False)
        shifts = tf.stack([shift_yy, shift_xx, shift_yy, shift_xx], axis=-1)
        shifts = tf.cast(shifts, base_anchors.dtype)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)
        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = tf.reshape(all_anchors, [-1, 4])
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def __repr__(self):
        """str: a string that describes the module"""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}strides={self.strides},\n'
        repr_str += f'{indent_str}ratios={self.ratios},\n'
        repr_str += f'{indent_str}scales={self.scales},\n'
        repr_str += f'{indent_str}scale_major={self.scale_major},\n'
        repr_str += f'{indent_str}octave_base_scale='
        repr_str += f'{self.octave_base_scale},\n'
        repr_str += f'{indent_str}scales_per_octave='
        repr_str += f'{self.scales_per_octave},\n'
        repr_str += f'{indent_str}num_levels={self.num_levels}\n'
        repr_str += f'{indent_str}center_offset={self.center_offset})'
        return repr_str


class SSDAnchorGenerator(AnchorGenerator):
    """Anchor generator for SSD.

    Args:
        strides (list[int]  | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels.
        ratios (list[list[float]]): The list of ratios between the height and width
            of anchors in a single level.
        basesize_ratio_range (tuple(float)): Ratio range of anchors.
        input_size (int): Size of feature map, 300 for SSD300,
            512 for SSD512.
        scale_major (bool): Whether to multiply scales first when generating
            base anchors. If true, the anchors in the same row will have the
            same scales. It is always set to be False in SSD.
    """

    def __init__(self,
                 strides,
                 ratios,
                 basesize_ratio_range,
                 extra_min_ratio=0.07,
                 input_size=300,
                 scale_major=False):
        assert len(strides) == len(ratios)

        self.strides = [_pair(stride) for stride in strides]
        self.input_size = input_size
        self.centers = [(stride[0] / 2., stride[1] / 2.)
                        for stride in self.strides]
        self.basesize_ratio_range = basesize_ratio_range

        # calculate anchor ratios and sizes
        min_ratio, max_ratio = basesize_ratio_range
        min_ratio = int(min_ratio * 100)
        max_ratio = int(max_ratio * 100)
        step = int(np.floor(max_ratio - min_ratio) / (self.num_levels - 2))
        sizes = []
        for ratio in range(min_ratio, max_ratio + step + 1, step):
            sizes.append(int(self.input_size * ratio / 100))
        sizes.insert(0, int(self.input_size * extra_min_ratio))
        min_sizes, max_sizes = sizes[:-1], sizes[1:]

        anchor_ratios = []
        anchor_scales = []
        for k in range(len(self.strides)):
            scales = [1., np.sqrt(max_sizes[k] / min_sizes[k])]
            anchor_ratio = [1.]
            for r in ratios[k]:
                anchor_ratio += [1 / r, r]  # 4 or 6 ratio
            anchor_ratios.append(np.array(anchor_ratio))
            anchor_scales.append(np.array(scales))

        self.base_sizes = min_sizes
        self.scales = anchor_scales
        self.ratios = anchor_ratios
        self.scale_major = scale_major
        self.center_offset = 0
        self.base_anchors = self.gen_base_anchors()

    def gen_base_anchors(self):
        """Generate base anchors.

        Returns:
            list(torch.Tensor): Base anchors of a feature grid in multiple \
                feature levels.
        """
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            base_anchors = self.gen_single_level_base_anchors(
                base_size,
                scales=self.scales[i],
                ratios=self.ratios[i],
                center=self.centers[i])
            indices = list(range(len(self.ratios[i])))
            indices.insert(1, len(indices))
            base_anchors = base_anchors[indices]
            multi_level_base_anchors.append(tf.constant(base_anchors, dtype=tf.float32))
        return multi_level_base_anchors

    def __repr__(self):
        """str: a string that describes the module"""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}strides={self.strides},\n'
        repr_str += f'{indent_str}scale_major={self.scale_major},\n'
        repr_str += f'{indent_str}input_size={self.input_size},\n'
        repr_str += f'{indent_str}scales={self.scales},\n'
        repr_str += f'{indent_str}ratios={self.ratios},\n'
        repr_str += f'{indent_str}num_levels={self.num_levels},\n'
        repr_str += f'{indent_str}base_sizes={self.base_sizes},\n'
        repr_str += f'{indent_str}basesize_ratio_range='
        repr_str += f'{self.basesize_ratio_range})'
        return repr_str
