import numpy as np

import mmcv

# class Resize(object):
#     """Resize images & seg.
#
#     This transform resizes the input image to some scale. If the input dict
#     contains the key "scale", then the scale in the input dict is used,
#     otherwise the specified scale in the init method is used.
#
#     ``img_scale`` can be Nong, a tuple (single-scale) or a list of tuple
#     (multi-scale). There are 4 multiscale modes:
#
#     - ``ratio_range is not None``:
#     1. When img_scale is None, img_scale is the shape of image in results
#         (img_scale = results['img'].shape[:2]) and the image is resized based
#         on the original size. (mode 1)
#     2. When img_scale is a tuple (single-scale), randomly sample a ratio from
#         the ratio range and multiply it with the image scale. (mode 2)
#
#     - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
#     scale from the a range. (mode 3)
#
#     - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
#     scale from multiple scales. (mode 4)
#
#     Args:
#         img_scale (tuple or list[tuple]): Images scales for resizing.
#         multiscale_mode (str): Either "range" or "value".
#         ratio_range (tuple[float]): (min_ratio, max_ratio)
#         keep_ratio (bool): Whether to keep the aspect ratio when resizing the
#             image.
#     """
#
#     def __init__(self,
#                  img_scale=(2048, 1024),
#                  multiscale_mode='range',
#                  ratio_range=(0.5, 2.0),
#                  keep_ratio=True):
#         self.img_scale = [img_scale]
#         self.multiscale_mode = multiscale_mode
#         self.ratio_range = ratio_range
#         self.keep_ratio = keep_ratio
#
#     @staticmethod
#     def random_select(img_scales):
#         """Randomly select an img_scale from given candidates.
#
#         Args:
#             img_scales (list[tuple]): Images scales for selection.
#
#         Returns:
#             (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
#                 where ``img_scale`` is the selected image scale and
#                 ``scale_idx`` is the selected index in the given candidates.
#         """
#
#         assert mmcv.is_list_of(img_scales, tuple)
#         scale_idx = np.random.randint(len(img_scales))
#         img_scale = img_scales[scale_idx]
#         return img_scale, scale_idx
#
#     @staticmethod
#     def random_sample(img_scales):
#         """Randomly sample an img_scale when ``multiscale_mode=='range'``.
#
#         Args:
#             img_scales (list[tuple]): Images scale range for sampling.
#                 There must be two tuples in img_scales, which specify the lower
#                 and uper bound of image scales.
#
#         Returns:
#             (tuple, None): Returns a tuple ``(img_scale, None)``, where
#                 ``img_scale`` is sampled scale and None is just a placeholder
#                 to be consistent with :func:`random_select`.
#         """
#
#         assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
#         img_scale_long = [max(s) for s in img_scales]
#         img_scale_short = [min(s) for s in img_scales]
#         long_edge = np.random.randint(
#             min(img_scale_long),
#             max(img_scale_long) + 1)
#         short_edge = np.random.randint(
#             min(img_scale_short),
#             max(img_scale_short) + 1)
#         img_scale = (long_edge, short_edge)
#         return img_scale, None
#
#     @staticmethod
#     def random_sample_ratio(img_scale, ratio_range):
#         """Randomly sample an img_scale when ``ratio_range`` is specified.
#
#         A ratio will be randomly sampled from the range specified by
#         ``ratio_range``. Then it would be multiplied with ``img_scale`` to
#         generate sampled scale.
#
#         Args:
#             img_scale (tuple): Images scale base to multiply with ratio.
#             ratio_range (tuple[float]): The minimum and maximum ratio to scale
#                 the ``img_scale``.
#
#         Returns:
#             (tuple, None): Returns a tuple ``(scale, None)``, where
#                 ``scale`` is sampled ratio multiplied with ``img_scale`` and
#                 None is just a placeholder to be consistent with
#                 :func:`random_select`.
#         """
#
#         min_ratio, max_ratio = ratio_range
#         ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
#         scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
#         return scale
#
#     def _random_scale(self, results):
#         """Randomly sample an img_scale according to ``ratio_range`` and
#         ``multiscale_mode``.
#
#         If ``ratio_range`` is specified, a ratio will be sampled and be
#         multiplied with ``img_scale``.
#         If multiple scales are specified by ``img_scale``, a scale will be
#         sampled according to ``multiscale_mode``.
#         Otherwise, single scale will be used.
#
#         Args:
#             results (dict): Result dict from :obj:`dataset`.
#
#         Returns:
#             dict: Two new keys 'scale` and 'scale_idx` are added into
#                 ``results``, which would be used by subsequent pipelines.
#         """
#
#         scale = self.random_sample_ratio(
#                 self.img_scale[0], self.ratio_range)
#
#         results['scale'] = scale
#
#     def _resize_img(self, results):
#         """Resize images with ``results['scale']``."""
#         if self.keep_ratio:
#             img, scale_factor = mmcv.imrescale(
#                 results['img'], results['scale'], return_scale=True)
#             # the w_scale and h_scale has minor difference
#             # a real fix should be done in the mmcv.imrescale in the future
#             new_h, new_w = img.shape[:2]
#             h, w = results['img'].shape[:2]
#             w_scale = new_w / w
#             h_scale = new_h / h
#         else:
#             img, w_scale, h_scale = mmcv.imresize(
#                 results['img'], results['scale'], return_scale=True)
#         scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
#                                 dtype=np.float32)
#         results['img'] = img
#         results['img_shape'] = img.shape
#         results['pad_shape'] = img.shape  # in case that there is no padding
#         results['scale_factor'] = scale_factor
#         results['keep_ratio'] = self.keep_ratio
#
#     def _resize_seg(self, results):
#         """Resize semantic segmentation map with ``results['scale']``."""
#         for key in results.get('seg_fields', []):
#             if self.keep_ratio:
#                 gt_seg = mmcv.imrescale(
#                     results[key], results['scale'], interpolation='nearest')
#             else:
#                 gt_seg = mmcv.imresize(
#                     results[key], results['scale'], interpolation='nearest')
#             results[key] = gt_seg
#
#     def __call__(self, results):
#         """Call function to resize images, bounding boxes, masks, semantic
#         segmentation map.
#
#         Args:
#             results (dict): Result dict from loading pipeline.
#
#         Returns:
#             dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
#                 'keep_ratio' keys are added into result dict.
#         """
#
#         self._random_scale(results)
#         self._resize_img(results)
#         self._resize_seg(results)
#         return results
#
#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         repr_str += (f'(img_scale={self.img_scale}, '
#                      f'multiscale_mode={self.multiscale_mode}, '
#                      f'ratio_range={self.ratio_range}, '
#                      f'keep_ratio={self.keep_ratio})')
#         return repr_str



class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        return img

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        img = results['img']
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str