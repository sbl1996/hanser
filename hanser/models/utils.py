import re
import tensorflow as tf

from hanser.models.hub import load_model_from_hub


def load_checkpoint(ckpt_path, **ckpt_kwargs):
    r"""Load a checkpoint, either on gcs or local filesystems.

    Args:
        ckpt_path: The path to the checkpoint as returned by `ckpt.write`.
        **ckpt_kwargs: Keyword arguments passed to tf.train.Checkpoint.

    Returns:
        A load status object.  See `ckpt.restore` for details.

    Examples:
        >>> from hanser.models.backbone.resnet_vd import resnet50
        >>> from hanser.models.detection.retinanet import RetinaNet
        >>> backbone = resnet50()
        >>> model = RetinaNet(backbone, num_anchors=9, num_classes=80)
        >>> model.build((None, 640, 640, 3))
        >>> load_checkpoint("./models/ImageNet-83/model", model=backbone)
    """
    ckpt = tf.train.Checkpoint(**ckpt_kwargs)
    ckpt_options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
    status = ckpt.read(ckpt_path, ckpt_options)
    return status.assert_nontrivial_match().expect_partial()


def load_pretrained_model(name_or_url_or_path, model, with_fc=False, github_access_token=None):
    r"""Load pretrained weights to the model.

    Args:
        name_or_url_or_path: The name of a model registered in hanser.models.hub, or url to the model zip file,
            or path to the model zip file.
        model: The model to load weights.
        with_fc: Whether to load the weight of final fc layer, only used for ImageNet inference.

    Returns:
        A load status object.  See `tf.train.Checkpoint.restore` for details.

    Examples:
        >>> # Object detection
        >>> from hanser.models.backbone.resnet_vd import resnet50
        >>> from hanser.models.detection.retinanet import RetinaNet
        >>> backbone = resnet50()
        >>> model = RetinaNet(backbone, num_anchors=9, num_classes=80)
        >>> model.build((None, 640, 640, 3))
        >>> load_pretrained_model("resnetvd50_nlb", backbone)

        >>> # Transfer learning for classification
        >>> from hanser.models.imagenet.resnet_vd import resnet50
        >>> model = resnet50(num_classes=102)
        >>> load_pretrained_model("resnetvd50", model)

        >>> # ImageNet inference
        >>> from hanser.models.imagenet.resnet_vd import resnet50
        >>> model = resnet50()
        >>> load_pretrained_model("resnetvd50", model, with_fc=True)
    """
    ckpt_path = load_model_from_hub(name_or_url_or_path, github_access_token=github_access_token)
    status = load_checkpoint(ckpt_path, model=model)
    if with_fc:
        assert hasattr(model, "fc")
        assert hasattr(model.fc, "kernel")
        assert hasattr(model.fc, "bias")
        weight = tf.train.load_variable(ckpt_path, "model/fc_ckpt_ignored/kernel/.ATTRIBUTES/VARIABLE_VALUE")
        model.fc.kernel.assign(weight)
        weight = tf.train.load_variable(ckpt_path, "model/fc_ckpt_ignored/bias/.ATTRIBUTES/VARIABLE_VALUE")
        model.fc.bias.assign(weight)
    return status


def convert_checkpoint(ckpt_path, save_path, key_map=None):
    r"""
    This function does two things:
        1. select model variables and exclude others (optimizer, epoch, ...)
        2. map keys to different name

    Args:
        ckpt_path:
        save_path:
        key_map:

    Returns:
        
    Examples:
        >>> convert_checkpoint("~/Downloads/resnet50/ckpt", "~/Downloads/resnet50/model")
    """

    key_map = key_map or {
        'model/fc': 'model/fc_ckpt_ignored'
    }

    reader = tf.train.load_checkpoint(str(ckpt_path))
    keys = list(reader.get_variable_to_shape_map().keys())
    model_keys = [ k for k in keys if k.startswith("model/") and 'OPTIMIZER_SLOT' not in k ]

    end = '/.ATTRIBUTES/VARIABLE_VALUE'
    assert all(k.endswith(end) for k in model_keys)

    root = tf.keras.layers.Layer()
    for key in model_keys:
        layer = root
        key_name = key
        for k, km in key_map.items():
            if key[:len(k)] == k:
                key_name = km + key[len(k):]
                break
        path = key_name[:-len(end)].split('/')
        for i in range(1, len(path) - 1):
            if not hasattr(layer, path[i]):
                child = tf.keras.layers.Layer()
                setattr(layer, path[i], child)
                layer = child
            else:
                layer = getattr(layer, path[i])
        if hasattr(layer, path[-1]):
            print(key)
            raise ValueError("Variable duplicate")
        setattr(layer, path[-1], tf.Variable(reader.get_tensor(key)))

    fake_ckpt = tf.train.Checkpoint(model=root)
    return fake_ckpt.write(str(save_path))


def init_layer_ascending_drop_path(model, max_drop_path=None):
    r"""Initialize drop path rate of layers by depth in ascending.

    Args:
        model (tf.keras.Model or tf.keras.layers.Layer): Model or layer with .submodules methods.
        max_drop_path (float): Maximum drop path rate. Defaults to the rate of the last DropPath.

    Returns:
        None
    """
    def get_layer_number(l):
        match = re.match("drop_path_?([0-9]+)?", l.name)
        if match is None:
            raise AttributeError("Not supported layer name")
        if match.group(1) is not None:
            return int(match.group(1))
        else:
            return 0
    from hanser.models.modules import DropPath
    ls = [ l for l in model.submodules if isinstance(l, DropPath) ]
    ls = sorted(ls, key=get_layer_number)
    if max_drop_path is None:
        max_drop_path = ls[-1].rate.numpy()
    for i, l in enumerate(ls):
        rate = (i + 1) / len(ls) * max_drop_path
        l.rate.assign(rate)