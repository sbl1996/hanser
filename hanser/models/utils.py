import tensorflow as tf

from hanser.models.hub import load_model_from_hub


def load_checkpoint(ckpt_path, **ckpt_kwargs):
    ckpt = tf.train.Checkpoint(**ckpt_kwargs)
    ckpt_options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
    status = ckpt.read(ckpt_path, ckpt_options)
    return status.assert_nontrivial_match().expect_partial()


def load_pretrained_model(name_or_url, model, with_fc=False):
    ckpt_path = load_model_from_hub(name_or_url)
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
    # This function does two things:
    #   1. select model variables and exclude others (optimizer, epoch, ...)
    #   2. map keys to different name

    key_map = key_map or {
        'model/fc': 'model/fc_ckpt_ignored'
    }

    reader = tf.train.load_checkpoint(ckpt_path)
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
    return fake_ckpt.write(save_path)
