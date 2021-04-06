import tensorflow as tf

def load_checkpoint(ckpt_path, **ckpt_kwargs):
    ckpt = tf.train.Checkpoint(**ckpt_kwargs)
    ckpt_options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
    status = ckpt.read(ckpt_path, ckpt_options)
    return status.assert_existing_objects_matched()