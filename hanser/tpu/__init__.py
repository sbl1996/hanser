import os

import tensorflow as tf
from tensorflow.python.distribute.values import PerReplica

def get_colab_tpu():
    tpu = os.environ.get('COLAB_TPU_ADDR')
    if tpu:
        tf.keras.backend.clear_session()
        tpu = 'grpc://' + os.environ['COLAB_TPU_ADDR']
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        cluster_spec = resolver.cluster_spec()
        if cluster_spec:
            config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
        return resolver

def auth():
    from google.colab import auth
    auth.authenticate_user()


def local_results(strategy, values):
    if isinstance(values, PerReplica):
        return strategy.experimental_local_results(values)
    elif isinstance(values, (list, tuple)):
        return values.__class__(strategy.experimental_local_results(v) for v in values)
    else:
        raise ValueError("`values` must be PerReplica, list or tuple or PerReplica, got %s" % type(values))
