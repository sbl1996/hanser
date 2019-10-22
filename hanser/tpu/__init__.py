import os

import tensorflow as tf


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

