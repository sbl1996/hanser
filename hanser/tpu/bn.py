import tensorflow as tf

from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.tpu import tpu_function


class TPUBatchNormalization(BatchNormalization):
    # class TpuBatchNormalization(tf.layers.BatchNormalization):
    """Cross replica batch normalization."""

    def __init__(self, fused=False, **kwargs):
        if fused in (True, None):
            raise ValueError('TpuBatchNormalization does not support fused=True.')
        super().__init__(fused=fused, **kwargs)

    def _cross_replica_average(self, t, num_groups=1):
        """Calculates the average value of input tensor across TPU replicas."""
        num_shards = tpu_function.get_tpu_context().number_of_shards
        num_shards_per_group = 1
        group_assignment = None
        if num_groups > 0:
            if num_shards % num_groups != 0:
                raise ValueError('num_shards: %d mod num_groups: %d, should be 0'
                                 % (num_shards, num_groups))
            num_shards_per_group = num_shards // num_groups
            group_assignment = [[
                x for x in range(num_shards) if x // num_shards_per_group == y
            ] for y in range(num_groups)]
        return tf.compat.v1.tpu.cross_replica_sum(t, group_assignment) / tf.cast(
            num_shards_per_group, t.dtype)

    def _moments(self, inputs, reduction_axes, keep_dims):
        """Compute the mean and variance: it overrides the original _moments."""
        shard_mean, shard_variance = super()._moments(
            inputs, reduction_axes, keep_dims=keep_dims)

        shard_square_of_mean = tf.math.square(shard_mean)
        shard_mean_of_square = shard_variance + shard_square_of_mean
        group_mean = self._cross_replica_average(shard_mean)
        group_mean_of_square = self._cross_replica_average(shard_mean_of_square)
        group_variance = group_mean_of_square - tf.math.square(group_mean)
        return group_mean, group_variance
        # num_shards = tpu_function.get_tpu_context().number_of_shards or 1
        # if num_shards <= 8:  # Skip cross_replica for 2x2 or smaller slices.
        #     num_shards_per_group = 1
        # else:
        #     num_shards_per_group = max(8, num_shards // 8)
        #
        # if num_shards_per_group > 1:
        #     # Compute variance using: Var[X]= E[X^2] - E[X]^2.
        #     shard_square_of_mean = tf.math.square(shard_mean)
        #     shard_mean_of_square = shard_variance + shard_square_of_mean
        #     group_mean = self._cross_replica_average(
        #         shard_mean, num_shards_per_group)
        #     group_mean_of_square = self._cross_replica_average(
        #         shard_mean_of_square, num_shards_per_group)
        #     group_variance = group_mean_of_square - tf.math.square(group_mean)
        #     return (group_mean, group_variance)
        # else:
        #     return (shard_mean, shard_variance)
