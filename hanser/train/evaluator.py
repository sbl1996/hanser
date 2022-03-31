from packaging.version import parse as vparse

import tensorflow as tf
from tensorflow.keras.metrics import Mean

from hanser.distribute import is_distribute_strategy
from hanser.train.learner import cast
from hanser.train.learner_v4 import log_metrics, reduce_per_replica


class InternalModel(tf.keras.Model):

    def __init__(self, model, metrics):
        super().__init__()
        self.model = model
        self.eval_metrics = metrics

        self._num_replicas = tf.distribute.get_strategy().num_replicas_in_sync

    def test_step(self, batch):
        metrics = self.eval_metrics
        x, y = batch
        y_pred = self.model(x, training=False)
        y_pred = cast(y_pred, tf.float32)
        for name, metric in metrics.items():
            metric.update_state(y, y_pred, None)
        return {k: m.result() for k, m in metrics.items()}

    def local_eval_step(self, batch):
        x, y = batch
        y_pred = self.model(x, training=False)
        y_pred = cast(y_pred, tf.float32)
        return y, y_pred

    @tf.function
    def _local_eval_step(self, batch):
        outputs = self.distribute_strategy.run(self.local_eval_step, args=(batch,))
        outputs = reduce_per_replica(
            outputs, self.distribute_strategy, reduction='concat')
        return outputs


class Evaluator:

    def __init__(self, model, metrics, steps_per_loop=1, jit_compile=True):
        super().__init__()
        self.model = model
        self.steps_per_loop = steps_per_loop
        self.jit_compile = jit_compile

        self._internal = InternalModel(model, metrics)
        if vparse(tf.__version__) >= vparse("2.8"):
            self._internal.compile(steps_per_execution=steps_per_loop, jit_compile=self.jit_compile)
        elif vparse(tf.__version__) < vparse("2.4"):
            self._internal.compile(experimental_steps_per_execution=steps_per_loop)
        else:
            self._internal.compile(steps_per_execution=steps_per_loop)

    def load(self, fp):
        ckpt = tf.train.Checkpoint(model=self.model)
        ckpt_options = tf.train.CheckpointOptions(
            experimental_io_device="/job:localhost") if is_distribute_strategy(tf.distribute.get_strategy()) else None
        ckpt.restore(fp, ckpt_options)
        print("Load learner from %s" % (fp,))
        return True

    def evaluate(self, ds_val, val_steps=None, callbacks=None):
        callbacks = callbacks or []
        val_steps = val_steps or len(ds_val)

        for c in callbacks:
            c.begin_eval(None)

        for m in self._internal.eval_metrics.values():
            m.reset_states()
        eval_logs = self._internal.evaluate(
            ds_val, steps=val_steps, verbose=0,
            return_dict=True, _use_cached_eval_dataset=True)

        for c in callbacks:
            c.after_eval(None)

        log_metrics('eval', eval_logs, 0, stage_name='valid')

    def evaluate_local(self, ds_val, steps, metrics, callbacks=None):
        callbacks = callbacks or []
        iterator = iter(ds_val)

        for c in callbacks:
            c.begin_eval(None)

        for m in metrics.values():
            m.reset_states()
        for step in range(steps):
            y_true, y_pred = self._internal._local_eval_step(next(iterator))
            for m in metrics.values():
                m.update_state(y_true, y_pred, None)
        metric_results = {}
        for k, m in metrics.items():
            metric_results[k] = m.result().numpy()

        for c in callbacks:
            c.after_eval(None)

        log_metrics('eval', metric_results, 0, stage_name='valid')
