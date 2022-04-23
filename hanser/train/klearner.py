import tensorflow as tf
from tensorflow.keras.metrics import Mean
import tensorflow.keras.mixed_precision as mixed_precision

from hanser.distribute import reduce_per_replica

from hhutil.io import time_now


def log_metrics(stage, metrics, stage_name=None, print_fn=print):
    stage_name = stage_name or stage
    end_at = time_now()
    log_str = "%s %s - " % (end_at, stage_name)
    metric_logs = []
    for k, v in metrics.items():
        metric_logs.append("%s: %.4f" % (k, v))
    log_str += ", ".join(metric_logs)
    print_fn(log_str)


def cast(xs, dtype, whiltelist=(tf.int32, tf.int64, tf.bool)):
    def func(x):
        if x.dtype != dtype and all(x.dtype != wdtype for wdtype in whiltelist):
            x = tf.cast(x, dtype)
        return x
    return tf.nest.map_structure(func, xs)


def reduce_loss(per_example_loss):
    loss = tf.reduce_mean(per_example_loss)
    num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
    if num_replicas > 1.0:
        loss = loss / num_replicas
    return loss


def update_metrics(metrics, y_true, y_pred, per_example_loss=None):
    for name, metric in metrics.items():
        if 'loss' in name and type(metric) == Mean:
            metric.update_state(per_example_loss)
        else:
            metric.update_state(y_true, y_pred, None)


class SuperLearner:

    def __init__(self, model, criterion, optimizer, train_metrics, eval_metrics,
                 steps_per_loop=-1, eval_steps_per_loop=None, jit_compile=True, work_dir=None):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_metrics = train_metrics
        self.eval_metrics = eval_metrics
        self.steps_per_loop = steps_per_loop
        self.eval_steps_per_loop = eval_steps_per_loop or steps_per_loop
        self.jit_compile = jit_compile
        self.work_dir = work_dir

        self._dtype = tf.dtypes.as_dtype(mixed_precision.global_policy().compute_dtype)

        self._train_function = None
        self._eval_function = None

        self._distribute_strategy = tf.distribute.get_strategy()

    def train_step(self, batch):
        model = self.model
        optimizer = self.optimizer

        inputs, target = batch
        with tf.GradientTape() as tape:
            inputs = cast(inputs, self._dtype)
            preds = model(inputs, training=True)
            preds = cast(preds, tf.float32)
            per_example_loss = self.criterion(target, preds)
            loss = reduce_loss(per_example_loss)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        update_metrics(self.train_metrics, target, preds, per_example_loss)
        return {k: m.result() for k, m in self.train_metrics.items()}

    def eval_step(self, batch):
        inputs, target = batch
        inputs = cast(inputs, self._dtype)
        preds = self.model(inputs, training=False)
        preds = cast(preds, tf.float32)
        update_metrics(self.eval_metrics, target, preds)
        return {k: m.result() for k, m in self.eval_metrics.items()}

    def make_train_function(self):
        if self._train_function is not None:
            return self._train_function

        train_step = self.train_step
        if self.jit_compile:
            train_step = tf.function(
                train_step, jit_compile=True, experimental_relax_shapes=True)

        def train_function(iterator, steps):
            for _ in tf.range(steps):
                outputs = self._distribute_strategy.run(train_step, args=(next(iterator),))
                outputs = reduce_per_replica(
                    outputs, self._distribute_strategy, reduction='first')
            return outputs

        train_function = tf.function(
            train_function, experimental_relax_shapes=True)
        self._train_function = train_function
        return self._train_function

    def make_eval_function(self):
        if self._eval_function is not None:
            return self._eval_function

        eval_step = self.eval_step
        if self.jit_compile:
            eval_step = tf.function(
                eval_step, jit_compile=True, experimental_relax_shapes=True)

        def eval_function(iterator, steps):
            for _ in tf.range(steps):
                outputs = self._distribute_strategy.run(eval_step, args=(next(iterator),))
                outputs = reduce_per_replica(
                    outputs, self._distribute_strategy, reduction='first')
            return outputs

        eval_function = tf.function(
            eval_function, experimental_relax_shapes=True)
        self._eval_function = eval_function
        return self._eval_function

    def train_epoch(self, iterator, steps):
        metrics = self.train_metrics
        train_function = self.make_train_function()
        steps_per_loop = self.steps_per_loop
        if steps_per_loop == -1:
            steps_per_loop = steps

        for metric in metrics.values():
            metric.reset_states()

        current_step = 0
        while current_step < steps:
            run_steps = min(steps - current_step, steps_per_loop)
            train_function(iterator, tf.convert_to_tensor(run_steps, tf.int64))
            current_step += run_steps
        return {name: metric.result() for name, metric in metrics.items()}

    def evaluate(self, iterator, steps):
        metrics = self.eval_metrics
        eval_function = self.make_eval_function()
        steps_per_loop = self.eval_steps_per_loop
        if steps_per_loop == -1:
            steps_per_loop = steps

        for metric in metrics.values():
            metric.reset_states()

        current_step = 0
        while current_step < steps:
            run_steps = min(steps - current_step, steps_per_loop)
            eval_function(iterator, tf.convert_to_tensor(run_steps, tf.int64))
            current_step += run_steps
        return {name: metric.result() for name, metric in metrics.items()}

    def fit(self, ds_train, epochs, ds_val=None, val_freq=1,
            steps_per_epoch=None, val_steps=None):

        steps_per_epoch = steps_per_epoch or len(ds_train)

        train_iter = iter(ds_train)
        if ds_val is not None:
            eval_iter = iter(ds_val)
        print(f"{time_now()} Start training")
        for epoch in range(epochs):
            self._epoch = epoch
            print("Epoch %d/%d" % (epoch + 1, epochs))

            logs = self.train_epoch(train_iter, steps_per_epoch)
            log_metrics('train', logs)

            if ds_val is not None and (epoch + 1) % val_freq == 0:
                logs = self.evaluate(eval_iter, val_steps)
                log_metrics('valid', logs)
