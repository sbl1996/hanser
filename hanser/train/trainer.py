import time

import tensorflow as tf
from tensorflow.python.distribute.input_lib import DistributedDataset
import tensorflow.keras.mixed_precision.experimental as mixed_precision

from tensorflow.python.distribute.tpu_strategy import TPUStrategy, TPUStrategyV2, TPUStrategyV1
from tensorflow.python.keras.callbacks import CallbackList

from hanser.io import time_now
from hanser.tpu import local_results


@tf.function
def identity(x):
    return x


def join_metric_logs(results, delim=" - "):
    logs = []
    for k, v in results:
        logs.append("%s: %.4f" % (k, v))
    return delim.join(logs)


def print_results(stage, steps, results):
    print("%s %s %d/%d - %s" % (
        time_now(), stage, steps, steps, join_metric_logs(results, delim=" - ")))


def run_epoch(step_fn, iterator, steps, metrics, stage="Train", multiple_steps=False):
    # start = time.time()
    for m in metrics:
        m.reset_states()
    if multiple_steps:
        step_fn(iterator, steps)
    else:
        for step in range(steps):
            step_fn(iterator)
    metric_results = []
    for m in metrics:
        metric_results.append((m.name, m.result().numpy()))
    # elapsed = time.time() - start
    print_results(stage, steps, metric_results)


@tf.function
def identity(x):
    return x


def parse_strategy(strategy='auto'):
    if strategy is not None:
        if strategy == 'auto':
            strategy = tf.distribute.get_strategy()
        if not isinstance(strategy, (TPUStrategy, TPUStrategyV1, TPUStrategyV2)):
            strategy = None
    return strategy


def is_global_bfloat16():
    return mixed_precision.global_policy().compute_dtype == 'bfloat16'


def cast_fp32(xs):
    if isinstance(xs, tf.Tensor):
        return tf.cast(xs, tf.float32)
    elif isinstance(xs, (tuple, list)):
        return xs.__class__(cast_fp32(x) for x in xs)
    elif isinstance(xs, dict):
        return {k: cast_fp32(v) for k, v in xs.items()}
    else:
        return xs


def misc_concat(values):
    if isinstance(values, (tuple, list)):
        val = values[0]
        if tf.is_tensor(val):
            return tf.concat(values, 0)
        elif isinstance(val, dict):
            d = {}
            for k in val.keys():
                d[k] = misc_concat([v[k] for v in values])
            return d
        elif isinstance(val, (tuple, list)):
            return val.__class__(v for l in values for v in l)
        else:
            return values
    elif isinstance(values, dict):
        return {k: misc_concat(v) for k, v in values.items()}
    else:
        return values


def maybe_cat(values, strategy):
    if isinstance(strategy, TPUStrategy):
        return misc_concat(values)
    return values


class Trainer:

    def __init__(self, model, criterion, optimizer, metrics=(), test_metrics=(),
                 strategy='auto', model_dir=None, metric_transform=identity,
                 grad_clip_norm=None, multiple_steps=False):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics
        self.test_metrics = test_metrics
        self.metric_transform = metric_transform
        self.model_dir = model_dir

        self.strategy = parse_strategy(strategy)
        if strategy and model_dir:
            assert model_dir.startswith('gs'), "Use gs://... as `model_dir` on TPU"

        self.grad_clip_norm = grad_clip_norm
        self.multiple_steps = multiple_steps

        self.bfloat16 = is_global_bfloat16()

        self._epoch = tf.Variable(0, trainable=False)
        self._use_weight_decay = len(model.losses) != 0

    def _make_ckpt(self, name, **kwargs):
        ckpt = tf.train.Checkpoint(**kwargs)
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, directory=self.model_dir + "/" + name, max_to_keep=1)
        return ckpt, ckpt_manager

    @tf.function
    def _train_step(self, iterator):

        def step_fn(data):
            inputs, target = data
            with tf.GradientTape() as tape:
                preds = self.model(inputs, training=True)
                if self.bfloat16:
                    preds = cast_fp32(preds)
                per_example_loss = self.criterion(target, preds)
                loss = tf.reduce_mean(per_example_loss)
                if self._use_weight_decay:
                    loss = loss + tf.add_n(self.model.losses)
                if self.strategy:
                    loss = loss / self.strategy.num_replicas_in_sync
            grads = tape.gradient(loss, self.model.trainable_variables)
            if self.grad_clip_norm:
                grads = tf.clip_by_global_norm(grads, self.grad_clip_norm)[0]
            self.optimizer.apply_gradients(
                zip(grads, self.model.trainable_variables))

            preds = self.metric_transform(preds)
            for metric in self.metrics:
                if 'loss' in metric.name:
                    metric.update_state(per_example_loss)
                else:
                    metric.update_state(target, preds, None)

        if self.strategy:
            self.strategy.run(step_fn, args=(next(iterator),))
        else:
            step_fn(next(iterator))

    @tf.function
    def _train_multiple_steps(self, iterator, steps):

        def step_fn(data):
            inputs, target = data
            with tf.GradientTape() as tape:
                preds = self.model(inputs, training=True)
                if self.bfloat16:
                    preds = cast_fp32(preds)
                per_example_loss = self.criterion(target, preds)
                loss = tf.reduce_mean(per_example_loss)
                if self._use_weight_decay:
                    loss = loss + tf.add_n(self.model.losses)
                if self.strategy:
                    loss = loss / self.strategy.num_replicas_in_sync
            grads = tape.gradient(loss, self.model.trainable_variables)
            if self.grad_clip_norm:
                grads = tf.clip_by_global_norm(grads, self.grad_clip_norm)[0]
            self.optimizer.apply_gradients(
                zip(grads, self.model.trainable_variables))

            preds = self.metric_transform(preds)
            for metric in self.metrics:
                if 'loss' in metric.name:
                    metric.update_state(per_example_loss)
                else:
                    metric.update_state(target, preds, None)

        for _ in tf.range(steps):
            if self.strategy:
                self.strategy.run(step_fn, args=(next(iterator),))
            else:
                step_fn(next(iterator))

    @tf.function
    def _test_step(self, iterator):

        def step_fn(data):
            inputs, target = data
            preds = self.model(inputs, training=False)
            if self.bfloat16:
                preds = cast_fp32(preds)

            preds = self.metric_transform(preds)
            for metric in self.test_metrics:
                metric.update_state(target, preds, None)

        if self.strategy:
            self.strategy.run(step_fn, args=(next(iterator),))
        else:
            step_fn(next(iterator))

    @tf.function
    def _test_multiple_steps(self, iterator, steps):

        def step_fn(data):
            inputs, target = data
            preds = self.model(inputs, training=False)
            if self.bfloat16:
                preds = cast_fp32(preds)

            preds = self.metric_transform(preds)
            for metric in self.test_metrics:
                metric.update_state(target, preds, None)

        for _ in tf.range(steps):
            if self.strategy:
                self.strategy.run(step_fn, args=(next(iterator),))
            else:
                step_fn(next(iterator))

    @tf.function
    def _predict_step(self, iterator, debug=False):

        def step_fn(data):
            inputs, target = data
            output = self.model(inputs, training=debug)
            if self.bfloat16:
                output = cast_fp32(output)
            return target, output

        if self.strategy:
            return local_results(
                self.strategy, self.strategy.run(step_fn, args=(next(iterator),)))
        else:
            return step_fn(next(iterator))

    def restore(self, checkpoint, manager):
        assert self.model_dir is not None, "`model_dir` should be provided."

        if manager.latest_checkpoint:
            if self.strategy:
                with self.strategy.scope():
                    checkpoint.restore(manager.latest_checkpoint)
            else:
                checkpoint.restore(manager.latest_checkpoint)
            print("Restored from {}".format(manager.latest_checkpoint))
            return True
        else:
            print("Initializing from scratch.")
            return False

    def resume(self):
        model_ckpt, model_ckpt_manager = self._make_ckpt("models", model=self.model)
        optim_ckpt, optim_ckpt_manager = self._make_ckpt("optim", optimizer=self.optimizer, epoch=self._epoch)

        self.restore(model_ckpt, model_ckpt_manager)
        self.restore(optim_ckpt, optim_ckpt_manager)

    def fit(self, epochs, ds_train, steps_per_epoch,
            ds_val, val_steps, val_freq=1, save_per_epochs=None,
            extra_metrics=(), extra_eval_freq=1,
            extra_target_transform=tf.identity, extra_output_transform=tf.identity,
            debug=False, callbacks=None):

        callbacks = CallbackList(callbacks, model=self.model)

        if self.strategy:
            assert isinstance(ds_train, DistributedDataset)
            assert isinstance(ds_val, DistributedDataset)

        train_it = iter(ds_train)
        val_it = iter(ds_val)

        if save_per_epochs:
            assert self.model_dir is not None, "`model_dir` should be provided."

            model_ckpt, model_ckpt_manager = self._make_ckpt("models", model=self.model)
            optim_ckpt, optim_ckpt_manager = self._make_ckpt("optim", optimizer=self.optimizer, epoch=self._epoch)

        epoch = self._epoch.numpy()
        max_epochs = epoch + epochs

        if extra_metrics:
            assert extra_target_transform
            assert extra_output_transform
            assert extra_eval_freq

        callbacks.on_train_begin()
        while epoch < max_epochs:
            print('Epoch %d/%d' % (epoch + 1, epochs))

            callbacks.on_epoch_begin(epoch + 1)
            if self.multiple_steps:
                run_epoch(self._train_multiple_steps, train_it, steps_per_epoch, self.metrics, "Train", self.multiple_steps)
            else:
                run_epoch(self._train_step, train_it, steps_per_epoch, self.metrics, "Train", self.multiple_steps)

            if (epoch + 1) % val_freq == 0:
                if self.multiple_steps:
                    run_epoch(self._test_multiple_steps, val_it, val_steps, self.test_metrics, "Valid", self.multiple_steps)
                else:
                    run_epoch(self._test_step, val_it, val_steps, self.test_metrics, "Valid", self.multiple_steps)

            if extra_metrics and (epoch + 1) % extra_eval_freq == 0:
                start = time.time()
                for step in range(val_steps):
                    target, output = self._predict_step(val_it, debug)

                    target = maybe_cat(target, self.strategy)
                    output = maybe_cat(output, self.strategy)

                    output = extra_output_transform(output)
                    target = extra_target_transform(target)

                    for m in extra_metrics:
                        m.update_state(target, output, None)
                metric_results = []
                for m in extra_metrics:
                    metric_results.append((m.name, m.result().numpy()))
                    m.reset_states()
                elapsed = time.time() - start
                print_results("Eval", elapsed, metric_results)

            if save_per_epochs and (epoch + 1) % save_per_epochs == 0:
                print("Saved models: %s" % model_ckpt_manager.save(epoch + 1))
                print("Saved optimizer: %s" % optim_ckpt_manager.save(epoch + 1))

            callbacks.on_epoch_end(epoch + 1)

            epoch = self._epoch.assign_add(1).numpy()

        callbacks.on_train_end()

    def evaluate(self, ds_test, test_steps):

        if self.strategy:
            assert isinstance(ds_test, DistributedDataset)
        test_it = iter(ds_test)

        run_epoch(self._test_step, test_it, test_steps, self.test_metrics, "Test")

    def evaluate2(self, ds_test, test_steps, metrics,
                  output_transform=tf.identity, target_transform=tf.identity, debug=False):

        if self.strategy:
            assert isinstance(ds_test, DistributedDataset)

        test_it = iter(ds_test)

        for step in range(test_steps):
            target, output = self._predict_step(test_it, debug)

            target = maybe_cat(target, self.strategy)
            output = maybe_cat(output, self.strategy)

            output = output_transform(output)
            target = target_transform(target)

            for m in metrics:
                m.update_state(target, output, None)
        results = {m.name: m.result().numpy() for m in metrics}
        return results

    def collect(self, ds_test, test_steps, output_transform=tf.identity, target_transform=tf.identity):

        if self.strategy:
            assert isinstance(ds_test, DistributedDataset)
        test_it = iter(ds_test)

        targets = []
        outputs = []

        for step in range(test_steps):
            target, output = self._predict_step(test_it)

            targets.append(target)
            outputs.append(output)

        target = misc_concat(targets)
        output = misc_concat(outputs)

        output = output_transform(output)
        target = target_transform(target)

        return target, output
