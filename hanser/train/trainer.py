from hhutil.io import time_now

import tensorflow as tf
from tensorflow.python.distribute.input_lib import DistributedDataset
import tensorflow.keras.mixed_precision as mixed_precision

from tensorflow.python.keras.callbacks import CallbackList

from hanser.tpu import local_results


def minimize(strategy, tape, optimizer, loss, trainable_variables, grad_clip_norm=None):
    grads = tape.gradient(loss, trainable_variables)
    aggregate_grads_outside_optimizer = grad_clip_norm and is_tpu_strategy(strategy)

    if aggregate_grads_outside_optimizer:
        grads = tf.distribute.get_replica_context().all_reduce('sum', grads)

    if grad_clip_norm:
        grads = tf.clip_by_global_norm(grads, grad_clip_norm)[0]
    if trainable_variables:
        if aggregate_grads_outside_optimizer:
            optimizer.apply_gradients(
                zip(grads, trainable_variables),
                experimental_aggregate_gradients=False)
        else:
            optimizer.apply_gradients(zip(grads, trainable_variables))


@tf.function
def identity(x):
    return x


def strategy_run(strategy, fn, args):
    if strategy is not None:
        return strategy.run(fn, args=args)
    else:
        return fn(*args)


def log_metrics(stage, steps, results, metric_history=None, epoch=None):
    log_str = "%s %s %d/%d - " % (time_now(), stage, steps, steps)
    metric_logs = []
    for k, v in results:
        metric_logs.append("%s: %.4f" % (k, v))
        if metric_history:
            metric_history.record(stage, epoch, k, v)
    log_str += " - ".join(metric_logs)
    print(log_str)


def run_epoch(step_fn, iterator, steps, metrics, multiple_steps=False):
    for m in metrics:
        m.reset_states()
    if multiple_steps:
        step_fn(iterator, steps)
    else:
        for step in range(steps):
            step_fn(iterator)
    metric_results = [
        (m.name, m.result().numpy())
        for m in metrics
    ]
    return metric_results


@tf.function
def identity(x):
    return x

def is_tpu_strategy(strategy):
    if strategy is None:
        return False
    return "TPUStrategy" in type(strategy).__name__


def parse_strategy(strategy='auto'):
    if strategy is not None:
        if strategy == 'auto':
            strategy = tf.distribute.get_strategy()
        if not is_tpu_strategy(strategy):
            strategy = None
    return strategy


def is_global_bfloat16():
    return mixed_precision.global_policy().compute_dtype == 'bfloat16'


def is_global_float16():
    return mixed_precision.global_policy().compute_dtype == 'float16'


def validate_dataset(strategy, *datsets):
    if is_tpu_strategy(strategy):
        for ds in datsets:
            assert isinstance(ds, DistributedDataset)


def cast(xs, dtype):
    if isinstance(xs, tf.Tensor):
        if xs.dtype != dtype:
            xs = tf.cast(xs, dtype)
        return xs
    elif isinstance(xs, (tuple, list)):
        return xs.__class__(cast(x, dtype) for x in xs)
    elif isinstance(xs, dict):
        return {k: cast(v, dtype) for k, v in xs.items()}
    else:
        return xs


def cast_fp32(xs):
    return cast(xs, tf.float32)

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
    if is_tpu_strategy(strategy):
        return misc_concat(values)
    return values


class MetricHistory:

    def __init__(self, stages):
        self.stages = stages
        # stage -> epoch -> metric -> value
        self._history = {
            stage: {}
            for stage in stages
        }

    def record(self, stage, epoch, metric, value):
        h = self._history[stage]
        if epoch not in h:
            h[epoch] = {}
        h[epoch][metric] = value

    def get_metric(self, metric, stage=None, start=None, end=None):
        if stage is None:
            return {
                stage: self.get_metric(metric, stage, start, end)
                for stage in self.stages
            }
        else:
            h = self._history[stage]
            epochs = list(h.keys())
            if len(epochs) == 0:
                return None
            min_epoch, max_epochs = min(epochs), max(epochs)
            if start is None:
                start = min_epoch
            if end is None:
                end = max_epochs
            values = []
            for e in range(start, end + 1):
                if e in h:
                    values.append(h[e].get(metric))
            if all(v is None for v in values):
                return None
            elif len(values) == 1:
                return values[0]
            else:
                return values

    def get_epochs(self, start, end, stage=None):
        if stage is None:
            h = {
                stage: self.get_epochs(start, end, stage)
                for stage in self.stages
            }
            for k in h.keys():
                if h[k] is None:
                    del h[k]
            return h
        else:
            h = self._history[stage]
            metrics = set()
            for e in range(start, end + 1):
                if e not in h:
                    continue
                for m in h[e].keys():
                    metrics.add(m)
            return {
                m: self.get_metric(m, stage, start, end)
                for m in metrics
            }


class Trainer:

    def __init__(self, model, criterion, optimizer, metrics=(), test_metrics=(),
                 strategy='auto', model_dir=None, metric_transform=identity,
                 grad_clip_norm=None, multiple_steps=True):
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

        self.float16 = is_global_bfloat16() or is_global_float16()

        self._epoch = tf.Variable(0, trainable=False)
        self._use_weight_decay = len(model.losses) != 0

        self.metric_history = MetricHistory(["Train", "Valid", "Test", "Eeval"])

        self._end = False

    def _make_ckpt(self, name, **kwargs):
        ckpt = tf.train.Checkpoint(**kwargs)
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, directory=self.model_dir + "/" + name, max_to_keep=1)
        return ckpt, ckpt_manager

    def _train_step_fn(self, data):
        inputs, target = data
        with tf.GradientTape() as tape:
            preds = self.model(inputs, training=True)
            if self.float16:
                preds = cast_fp32(preds)
            per_example_loss = self.criterion(target, preds)
            loss = tf.reduce_mean(per_example_loss)
            if self._use_weight_decay:
                loss = loss + tf.add_n(self.model.losses)
            if self.strategy:
                loss = loss / self.strategy.num_replicas_in_sync

        minimize(self.strategy, tape, self.optimizer, loss,
                 self.model.trainable_variables, self.grad_clip_norm)

        preds = self.metric_transform(preds)
        for metric in self.metrics:
            if 'loss' in metric.name:
                metric.update_state(per_example_loss)
            else:
                metric.update_state(target, preds, None)

    @tf.function
    def _train_step(self, iterator):
        strategy_run(self.strategy, self._train_step_fn, (next(iterator),))

    @tf.function
    def _train_multiple_steps(self, iterator, steps):
        for _ in tf.range(steps):
            strategy_run(self.strategy, self._train_step_fn, (next(iterator),))

    def _test_step_fn(self, data):
        inputs, target = data
        preds = self.model(inputs, training=False)
        if self.float16:
            preds = cast(preds, tf.float32)

        preds = self.metric_transform(preds)
        for metric in self.test_metrics:
            metric.update_state(target, preds, None)

    @tf.function
    def _test_step(self, iterator):
        strategy_run(self.strategy, self._test_step_fn, (next(iterator),))

    @tf.function
    def _test_multiple_steps(self, iterator, steps):
        for _ in tf.range(steps):
            strategy_run(self.strategy, self._test_step_fn, (next(iterator),))

    @tf.function
    def _predict_step(self, iterator, debug=False):

        def step_fn(data):
            inputs, target = data
            output = self.model(inputs, training=debug)
            if self.float16:
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
            ds_val, val_steps, val_freq=1, valid_after=None, save_per_epochs=None,
            extra_metrics=(), extra_eval_freq=1,
            extra_target_transform=tf.identity, extra_output_transform=tf.identity,
            debug=False, callbacks=None):

        callbacks = CallbackList(callbacks, model=self.model)

        validate_dataset(self.strategy, ds_train, ds_val)

        train_it = iter(ds_train)

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
                train_step_fn = self._train_multiple_steps
            else:
                train_step_fn = self._train_step
            metric_results = run_epoch(train_step_fn, train_it, steps_per_epoch, self.metrics, self.multiple_steps)
            log_metrics("Train", steps_per_epoch, metric_results, self.metric_history, epoch + 1)

            if (epoch + 1) % val_freq == 0 or (valid_after and (epoch + 1) > valid_after):
                val_it = iter(ds_val)
                if self.multiple_steps:
                    test_step_fn = self._test_multiple_steps
                else:
                    test_step_fn = self._test_step
                metric_results = run_epoch(test_step_fn, val_it, val_steps, self.test_metrics, self.multiple_steps)
                log_metrics("Valid", val_steps, metric_results, self.metric_history, epoch + 1)

            if extra_metrics and (epoch + 1) % extra_eval_freq == 0:
                val_it = iter(ds_val)
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
                log_metrics("Eeval", val_steps, metric_results, self.metric_history, epoch + 1)

            if save_per_epochs and (epoch + 1) % save_per_epochs == 0:
                print("Saved models: %s" % model_ckpt_manager.save(epoch + 1))
                print("Saved optimizer: %s" % optim_ckpt_manager.save(epoch + 1))

            callbacks.on_epoch_end(epoch + 1)
            epoch = self._epoch.assign_add(1).numpy()
            if self._end:
                print("Train end")
                break
        callbacks.on_train_end()
        return self.metric_history.get_epochs(1, epoch, "Valid")

    def evaluate(self, ds_test, test_steps):

        validate_dataset(self.strategy, ds_test)
        test_it = iter(ds_test)

        if self.multiple_steps:
            test_step_fn = self._test_multiple_steps
        else:
            test_step_fn = self._test_step
        metric_results = run_epoch(test_step_fn, test_it, test_steps, self.test_metrics, self.multiple_steps)
        log_metrics("Test", test_steps, metric_results)

    def evaluate2(self, ds_test, test_steps, metrics,
                  output_transform=tf.identity, target_transform=tf.identity, debug=False):

        validate_dataset(self.strategy, ds_test)

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

        validate_dataset(self.strategy, ds_test)
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
