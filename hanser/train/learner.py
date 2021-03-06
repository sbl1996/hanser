from abc import ABCMeta
from typing import Sequence, Mapping, Optional

import tensorflow as tf
import tensorflow.keras.mixed_precision.experimental as mixed_precision
from tensorflow.keras.metrics import Metric, Mean

from hhutil.io import fmt_path, eglob, rm, time_now

from hanser.train.metric_history import MetricHistory
from hanser.train.callbacks import config_callbacks


def find_most_recent(work_dir, pattern):
    d = fmt_path(work_dir)
    pattern = pattern
    saves = list(d.glob(pattern))
    if len(saves) == 0:
        return None
    else:
        fp = max(saves, key=lambda f: f.stat().st_mtime)
        return fp


@tf.function
def default_metric_transform(x):
    if isinstance(x, (tuple, list)):
        return x[0]
    return x


def strategy_run(strategy, fn, args):
    if strategy is not None:
        return strategy.run(fn, args=args)
    else:
        return fn(*args)


def is_tpu_strategy(strategy):
    if strategy is None:
        return False
    return "TPUStrategy" in type(strategy).__name__


def is_mirrored_strategy(strategy):
    if strategy is None:
        return False
    return "MirroredStrategy" in type(strategy).__name__


def is_distribute_strategy(strategy):
    return is_tpu_strategy(strategy) or is_mirrored_strategy(strategy)


def parse_strategy(strategy='auto') -> Optional[tf.distribute.Strategy]:
    if strategy is not None:
        if strategy == 'auto':
            strategy = tf.distribute.get_strategy()
        if not is_distribute_strategy(strategy):
            strategy = None
    return strategy


def is_global_bfloat16():
    return mixed_precision.global_policy().compute_dtype == 'bfloat16'


def is_global_float16():
    return mixed_precision.global_policy().compute_dtype == 'float16'


class Learner(metaclass=ABCMeta):

    def __init__(self, model, criterion, optimizers,
                 train_metrics: Mapping[str, Metric],
                 eval_metrics: Mapping[str, Metric], work_dir,
                 grad_clip_norm=0.0, multiple_steps=True, xla_compile=False,
                 metric_transform=default_metric_transform,
                 target_metric_transform=default_metric_transform,
                 output_metric_transform=default_metric_transform,
                 n_batches_per_step=None):
        if not isinstance(optimizers, Sequence):
            optimizers = [optimizers]
        optimizers = list(optimizers)

        self._strategy = parse_strategy('auto')
        work_dir = fmt_path(work_dir)

        self.model = model
        self.criterion = criterion
        self.optimizers = optimizers
        self.train_metrics = train_metrics
        self.eval_metrics = eval_metrics
        self.work_dir = work_dir
        self.dtype = tf.dtypes.as_dtype(mixed_precision.global_policy().compute_dtype)
        if self.dtype == tf.float16:
            self.optimizers = [
                mixed_precision.LossScaleOptimizer(optimizer, 'dynamic')
                if not isinstance(optimizer, mixed_precision.LossScaleOptimizer) else optimizer
                for optimizer in self.optimizers
            ]
        self.grad_clip_norm = grad_clip_norm
        self.multiple_steps = multiple_steps
        self.xla_compile = xla_compile
        self.metric_transform = metric_transform
        self.target_metric_transform = target_metric_transform
        self.output_metric_transform = output_metric_transform

        self._log_dir = self.work_dir / "runs"
        self._writer = None

        self._verbose = True
        self._state = {
            "train": {},
            "eval": {},
            "test": {},
        }

        # epoch -> stage -> metric -> value
        self.metric_history = MetricHistory(["train", "eval", "test"])

        self._terminated = False
        self.set_global_state("epoch", tf.Variable(-1, dtype=tf.int64))

        if self.xla_compile:
            self.xla_train_batch = tf.function(self.train_batch, experimental_compile=True)

        self.n_batches_per_step = n_batches_per_step

    def _make_ckpt(self):
        optimizers = self.optimizers
        # if len(optimizers) == 1 and hasattr(self, "original_optimizer"):
        #     optimizers = [self.original_optimizer]
        ckpt = tf.train.Checkpoint(
            model=self.model, optimizers=optimizers,
            epoch=self._state['train']['epoch']
        )
        ckpt_options = tf.train.CheckpointOptions(
            experimental_io_device="/job:localhost") if self._strategy else None
        return ckpt, ckpt_options

    def train_batch(self, batch):
        pass

    def train_batches(self, *batches):
        pass

    def eval_batch(self, batch):
        pass

    def test_batch(self, batch):
        pass

    @property
    def epoch(self):
        return self._state['train']['epoch'].numpy()

    def set_state(self, k, v, mode):
        if k not in self._state[mode] or not isinstance(self._state[mode][k], tf.Variable):
            self._state[mode][k] = v
        else:
            self._state[mode][k].assign(v)

    def set_global_state(self, k, v):
        modes = ['train', 'eval', 'test']
        for m in modes:
            self.set_state(k, v, m)

    def fit(self, ds_train, max_epochs, ds_val=None, val_freq=1,
            steps_per_epoch=None, val_steps=None, save_freq=None, callbacks=None,
            reuse_train_iterator=False):

        steps_per_epoch = steps_per_epoch or len(ds_train)
        steps_per_epoch = tf.convert_to_tensor(steps_per_epoch, dtype=tf.int32)

        do_eval = ds_val is not None
        if do_eval:
            self._val_freq = val_freq
            val_steps = val_steps or len(ds_val)
            val_steps = tf.convert_to_tensor(val_steps, dtype=tf.int32)

        cbks = config_callbacks(
            self,
            callbacks,
            save_freq=save_freq,
        )
        start_epoch = self.epoch + 1
        self.set_global_state("epochs", max_epochs)
        self.set_global_state("step", tf.Variable(0, dtype=tf.int64))

        print("%s Start training" % (time_now(),))

        # May have problem when recover training from checkpoint
        if reuse_train_iterator:
            self._train_it = iter(ds_train)
        cbks.begin_train(self._state['train'])
        for epoch in range(start_epoch, max_epochs):
            self.set_global_state("epoch", epoch)

            state = self._state['train']
            state['metrics'] = {}
            cbks.begin_epoch(state)
            if not reuse_train_iterator:
                self._train_it = iter(ds_train)
            self._run_epoch(self._train_it, steps_per_epoch, cbks, 'train')
            cbks.after_epoch(state)

            if do_eval and (epoch + 1) % self._val_freq == 0:
                state = self._state['eval']
                state['metrics'] = {}
                cbks.begin_eval(state)
                self._run_epoch(iter(ds_val), val_steps, cbks, 'eval')
                cbks.after_eval(state)

            if self._terminated:
                print("Terminated at epoch %d" % (epoch + 1))
                break
        cbks.after_train(self._state['train'])

    @tf.function
    def _xla_train_step(self, batch):
        strategy_run(self._strategy, self.xla_train_batch, (batch,))

    @tf.function
    def _train_step(self, batch):
        strategy_run(self._strategy, self.train_batch, (batch,))

    @tf.function
    def _train_step_on_batches(self, batches):
        strategy_run(self._strategy, self.train_batches, batches)

    @tf.function
    def _eval_step(self, batch):
        strategy_run(self._strategy, self.eval_batch, (batch,))

    @tf.function
    def _test_step(self, inputs):
        strategy_run(self._strategy, self.test_batch, (inputs,))

    @tf.function
    def _run_steps(self, step_fn, iterator, n_batches_per_step, n_steps, callbacks, state):
        for i in tf.range(n_steps):
            state['step'].assign_add(1)
            callbacks.begin_batch(state)
            if n_batches_per_step is not None:
                batches = tuple(next(iterator) for bi in range(n_batches_per_step))
                step_fn(batches)
            else:
                batch = next(iterator)
                step_fn(batch)
            callbacks.after_batch(state)

    @tf.function
    def _run_xla_train_steps(self, iterator, n_steps, callbacks, state):
        for i in tf.range(n_steps):
            batch = next(iterator)
            state['step'].assign_add(1)
            callbacks.begin_batch(state)
            self._xla_train_step(batch)
            callbacks.after_batch(state)

    def update_metrics(self, metrics, y_true, y_pred, per_example_loss=None):
        y_true = self.target_metric_transform(y_true)
        y_pred = self.output_metric_transform(y_pred)
        for name, metric in metrics.items():
            if 'loss' in name and type(metric) == Mean:
                metric.update_state(per_example_loss)
            else:
                metric.update_state(y_true, y_pred, None)


    def _run_epoch(self, iterator, steps, callbacks, mode):
        state = self._state[mode]
        metrics = getattr(self, mode + "_metrics")
        step_fn = getattr(self, f"_{mode}_step")

        state.update({
            'steps': steps,
        })
        state['step'].assign(-1)

        for metric in metrics.values():
            metric.reset_states()

        if self.multiple_steps:
            sub_state = {
                k: state[k] for k in ["step", "steps", "epoch", "epochs"]
            }
            if self.xla_compile and mode == 'train':
                self._run_xla_train_steps(iterator, steps, callbacks, sub_state)
            elif mode == 'train' and self.n_batches_per_step is not None:
                step_fn = self._train_step_on_batches
                self._run_steps(
                    step_fn, iterator, self.n_batches_per_step, steps, callbacks, sub_state)
            else:
                self._run_steps(
                    step_fn, iterator, None, steps, callbacks, sub_state)
        else:
            for _ in range(steps):
                batch = next(iterator)
                state['step'].assign_add(1)
                callbacks.begin_batch(state)
                step_fn(batch)
                callbacks.after_batch(state)

        for name, metric in metrics.items():
            state['metrics'][name] = metric.result().numpy()

    def evaluate(self, ds_val, val_steps=None, callbacks=None):
        if 'step' not in self._state['eval']:
            self.set_state('step', tf.Variable(0, dtype=tf.int64), 'eval')
        if 'epoch' not in self._state['eval']:
            self.set_state('epoch', tf.Variable(0, dtype=tf.int64), 'eval')
        if 'epochs' not in self._state['eval']:
            self.set_state('epochs', tf.Variable(0, dtype=tf.int64), 'eval')

        val_steps = val_steps or len(ds_val)
        cbks = config_callbacks(self, callbacks, mode='eval')

        state = self._state['eval']
        state['metrics'] = {}
        cbks.begin_eval(state)
        self._run_epoch(iter(ds_val), val_steps, cbks, 'eval')
        cbks.after_eval(state)

    def predict(self, test_loader):
        pass

    def reduce_loss(self, per_example_loss):
        loss = tf.reduce_mean(per_example_loss)
        if self._strategy:
            loss = loss / self._strategy.num_replicas_in_sync
        return loss

    def minimize(self, tape, optimizer, loss, trainable_variables):
        grad_clip_norm = self.grad_clip_norm
        grads = tape.gradient(loss, trainable_variables)
        if self.dtype == tf.float16:
            grads = optimizer.get_unscaled_gradients(grads)
        aggregate_grads_outside_optimizer = grad_clip_norm and is_distribute_strategy(self._strategy)

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

    def save(self):
        files = list(eglob(self.work_dir, "ckpt.*"))
        if len(files) != 0:
            for f in files:
                f.write_bytes(b'')
                rm(f)

        save_path = str(self.work_dir / "ckpt")
        ckpt, ckpt_options = self._make_ckpt()
        path = ckpt.write(save_path, ckpt_options)
        print('Save learner to %s' % path)

    def load(self, fp=None, miss_ok=False):
        if fp is None:
            fp = find_most_recent(self.work_dir, "ckpt.index")
            if fp is None:
                if miss_ok:
                    print("No checkpoint in %s" % self.work_dir)
                    return
                else:
                    raise FileNotFoundError("No checkpoint to load in %s" % self.work_dir)
            fp = str(fp)[:-6]
        ckpt, ckpt_options = self._make_ckpt()
        ckpt.restore(fp, ckpt_options)
        self.set_global_state('epoch', self.epoch)
        print("Load learner at epoch %d from %s" % (self.epoch + 1, fp))


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