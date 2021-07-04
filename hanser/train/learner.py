from abc import ABCMeta
from bisect import bisect_right
from typing import Sequence, Mapping

from dateutil.parser import parse
from datetime import timedelta

import tensorflow as tf
import tensorflow.keras.mixed_precision.experimental as mixed_precision
from tensorflow.keras.metrics import Metric, Mean

from hhutil.io import fmt_path, eglob, rm, time_now

from hanser.distribute import parse_strategy, strategy_run, is_distribute_strategy, local_results
from hanser.train.metric_history import MetricHistory
from hanser.train.callbacks import config_callbacks, log_metrics


def validate_freq(freqs):
    if isinstance(freqs, list):
        assert all([isinstance(f, tuple) and len(f) == 2 for f in freqs])
        start_epochs = [f[0] for f in freqs]
        assert start_epochs[0] == 0
        for i in range(1, len(start_epochs)):
            assert start_epochs[i - 1] < start_epochs[i]


def parse_freq(epoch, freqs):
    # epochs is 0-based
    if freqs is None:
        return False
    if isinstance(freqs, int):
        return (epoch + 1) % freqs == 0
    if isinstance(freqs, list):
        start_epochs, freqs = zip(*freqs)
        i = bisect_right(start_epochs, epoch) - 1
        freq = freqs[i]
        if i != 0:
            epoch -= start_epochs[i-1]
        return (epoch + 1) % freq == 0
    return True


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

def is_global_bfloat16():
    return mixed_precision.global_policy().compute_dtype == 'bfloat16'


def is_global_float16():
    return mixed_precision.global_policy().compute_dtype == 'float16'


def cast(xs, dtype, whiltelist=(tf.int32, tf.int64, tf.bool)):
    def func(x):
        if x.dtype != dtype and all(x.dtype != wdtype for wdtype in whiltelist):
            x = tf.cast(x, dtype)
        return x
    return tf.nest.map_structure(func, xs)


class Learner(metaclass=ABCMeta):

    def __init__(self, model, criterion, optimizers,
                 train_metrics: Mapping[str, Metric],
                 eval_metrics: Mapping[str, Metric], work_dir,
                 grad_clip_norm=0.0, multiple_steps=True, xla_compile=False,
                 output_transform=default_metric_transform,
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
        self.output_transform = output_transform

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
        self.set_global_state("epoch", -1)
        self._epoch_var = tf.Variable(self.epoch, dtype=tf.int64)

        if self.xla_compile:
            self.train_batch = tf.function(self.train_batch, experimental_compile=True)

        if multiple_steps:
            self._run_steps = tf.function(self._run_steps)

        self.n_batches_per_step = n_batches_per_step

    def _make_ckpt(self, model_only=False):
        optimizers = self.optimizers
        # if len(optimizers) == 1 and hasattr(self, "original_optimizer"):
        #     optimizers = [self.original_optimizer]
        self._epoch_var.assign(self.epoch)
        if model_only:
            ckpt = tf.train.Checkpoint(
                model=self.model, epoch=self._epoch_var)
        else:
            ckpt = tf.train.Checkpoint(
                model=self.model, optimizers=optimizers,
                epoch=self._epoch_var,
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

    def local_eval_batch(self, batch):
        pass

    def test_batch(self, batch):
        pass

    @property
    def epoch(self):
        return self._state['train']['epoch']

    def init_state(self, mode, epochs=None):
        if mode == 'eval':
            if 'step' not in self._state['eval']:
                self.set_state('step', tf.Variable(0, dtype=tf.int32), 'eval')
            if 'epoch' not in self._state['eval']:
                self.set_state('epoch', 0, 'eval')
            if 'epochs' not in self._state['eval']:
                self.set_state('epochs', epochs or 0, 'eval')
        elif mode == 'train':
            self.set_global_state("epochs", epochs)
            self.set_global_state("step", tf.Variable(0, dtype=tf.int32))

    def set_state(self, k, v, mode):
        # State
        # epoch: int (Variable by _epoch_var), for save and load
        # epochs: int
        # step: Variable
        # steps: int
        if k in self._state[mode] and isinstance(self._state[mode][k], tf.Variable):
            self._state[mode][k].assign(v)
        else:
            self._state[mode][k] = v

    def set_global_state(self, k, v):
        modes = ['train', 'eval', 'test']
        for m in modes:
            self.set_state(k, v, m)

    def _print(self, *args, **kwargs):
        if self._verbose:
            print(*args, **kwargs)

    def fit(self, ds_train, max_epochs, ds_val=None, val_freq=1,
            steps_per_epoch=None, val_steps=None, save_freq=None, callbacks=None,
            reuse_train_iterator=True, local_eval_metrics=None, local_eval_freq=None):
        # It seems that reuse_train_iterator speed up the first epoch significantly

        steps_per_epoch = steps_per_epoch or len(ds_train)
        steps_per_epoch = tf.convert_to_tensor(steps_per_epoch, dtype=tf.int32)

        if ds_val is not None:
            val_steps = val_steps or len(ds_val)
            val_steps = tf.convert_to_tensor(val_steps, dtype=tf.int32)

        self.init_state('train', epochs=max_epochs)
        cbks = config_callbacks(
            self, callbacks, save_freq=save_freq, mode='train')

        start_epoch = self.epoch + 1

        self._print("%s Start training" % (time_now(),))

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

            do_local_eval = local_eval_metrics and parse_freq(epoch, local_eval_freq)
            do_eval = ds_val is not None and (not do_local_eval) and parse_freq(epoch, val_freq)

            if do_eval:
                state = self._state['eval']
                state['metrics'] = {}
                cbks.begin_eval(state)
                self._run_epoch(iter(ds_val), val_steps, cbks, 'eval')
                cbks.after_eval(state)

            if do_local_eval:
                self.evaluate_local(ds_val, val_steps, local_eval_metrics)

            if self._terminated:
                self._print("Terminated at epoch %d" % (epoch + 1))
                break
        cbks.after_train(self._state['train'])

    def evaluate(self, ds_val, val_steps=None, callbacks=None):
        self.init_state('eval')

        val_steps = val_steps or len(ds_val)
        cbks = config_callbacks(self, callbacks, mode='eval')

        state = self._state['eval']
        state['metrics'] = {}
        cbks.begin_eval(state)
        self._run_epoch(iter(ds_val), val_steps, cbks, 'eval')
        cbks.after_eval(state)

    def evaluate_local(self, iterator, steps, metrics):
        for m in metrics.values():
            m.reset_states()
        it = iter(iterator)
        for step in range(steps):
            y_true, y_pred = self._local_eval_step(next(it))
            for m in metrics.values():
                m.update_state(y_true, y_pred, None)
        metric_results = {}
        for k, m in metrics.items():
            metric_results[k] = m.result().numpy()
        log_metrics('eval', metric_results, self.epoch, stage_name='valid',
                    metric_history=self.metric_history, print_fn=self._print)

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
    def _local_eval_step(self, batch):
        return local_results(
            strategy_run(self._strategy, self._local_eval_step, (batch,)), self._strategy)

    def _run_steps(self, step_fn, iterator, n_batches_per_step, n_steps, callbacks, state):
        state['step'].assign(-1)
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

    def _run_epoch(self, iterator, steps, callbacks, mode):
        state = self._state[mode]
        metrics = getattr(self, mode + "_metrics")
        step_fn = getattr(self, f"_{mode}_step")

        state.update({
            'steps': steps,
        })

        for metric in metrics.values():
            metric.reset_states()

        if self.multiple_steps:
            run_state = {
                k: state[k] for k in ["step", "steps", "epochs"]
            }
        else:
            run_state = state
        if mode == 'train' and self.n_batches_per_step is not None:
            step_fn = self._train_step_on_batches

        self._run_steps(
            step_fn, iterator, self.n_batches_per_step, steps, callbacks, run_state)

        for name, metric in metrics.items():
            state['metrics'][name] = metric.result().numpy()

    def update_metrics(self, metrics, y_true, y_pred, per_example_loss=None):
        y_pred = self.output_transform(y_pred)
        for name, metric in metrics.items():
            if 'loss' in name and type(metric) == Mean:
                metric.update_state(per_example_loss)
            else:
                metric.update_state(y_true, y_pred, None)

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

    def save(self, save_dir=None, model_only=False):
        if save_dir is None:
            save_dir = self.work_dir
        else:
            save_dir = fmt_path(save_dir)
        files = list(eglob(save_dir, "ckpt.*"))
        if len(files) != 0:
            for f in files:
                f.write_bytes(b'')
                rm(f)

        save_path = str(save_dir / "ckpt")
        ckpt, ckpt_options = self._make_ckpt(model_only=model_only)
        path = ckpt.write(save_path, ckpt_options)
        self._print('Save learner to %s' % path)

    def load(self, fp=None, miss_ok=False, model_only=False):
        if fp is None:
            fp = find_most_recent(self.work_dir, "ckpt.index")
            if fp is None:
                if miss_ok:
                    self._print("No checkpoint in %s" % self.work_dir)
                    return
                else:
                    raise FileNotFoundError("No checkpoint to load in %s" % self.work_dir)
            fp = str(fp)[:-6]
        ckpt, ckpt_options = self._make_ckpt(model_only=model_only)
        ckpt.restore(fp, ckpt_options)
        self.set_global_state('epoch', int(self._epoch_var.numpy()))
        self._print("Load learner at epoch %d from %s" % (self.epoch + 1, fp))

    def recover_log(self, start, from_epochs, total_epochs, train_time, eval_time):
        """Recover log from speficic epoch.

        Example:
            >>> learner = Learner()
            >>> learner.recover_log("22:03:42", 38, 60, 68, 8)
        """
        start = parse(start)
        train_metric_keys = self.train_metrics.keys()
        eval_metric_keys = self.eval_metrics.keys()
        for i in range(from_epochs - 1, total_epochs):
            m = self.metric_history.get_epochs(i, i)
            train_end = start + timedelta(seconds=train_time)
            print("Epoch %d/%d" % (i + 1, total_epochs))
            train_metric_logs = ", ".join(
                f"{k}: {m['train'][k]:.4f}" for k in train_metric_keys if k in m['train'])
            print(f"{str(train_end)[-8:]} train - {train_metric_logs}")
            start = train_end

            valid_metric_logs = ", ".join(
                f"{k}: {m['eval'][k]:.4f}" for k in eval_metric_keys if k in m['eval'])
            if valid_metric_logs:
                eval_end = train_end + timedelta(seconds=eval_time)
                print(f"{str(eval_end)[-8:]} valid - {valid_metric_logs}")
                start = eval_end
