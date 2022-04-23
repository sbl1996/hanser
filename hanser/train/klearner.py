from typing import Sequence

import pickle
import tensorflow as tf
from tensorflow.keras.metrics import Mean
import tensorflow.keras.mixed_precision as mixed_precision

from hanser.distribute import reduce_per_replica, is_distribute_strategy
from hanser.train.callbacks import config_callbacks
from hanser.train.learner import parse_freq, find_most_recent
from hanser.train.metric_history import MetricHistory

from hhutil.io import time_now, fmt_path, rm, eglob


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

    def __init__(self, model, criterion, optimizers, train_metrics, eval_metrics,
                 steps_per_loop=-1, eval_steps_per_loop=None, jit_compile=True, work_dir=None):
        super().__init__()
        if not isinstance(optimizers, Sequence):
            optimizers = [optimizers]
        optimizers = list(optimizers)

        work_dir = fmt_path(work_dir)

        self.model = model
        self.criterion = criterion
        self.optimizers = optimizers
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

        self._writer = None

        self._verbose = True
        self._state = {
            "train": {},
            "eval": {},
            "test": {},
        }

        self.metric_history = MetricHistory(["train", "eval", "test"])
        self._train_start = None
        self._max_epochs = None

        self._terminated = False
        self.set_global_state("epoch", -1)

    def train_step(self, batch):
        model = self.model
        optimizer = self.optimizers[0]

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
        if hasattr(self, "_ema") and self._ema is not None:
            self._ema.apply(self._ema_vars)
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

    @property
    def epoch(self):
        # Epoch is 0-based, not 1-based
        return self._state['train']['epoch']

    def init_state(self, mode, epochs=None):
        if mode == 'eval':
            if 'step' not in self._state['eval']:
                self.set_state('step', 0, 'eval')
            if 'epoch' not in self._state['eval']:
                self.set_state('epoch', 0, 'eval')
            if 'epochs' not in self._state['eval']:
                self.set_state('epochs', epochs or 0, 'eval')
        elif mode == 'train':
            self.set_global_state('epochs', epochs)
            self.set_global_state('step', 0)

    def set_state(self, k, v, mode):
        # State
        # epoch: int, for save and load
        # epochs: int
        # step: int
        # steps: int
        self._state[mode][k] = v

    def set_global_state(self, k, v):
        modes = ['train', 'eval', 'test']
        for m in modes:
            self.set_state(k, v, m)

    def _print(self, *args, **kwargs):
        if self._verbose:
            print(*args, **kwargs)

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

    def _run_eval(self, iterator, steps):
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
            steps_per_epoch=None, val_steps=None, max_epochs=None, save_freq=None,
            callbacks=None):

        if max_epochs is None:
            max_epochs = epochs
        self._max_epochs = max_epochs

        steps_per_epoch = steps_per_epoch or len(ds_train)
        train_iter = iter(ds_train)

        if ds_val is not None:
            val_steps = val_steps or len(ds_val)
            eval_iter = iter(ds_val)

        self.init_state('train', epochs=max_epochs)
        cbks = config_callbacks(
            self, callbacks, save_freq=save_freq, mode='train')

        start_epoch = self.epoch + 1

        train_start = time_now()
        self._print(f"{train_start} Start training")

        if self._train_start is None:
            self._train_start = train_start

        cbks.begin_train(self._state['train'])
        max_epochs = min(start_epoch + epochs, self._max_epochs)
        for epoch in range(start_epoch, max_epochs):
            self.set_global_state("epoch", epoch)
            state = self._state['train']
            state['metrics'] = {}

            cbks.begin_epoch(state)

            metric_results = self.train_epoch(train_iter, steps_per_epoch)
            for name, result in metric_results.items():
                state['metrics'][name] = result

            cbks.after_epoch(state)

            do_eval = ds_val is not None and parse_freq(epoch, val_freq)

            if do_eval:
                state = self._state['eval']
                state['metrics'] = {}
                cbks.begin_eval(state)

                metric_results = self._run_eval(eval_iter, val_steps)
                for name, result in metric_results.items():
                    state['metrics'][name] = result

                cbks.after_eval(state)

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
        self._run_eval(iter(ds_val), val_steps)
        cbks.after_eval(state)

    def save_state(self, save_dir=None):
        save_dir = save_dir or self.work_dir
        with open(save_dir / "learner_state.pickle", "wb") as f:
            pickle.dump({
                "metric_history": self.metric_history._history,
                "train_start": self._train_start,
                "epoch": self.epoch,
                "max_epochs": self._max_epochs,
            }, f)

    def load_state(self, save_dir=None):
        save_dir = save_dir or self.work_dir
        state_file = save_dir / "learner_state.pickle"
        if state_file.exists():
            with open(state_file, "rb") as f:
                d = pickle.load(f)
            return d
        else:
            return None

    def _make_ckpt(self, model_only=False):
        if model_only:
            ckpt = tf.train.Checkpoint(model=self.model)
        else:
            ckpt = tf.train.Checkpoint(
                model=self.model, optimizers=self.optimizers)
        ckpt_options = tf.train.CheckpointOptions(
            experimental_io_device="/job:localhost") if is_distribute_strategy(self._distribute_strategy) else None
        return ckpt, ckpt_options

    def save(self, save_dir=None, model_only=False, state=True):
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

        if state:
            self.save_state(save_dir)

        self._print('Save learner to %s' % path)

    def load(self, fp=None, miss_ok=False, model_only=False):
        if fp is None:
            fp = find_most_recent(self.work_dir, "ckpt.index")
            if fp is None:
                if miss_ok:
                    self._print("No checkpoint in %s" % self.work_dir)
                    return False
                else:
                    raise FileNotFoundError("No checkpoint to load in %s" % self.work_dir)
            fp = str(fp)[:-6]
        ckpt, ckpt_options = self._make_ckpt(model_only=model_only)
        ckpt.restore(fp, ckpt_options)

        save_dir = fmt_path(fp).parent
        d = self.load_state(save_dir)
        if d is not None:
            self.metric_history._history = d['metric_history']
            self._train_start = d['train_start']
            self._max_epochs = d['max_epochs']
            epoch = d['epoch']
            self.set_global_state('epoch', epoch)
            self._print("Load learner at epoch %d from %s" % (self.epoch + 1, fp))
        else:
            self._print("Load learner from %s" % (fp,))
        return True

    def recover_log(self):
        train_start = self._train_start
        self._print(f"{train_start} Start training")
        max_epochs = self._max_epochs
        train_metric_keys = self.train_metrics.keys()
        eval_metric_keys = self.eval_metrics.keys()
        for epoch in range(max_epochs):

            m = self.metric_history.get_epochs(epoch, epoch)

            train_metrics = {**m['train']}
            if 'end' not in train_metrics:
                break
            print("Epoch %d/%d" % (epoch + 1, max_epochs))
            train_end = train_metrics.pop("end")
            train_metric_logs = ", ".join(
                f"{k}: {train_metrics[k]:.4f}" for k in train_metric_keys)
            print(f"{train_end} train - {train_metric_logs}")

            eval_metrics = {**m['eval']}
            if 'end' in eval_metrics:
                eval_end = eval_metrics.pop("end")
                eval_metric_logs = ", ".join(
                    f"{k}: {eval_metrics[k]:.4f}" for k in eval_metric_keys)
                print(f"{eval_end} valid - {eval_metric_logs}")