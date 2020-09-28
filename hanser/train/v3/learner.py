from abc import ABCMeta
from typing import Sequence, Mapping

import tensorflow as tf
import tensorflow.keras.mixed_precision.experimental as mixed_precision
from tensorflow.keras.metrics import Metric, Mean

from hhutil.io import fmt_path

from hanser.train.trainer import MetricHistory
from hanser.train.v3.callbacks import config_callbacks


@tf.function
def identity(x):
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


class Learner(metaclass=ABCMeta):

    def __init__(self, model, criterion, optimizers,
                 train_metrics: Mapping[str, Metric],
                 eval_metrics: Mapping[str, Metric], work_dir,
                 fp16=False, grad_clip_norm=0.0, multiple_steps=True, metric_transform=identity):
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
        self.fp16 = is_global_bfloat16() or is_global_float16()
        self.grad_clip_norm = grad_clip_norm
        self.multiple_steps = multiple_steps
        self.metric_transform = metric_transform

        self._use_weight_decay = len(model.losses) != 0

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

    def train_batch(self, batch):
        pass

    def eval_batch(self, batch):
        pass

    def test_batch(self, batch):
        pass

    def state_dict(self):
        return self._state

    def set_global_state(self, k, v):
        modes = ['train', 'eval', 'test']
        for m in modes:
            if k not in self._state[m] or not isinstance(self._state[m][k], tf.Variable):
                self._state[m][k] = v
            else:
                self._state[m][k].assign(v)

    def fit(self, ds_train, epochs, ds_val=None, val_freq=1,
            steps_per_epoch=None, val_steps=None, save_freq=None, callbacks=None):

        steps_per_epoch = steps_per_epoch or len(ds_train)

        do_eval = ds_val is not None
        if do_eval:
            self._val_freq = val_freq
            val_steps = val_steps or len(ds_val)

        cbks = config_callbacks(
            self,
            callbacks,
            save_freq=save_freq,
        )
        start_epoch = self._state['train'].get('epoch', 0)
        epochs = epochs + start_epoch
        self.set_global_state("epochs", epochs)
        self.set_global_state("epoch", tf.Variable(0, dtype=tf.int64))
        self.set_global_state("step", tf.Variable(0, dtype=tf.int64))

        cbks.begin_train(self._state['train'])
        for epoch in range(start_epoch, epochs):
            self.set_global_state("epoch", epoch)

            state = self._state['train']
            state['metrics'] = {}
            cbks.begin_epoch(state)
            self._run_epoch(ds_train, steps_per_epoch, cbks, 'train')
            cbks.after_epoch(state)

            if do_eval and (epoch + 1) % self._val_freq == 0:
                state = self._state['eval']
                state['metrics'] = {}
                cbks.begin_eval(state)
                self._run_epoch(ds_val, val_steps, cbks, 'eval')
                cbks.after_eval(state)

            if self._terminated:
                print("Terminated at epoch %d" % epochs)
                break
        cbks.after_train(self._state['train'])

    @tf.function
    def _train_step(self, batch):
        strategy_run(self._strategy, self.train_batch, (batch,))

    @tf.function
    def _eval_step(self, batch):
        strategy_run(self._strategy, self.eval_batch, (batch,))

    @tf.function
    def _run_steps(self, step_fn, iterator, callbacks, state):
        for batch in iterator:
            state['step'].assign_add(1)
            callbacks.begin_batch(state)
            step_fn(batch)
            callbacks.after_batch(state)

    def update_metrics(self, metrics, y_true, y_pred, per_example_loss=None):
        y_pred = self.metric_transform(y_pred)
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
            self._run_steps(step_fn, iterator, callbacks, sub_state)
        else:
            for batch in iterator:
                state['step'].assign_add(1)
                callbacks.begin_batch(state)
                step_fn(batch)
                callbacks.after_batch(state)

        for name, metric in metrics.items():
            state['metrics'][name] = metric.result()

    def evaluate(self, ds_val, val_steps=None, callbacks=None):
        val_steps = val_steps or len(ds_val)
        cbks = config_callbacks(self, callbacks, mode='eval')

        state = self._state['eval']
        state['metrics'] = {}
        cbks.begin_eval(state)
        self._run_epoch(ds_val, val_steps, cbks, 'eval')
        cbks.after_eval(state)

    def predict(self, test_loader):
        pass

    def reduce_loss(self, per_example_loss):
        loss = tf.reduce_mean(per_example_loss)
        if self._use_weight_decay:
            loss = loss + tf.add_n(self.model.losses)
        if self._strategy:
            loss = loss / self._strategy.num_replicas_in_sync
        return loss

    def minimize(self, tape, optimizer, loss, trainable_variables):
        grad_clip_norm = self.grad_clip_norm
        grads = tape.gradient(loss, trainable_variables)
        aggregate_grads_outside_optimizer = grad_clip_norm and is_tpu_strategy(self._strategy)

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
