import warnings

from toolz import curry
import numpy as np
from hhutil.io import time_now

import tensorflow as tf
from hanser.models.modules import DropPath, DropBlock
from hanser.train.ema import get_ema_vars


@curry
def log_metrics(stage, metrics, epoch, writer=None, metric_history=None, stage_name=None, print_fn=print):
    stage_name = stage_name or stage
    end_at = time_now()
    log_str = "%s %s - " % (end_at, stage_name)
    metric_logs = []
    for k, v in metrics.items():
        metric_logs.append("%s: %.4f" % (k, v))
        if writer:
            writer.add_scalar("%s/%s" % (k, stage), v, epoch)
        if metric_history:
            metric_history.record(stage, epoch, k, v)
    if metric_history:
        metric_history.record(stage, epoch, "end", end_at)
    log_str += ", ".join(metric_logs)
    print_fn(log_str)


def config_callbacks(
    learner,
    callbacks=None,
    save_freq=None,
    mode='train',
):
    cbks = callbacks or []
    cbks = cbks if isinstance(cbks, (list, tuple)) else [cbks]
    cbks = [*cbks]

    if mode == 'train':
        if not any(isinstance(k, TrainEvalLogger) for k in cbks):
            cbks.insert(0, TrainEvalLogger(print_fn=learner._print))
        if not any(isinstance(k, TerminateOnNaN) for k in cbks):
            ckpt_idx = None
            for i, k in enumerate(cbks):
                if isinstance(k, ModelCheckpoint):
                    ckpt_idx = i
                    break
            if ckpt_idx is None:
                cbks.append(TerminateOnNaN())
            else:
                cbks.insert(ckpt_idx, TerminateOnNaN())
        if not any(isinstance(k, ModelCheckpoint) for k in cbks) and save_freq:
            cbks.append(ModelCheckpoint(save_freq))
    else:
        if not any(isinstance(k, EvalLogger) for k in cbks):
            cbks.insert(0, EvalLogger(print_fn=learner._print))
    cbks = sorted(cbks, key=lambda c: -c.priority)
    cbk_list = CallbackList(learner, cbks)
    return cbk_list


class CallbackList(object):
    def __init__(self, learner, callbacks=None):
        self.learner = learner
        self.callbacks = [c for c in callbacks]
        for c in self.callbacks:
            c.learner = learner
            c.init()

    def append(self, callback):
        callback.learner = self.learner
        callback.init()
        self.callbacks.append(callback)

    def __iter__(self):
        return iter(self.callbacks)

    def begin_train(self, state):
        for c in self.callbacks:
            c.begin_train(state)

    def begin_epoch(self, state):
        for c in self.callbacks:
            c.begin_epoch(state)

    def begin_batch(self, state):
        for c in self.callbacks:
            c.begin_batch(state)

    def after_pred(self, state):
        for c in self.callbacks:
            c.after_pred(state)

    def after_loss(self, state):
        for c in self.callbacks:
            c.after_loss(state)

    def after_back(self, state):
        for c in self.callbacks:
            c.after_back(state)

    def after_step(self, state):
        for c in self.callbacks:
            c.after_step(state)

    def after_batch(self, state):
        for c in self.callbacks:
            c.after_batch(state)

    def after_epoch(self, state):
        for c in self.callbacks:
            c.after_epoch(state)

    def begin_eval(self, state):
        for c in self.callbacks:
            c.begin_eval(state)

    def after_eval(self, state):
        for c in self.callbacks:
            c.after_eval(state)

    def after_train(self, state):
        for c in self.callbacks:
            c.after_train(state)


class Callback(object):

    priority = 0

    def __init__(self):
        self.learner = None

    def init(self):
        pass

    def begin_train(self, state):
        pass

    def begin_epoch(self, state):
        pass

    def begin_batch(self, state):
        pass

    def after_pred(self, state):
        pass

    def after_loss(self, state):
        pass

    def after_back(self, state):
        pass

    def after_step(self, state):
        pass

    def after_batch(self, state):
        pass

    def after_epoch(self, state):
        pass

    def begin_eval(self, state):
        pass

    def after_eval(self, state):
        pass

    def after_train(self, state):
        pass

class ModelCheckpoint(Callback):

    def __init__(self, save_freq=1):
        super().__init__()
        self.save_freq = save_freq

    def after_epoch(self, state):
        epoch = self.learner.epoch + 1
        if epoch % self.save_freq == 0:
            self.learner.save()

    def after_eval(self, state):
        epoch = self.learner.epoch + 1
        if epoch % self.save_freq == 0:
            self.learner.save_state()

class TrainEvalLogger(Callback):

    def __init__(self, print_fn=print):
        super().__init__()
        self.print_fn = print_fn

    def begin_epoch(self, state):
        self.print_fn("Epoch %d/%d" % (self.learner.epoch + 1, state['epochs']))

    def after_epoch(self, state):
        learner = self.learner
        log_metrics('train', state['metrics'], learner.epoch, learner._writer, learner.metric_history,
                    print_fn=self.print_fn)

    def after_eval(self, state):
        learner = self.learner
        log_metrics('eval', state['metrics'], learner.epoch, learner._writer, learner.metric_history,
                    stage_name='valid', print_fn=self.print_fn)


class EvalLogger(Callback):

    def __init__(self, print_fn=print):
        super().__init__()
        self.print_fn = print_fn

    def after_eval(self, state):
        learner = self.learner
        log_metrics('eval', state['metrics'], learner.epoch, learner._writer, learner.metric_history,
                    stage_name='valid', print_fn=self.print_fn)


class EMA(Callback):

    def __init__(self, decay, num_updates=None, zero_debias=False):
        super().__init__()
        self.decay = decay
        self.num_updates = num_updates
        self.zero_debias = zero_debias

    def init(self):
        self._ema = tf.train.ExponentialMovingAverage(
            decay=self.decay, num_updates=self.num_updates, zero_debias=self.zero_debias)
        self._ema_vars = get_ema_vars(self.learner.model)
        self.learner._ema = self._ema
        self.learner._ema_vars = self._ema_vars

    def begin_eval(self, state):
        self.swap_weights()

    def after_eval(self, state):
        self.swap_weights()

    def swap_weights(self):
        """Swap the average and moving weights.

        This is a convenience method to allow one to evaluate the averaged weights
        at test time. Loads the weights stored in `self._average_weights` into the model,
        keeping a copy of the original model weights. Swapping twice will return
        the original weights.
        """
        if tf.distribute.in_cross_replica_context():
            strategy = tf.distribute.get_strategy()
            strategy.run(self._swap_weights_dist, args=())
        else:
            self._swap_weights_local()

    @tf.function
    def _swap_weights_local(self):
        model_vars = self._ema_vars
        avg_vars = [self._ema._averages[var.ref()] for var in self._ema_vars]
        for a, b in zip(avg_vars, model_vars):
            a.assign_add(b)
            b.assign(a - b)
            a.assign_sub(b)

    @tf.function
    def _swap_weights_dist(self):
        def fn_0(a, b):
            return a.assign_add(b)

        def fn_1(b, a):
            return b.assign(a - b)

        def fn_2(a, b):
            return a.assign_sub(b)

        def swap(strategy, a, b):
            """Swap `a` and `b` and mirror to all devices."""
            for a_element, b_element in zip(a, b):
                strategy.extended.update(
                    a_element, fn_0, args=(b_element,)
                )  # a = a + b
                strategy.extended.update(
                    b_element, fn_1, args=(a_element,)
                )  # b = a - b
                strategy.extended.update(
                    a_element, fn_2, args=(b_element,)
                )  # a = a - b

        model_vars = self._ema_vars
        avg_vars = [self._ema._averages[var.ref()] for var in self._ema_vars]
        ctx = tf.distribute.get_replica_context()
        return ctx.merge_call(swap, args=(avg_vars, model_vars))


class DropPathRateSchedule(Callback):

    def __init__(self, drop_path):
        super().__init__()
        warnings.warn(
            "DropPathRateSchedule will be deprecated, use DropPathRateScheduleV2 instead.",
            DeprecationWarning,
        )
        self.drop_path = drop_path

    def begin_epoch(self, state):
        epoch = self.learner.epoch
        epochs = state['epochs']
        rate = (epoch + 1) / epochs * self.drop_path
        for l in self.learner.model.submodules:
            if isinstance(l, DropPath):
                l.rate.assign(rate)


class DropPathRateScheduleV2(Callback):

    def __init__(self):
        super().__init__()

    def begin_epoch(self, state):
        epoch = self.learner.epoch
        epochs = state['epochs']
        for l in self.learner.model.submodules:
            if isinstance(l, DropPath):
                if epoch == 0:
                    rate = 1 / epochs * l.rate
                else:
                    rate = (epoch + 1) / epoch * l.rate
                l.rate.assign(rate)


class DropBlockSchedule(Callback):

    def __init__(self):
        super().__init__()

    def begin_epoch(self, state):
        epoch = self.learner.epoch
        epochs = state['epochs']
        for l in self.learner.model.submodules:
            if isinstance(l, DropBlock):
                if epoch == 0:
                    keep_prob = 1. - 1 / epochs * (1. - l.keep_prob)
                else:
                    keep_prob = 1. - (epoch + 1) / epoch * (1. - l.keep_prob)
                l.keep_prob.assign(keep_prob)


class EvalEveryAfter(Callback):

    def __init__(self, eval_after):
        super().__init__()
        self.eval_after = eval_after

    def begin_epoch(self, state):
        if self.learner.epoch >= self.eval_after:
            self.learner._val_freq = 1


class SetEvalFreq(Callback):

    def __init__(self, after_epoch, eval_freq=1):
        super().__init__()
        self._after_epoch = after_epoch
        self._eval_freq = eval_freq

    def begin_epoch(self, state):
        if self.learner.epoch >= self._after_epoch:
            self.learner._val_freq = self._eval_freq


class TerminateOnNaN(Callback):
    def after_epoch(self, state):
        if not np.isfinite(state['metrics']['loss']):
            raise RuntimeError("Infinite encountered")


class NNIReportIntermediateResult(Callback):

    def __init__(self, metric):
        super().__init__()
        self.metric = metric

    def after_epoch(self, state):
        epoch = self.learner.epoch
        val_metric = self.learner.metric_history.get_metric(self.metric, "eval", epoch, epoch)
        if val_metric:
            import nni
            nni.report_intermediate_result(val_metric)

    def after_eval(self, state):
        final_metric = self.learner.metric_history.get_metric(self.metric, "eval")[-1]
        import nni
        nni.report_final_result(final_metric)


class OptunaReportIntermediateResult(Callback):

    def __init__(self, metric, trial):
        super().__init__()
        self.metric = metric
        self.trial = trial

    def after_eval(self, state):
        epoch = self.learner.epoch
        val_metric = self.learner.metric_history.get_metric(self.metric, "eval", epoch, epoch)
        if val_metric:
            self.trial.report(val_metric, epoch)
        if self.trial.should_prune():
            self.learner._terminated = True
            import optuna
            raise optuna.TrialPruned()
