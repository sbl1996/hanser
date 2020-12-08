from toolz import curry
import numpy as np
from hhutil.io import time_now
from tensorflow_addons.optimizers import MovingAverage


@curry
def log_metrics(stage, metrics, epoch, writer, metric_history, stage_name=None):
    stage_name = stage_name or stage
    log_str = "%s %s - " % (time_now(), stage_name)
    metric_logs = []
    for k, v in metrics.items():
        metric_logs.append("%s: %.4f" % (k, v))
        if writer:
            writer.add_scalar("%s/%s" % (k, stage), v, epoch)
        metric_history.record(stage, epoch, k, v)
    log_str += ", ".join(metric_logs)
    print(log_str)


def config_callbacks(
    learner,
    callbacks=None,
    save_freq=None,
    mode='train',
):
    cbks = callbacks or []
    cbks = cbks if isinstance(cbks, (list, tuple)) else [cbks]

    if mode == 'train':
        if not any(isinstance(k, TrainEvalLogger) for k in cbks):
            cbks = [TrainEvalLogger()] + cbks
        if not any(isinstance(k, ModelCheckpoint) for k in cbks) and save_freq:
            cbks = cbks + [ModelCheckpoint(save_freq)]
    else:
        if not any(isinstance(k, EvalLogger) for k in cbks):
            cbks = [EvalLogger()] + cbks

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
        epoch = state['epoch'] + 1
        if epoch % self.save_freq == 0:
            self.learner.save()


class TrainEvalLogger(Callback):

    def __init__(self):
        super().__init__()
        self.verbose = True

    def _is_print(self):
        return self.verbose

    def begin_epoch(self, state):
        if self._is_print():
            print("Epoch %d/%d" % (state['epoch'] + 1, state['epochs']))

    def after_epoch(self, state):
        learner = self.learner
        if self._is_print():
            log_metrics('train', state['metrics'], state['epoch'].numpy(), learner._writer, learner.metric_history)

    def after_eval(self, state):
        learner = self.learner
        if self._is_print():
            log_metrics('eval', state['metrics'], state['epoch'].numpy(), learner._writer, learner.metric_history,
                        stage_name='valid')


class EvalLogger(Callback):

    def __init__(self):
        super().__init__()
        self.verbose = True

    def _is_print(self):
        return self.verbose

    def after_eval(self, state):
        learner = self.learner
        if self._is_print():
            log_metrics('eval', state['metrics'], state['epochs'], learner._writer, learner.metric_history,
                        stage_name='valid')


class EMA(Callback):

    def __init__(self, decay=None):
        super().__init__()
        self.decay = decay

    def init(self):
        if self.decay is not None:
            self.original_optimizer = self.learner.optimizers[0]
            self.learner.optimizers[0] = MovingAverage(
                self.original_optimizer, average_decay=self.decay)
        else:
            assert isinstance(self.learner.optimizers[0], MovingAverage)

    def begin_eval(self, state):
        opt: MovingAverage = self.learner.optimizers[0]
        opt.shadow_copy(self.learner.model.trainable_variables)
        opt.swap_weights()

    def after_eval(self, state):
        opt: MovingAverage = self.learner.optimizers[0]
        opt.swap_weights()


class DropPathRateSchedule(Callback):

    def __init__(self, drop_path):
        super().__init__()
        self.drop_path = drop_path

    def begin_epoch(self, state):
        epoch = state['epoch'].numpy()
        epochs = state['epochs']
        rate = (epoch - 1) / epochs * self.drop_path
        for l in self.learner.model.submodules:
            if 'drop' in l.name:
                l.rate.assign(rate)


class EvalEveryAfter(Callback):

    def __init__(self, eval_after):
        super().__init__()
        self.eval_after = eval_after

    def begin_epoch(self, state):
        if state['epoch'] >= self.eval_after:
            self.learner._val_freq = 1


class TerminateOnNaN(Callback):
    def after_epoch(self, state):
        if not np.isfinite(state['metrics']['loss']):
            raise RuntimeError("Infinite encountered")


# class ColabPushResult(Callback):

#     def after_train(self, state):


class NNIReportIntermediateResult(Callback):

    def __init__(self, metric):
        super().__init__()
        self.metric = metric

    def after_epoch(self, state):
        epoch = state['epoch'].numpy()
        val_metric = self.learner.metric_history.get_metric(self.metric, "eval", epoch, epoch)
        if val_metric:
            import nni
            nni.report_intermediate_result(val_metric)

    def after_train(self, state):
        final_metric = self.learner.metric_history.get_metric(self.metric, "eval")[-1]
        import nni
        nni.report_final_result(final_metric)