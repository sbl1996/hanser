from abc import ABCMeta
from bisect import bisect_right
from typing import Sequence, Mapping, Optional

import pickle

import tensorflow as tf
from packaging.version import parse as vparse
if vparse(tf.__version__) >= vparse("2.4"):
    import tensorflow.keras.mixed_precision as mixed_precision
else:
    import tensorflow.keras.mixed_precision.experimental as mixed_precision
from tensorflow.keras.metrics import Metric, Mean

from hhutil.io import fmt_path, eglob, rm, time_now

from hanser.distribute import parse_strategy, strategy_run, is_distribute_strategy, local_results, discover_device
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


def cast(xs, dtype, whiltelist=(tf.int32, tf.int64, tf.bool)):
    def func(x):
        if x.dtype != dtype and all(x.dtype != wdtype for wdtype in whiltelist):
            x = tf.cast(x, dtype)
        return x
    return tf.nest.map_structure(func, xs)


def steps_to_run(current_step, steps_per_epoch, steps_per_loop):
    """Calculates steps to run on device."""
    if steps_per_loop <= 0:
        raise ValueError('steps_per_loop should be positive integer.')
    if steps_per_loop == 1:
        return steps_per_loop
    remainder_in_epoch = current_step % steps_per_epoch
    if remainder_in_epoch != 0:
        return min(steps_per_epoch - remainder_in_epoch, steps_per_loop)
    else:
        return steps_per_loop


class Learner(metaclass=ABCMeta):

    def __init__(self, model, criterion, optimizers,
                 train_metrics: Mapping[str, Metric], eval_metrics: Mapping[str, Metric],
                 work_dir: str, output_transform=default_metric_transform,
                 n_batches_per_step: Optional[int] = None, steps_per_loop: Optional[int] = None,
                 jit_compile: bool = True, eval_steps_per_loop: Optional[int] = None):
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
            if vparse(tf.__version__) >= vparse("2.4"):
                dynamic = True
            else:
                dynamic = 'dynamic'
            self.optimizers = [
                mixed_precision.LossScaleOptimizer(optimizer, dynamic)
                if not isinstance(optimizer, mixed_precision.LossScaleOptimizer) else optimizer
                for optimizer in self.optimizers
            ]
        self.output_transform = output_transform

        device = discover_device()
        if steps_per_loop is None:
            steps_per_loop = -1 if device == 'TPU' else 1
        self.steps_per_loop = steps_per_loop
        self.jit_compile = jit_compile
        if eval_steps_per_loop is None:
            eval_steps_per_loop = -1 if device == 'TPU' else 1
        self.eval_steps_per_loop = eval_steps_per_loop

        self._log_dir = self.work_dir / "runs"
        self._writer = None

        self._verbose = True
        self._state = {
            "train": {},
            "eval": {},
            "test": {},
        }

        # epoch -> stage -> metric -> value
        # Epoch is 0-based
        self.metric_history = MetricHistory(["train", "eval", "test"])
        self._train_start = None
        self._max_epochs = None

        self._terminated = False
        self.set_global_state("epoch", -1)

        if self.jit_compile:
            self.train_batch = tf.function(self.train_batch, experimental_compile=True)

        self.n_batches_per_step = n_batches_per_step

    def _make_ckpt(self, model_only=False):
        optimizers = self.optimizers
        # if len(optimizers) == 1 and hasattr(self, "original_optimizer"):
        #     optimizers = [self.original_optimizer]
        if model_only:
            ckpt = tf.train.Checkpoint(model=self.model)
        else:
            ckpt = tf.train.Checkpoint(
                model=self.model, optimizers=optimizers)
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
            self.set_global_state("epochs", epochs)
            self.set_global_state("step", 0)

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

    def fit(self, ds_train, epochs, ds_val=None, val_freq=1,
            steps_per_epoch=None, val_steps=None, max_epochs=None, save_freq=None,
            callbacks=None, reuse_train_iterator=True,
            local_eval_metrics=None, local_eval_freq=None):
        if max_epochs is None:
            max_epochs = epochs

        self._max_epochs = max_epochs

        steps_per_epoch = steps_per_epoch or len(ds_train)

        if ds_val is not None:
            val_steps = val_steps or len(ds_val)

        self.init_state('train', epochs=max_epochs)
        cbks = config_callbacks(
            self, callbacks, save_freq=save_freq, mode='train')

        start_epoch = self.epoch + 1

        train_start = time_now()
        self._print(f"{train_start} Start training")

        if self._train_start is None:
            self._train_start = train_start

        if reuse_train_iterator:
            self._train_it = iter(ds_train)

        cbks.begin_train(self._state['train'])
        max_epochs = min(start_epoch + epochs, self._max_epochs)
        for epoch in range(start_epoch, max_epochs):

            self.set_global_state("epoch", epoch)

            state = self._state['train']
            state['metrics'] = {}
            cbks.begin_epoch(state)

            if not reuse_train_iterator:
                self._train_it = iter(ds_train)

            self._run_epoch(self._train_it, steps_per_epoch, cbks)
            cbks.after_epoch(state)

            do_local_eval = local_eval_metrics and parse_freq(epoch, local_eval_freq)
            do_eval = ds_val is not None and (not do_local_eval) and parse_freq(epoch, val_freq)

            if do_eval:
                state = self._state['eval']
                state['metrics'] = {}
                cbks.begin_eval(state)
                self._run_eval(iter(ds_val), val_steps)
                cbks.after_eval(state)

            if do_local_eval:
                self.evaluate_local(iter(ds_val), val_steps, local_eval_metrics)

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

    def evaluate_local(self, iterator, steps, metrics):
        for m in metrics.values():
            m.reset_states()
        for step in range(steps):
            y_true, y_pred = self._local_eval_step(next(iterator))
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
            strategy_run(self._strategy, self.local_eval_batch, (batch,)), self._strategy)

    @tf.function
    def _run_steps(self, step_fn, iterator, n_batches_per_step, n_steps):
        for i in tf.range(n_steps):
            if n_batches_per_step is not None:
                batches = tuple(next(iterator) for bi in range(n_batches_per_step))
                step_fn(batches)
            else:
                batch = next(iterator)
                step_fn(batch)

    def _run_epoch(self, iterator, steps, callbacks):
        state = self._state['train']
        metrics = self.train_metrics
        steps_per_loop = self.steps_per_loop
        if steps_per_loop == -1:
            steps_per_loop = steps

        state.update({
            'steps': steps,
            'step': 0,
        })

        for metric in metrics.values():
            metric.reset_states()

        step_fn = self._train_step
        n_batches_per_step = self.n_batches_per_step
        if n_batches_per_step is not None:
            step_fn = self._train_step_on_batches

        current_step = 0
        while current_step < steps:
            callbacks.begin_batch(state)
            run_steps = steps_to_run(current_step, steps, steps_per_loop)
            self._run_steps(step_fn, iterator, n_batches_per_step,
                            tf.convert_to_tensor(run_steps, dtype=tf.int32))
            current_step += run_steps
            state['step'] = current_step
            callbacks.after_batch(state)

        for name, metric in metrics.items():
            state['metrics'][name] = metric.result().numpy()

    def _run_eval(self, iterator, steps):
        state = self._state['eval']
        metrics = self.eval_metrics
        steps_per_loop = self.eval_steps_per_loop
        if steps_per_loop == -1:
            steps_per_loop = steps

        state.update({
            'steps': steps,
            'step': 0,
        })

        for metric in metrics.values():
            metric.reset_states()

        current_step = 0
        while current_step < steps:
            run_steps = steps_to_run(current_step, steps, steps_per_loop)
            self._run_steps(self._eval_step, iterator, None,
                            tf.convert_to_tensor(run_steps, dtype=tf.int32))
            current_step += run_steps
            state['step'] = current_step

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

    def minimize(self, tape, optimizer, loss, trainable_variables, grad_clip_norm=None):
        grads = tape.gradient(loss, trainable_variables)
        if self.dtype == tf.float16:
            grads = optimizer.get_unscaled_gradients(grads)
        self.apply_gradients(optimizer, grads, trainable_variables, grad_clip_norm)

    def apply_gradients(self, optimizer, grads, vars, grad_clip_norm=None):
        aggregate_grads_outside_optimizer = grad_clip_norm and is_distribute_strategy(self._strategy)

        if aggregate_grads_outside_optimizer:
            grads = tf.distribute.get_replica_context().all_reduce('sum', grads)

        if grad_clip_norm:
            grads = tf.clip_by_global_norm(grads, grad_clip_norm)[0]
        if aggregate_grads_outside_optimizer:
            optimizer.apply_gradients(
                zip(grads, vars),
                experimental_aggregate_gradients=False)
        else:
            optimizer.apply_gradients(zip(grads, vars))


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