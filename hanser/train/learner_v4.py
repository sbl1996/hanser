import pickle

import tensorflow as tf
from tensorflow.keras.metrics import Mean
from packaging.version import parse as vparse

from hanser.distribute import is_distribute_strategy
from hanser.train.learner import find_most_recent, cast, parse_freq
from hanser.train.metric_history import MetricHistory

from hhutil.io import time_now, eglob, fmt_path, rm


def _is_per_replica_instance(obj):
    return (isinstance(obj, tf.distribute.DistributedValues) and
            isinstance(obj, tf.__internal__.CompositeTensor))


def reduce_per_replica(values, strategy, reduction='first'):
    def _reduce(v):
        if not _is_per_replica_instance(v):
            return v
        elif reduction == 'first':
            return strategy.unwrap(v)[0]
        elif reduction == 'concat':
            return tf.concat(strategy.experimental_local_results(v), axis=0)
        else:
            raise ValueError('`reduction` must be "first" or "concat". Received: '
                             f'reduction={reduction}.')

    return tf.nest.map_structure(_reduce, values)


class TrainableModel(tf.keras.Model):

    def __init__(self, model, criterion, train_metrics, eval_metrics, output_transform=lambda x: x):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.train_metrics = train_metrics
        self.eval_metrics = eval_metrics
        self.output_transform = output_transform

        self._num_replicas = tf.distribute.get_strategy().num_replicas_in_sync

    def update_metrics(self, metrics, y_true, y_pred, per_example_loss=None):
        y_pred = self.output_transform(y_pred)
        for name, metric in metrics.items():
            if 'loss' in name and type(metric) == Mean:
                metric.update_state(per_example_loss)
            else:
                metric.update_state(y_true, y_pred, None)

    def reduce_loss(self, per_example_loss):
        loss = tf.reduce_mean(per_example_loss)
        if self._num_replicas > 1.0:
            loss = loss / self._num_replicas
        return loss

    def train_step(self, batch):
        x, y = batch
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            y_pred = cast(y_pred, tf.float32)
            per_example_loss = self.criterion(y, y_pred)
            loss = self.reduce_loss(per_example_loss)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.update_metrics(self.train_metrics, y, y_pred, per_example_loss)
        return {k: m.result() for k, m in self.train_metrics.items()}

    def test_step(self, batch):
        x, y = batch
        y_pred = self.model(x, training=False)
        y_pred = cast(y_pred, tf.float32)
        self.update_metrics(self.eval_metrics, y, y_pred)
        return {k: m.result() for k, m in self.eval_metrics.items()}

    def local_eval_step(self, batch):
        x, y = batch
        y_pred = self.model(x, training=False)
        y_pred = cast(y_pred, tf.float32)
        return y, y_pred

    @tf.function
    def _local_eval_step(self, batch):
        outputs = self.distribute_strategy.run(self.local_eval_step, args=(batch,))
        outputs = reduce_per_replica(
            outputs, self.distribute_strategy, reduction='concat')
        return outputs



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


class TrainableController(tf.keras.callbacks.Callback):

    def __init__(self, learner, ds_val=None, val_steps=None, val_freq=1, save_freq=None,
                 local_eval_metrics=None, local_eval_freq=None):
        super().__init__()
        self.learner = learner

        self.ds_val = ds_val
        self.val_steps = val_steps
        self.val_freq = val_freq
        self.save_freq = save_freq
        self.local_eval_metrics = local_eval_metrics
        self.local_eval_freq = local_eval_freq

        self._print = self.learner._print

    def on_train_begin(self, logs=None):
        train_start = time_now()
        self._print(f"{train_start} Start training")

        if self.learner._train_start is None:
            self.learner._train_start = train_start

    def on_epoch_begin(self, epoch, logs=None):
        self.learner._epoch = epoch
        self._print("Epoch %d/%d" % (epoch + 1, self.learner._max_epochs))
        for m in self.model.train_metrics.values():
            m.reset_states()

    def on_epoch_end(self, epoch, logs=None):

        log_metrics('train', logs, self.learner.epoch, self.learner._writer, self.learner.metric_history,
                    print_fn=self._print)

        if self.save_freq and (epoch + 1) % self.save_freq == 0:
            self.learner.save()
            # This prevents state mismatch when training stop at eval
            self.learner.save_state()

        do_local_eval = self.local_eval_metrics and parse_freq(epoch, self.local_eval_freq)
        do_eval = self.ds_val is not None and (not do_local_eval) and parse_freq(epoch, self.val_freq)

        if do_eval:
            self.learner.evaluate(self.ds_val, self.val_steps)

            if self.save_freq and (epoch + 1) % self.save_freq == 0:
                self.learner.save_state()

        if do_local_eval:
            self.learner.evaluate_local(self.ds_val, self.val_steps, self.local_eval_metrics)



class SuperLearner:

    def __init__(self, model, criterion, optimizer, train_metrics, eval_metrics,
                 steps_per_loop=None, jit_compile=True, output_transform=lambda x: x, work_dir=None):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_metrics = train_metrics
        self.eval_metrics = eval_metrics
        assert steps_per_loop is not None, "You must set steps_per_loop for LearnerV4"
        self.steps_per_loop = steps_per_loop
        self.jit_compile = jit_compile
        self.output_transform = output_transform
        self.work_dir = fmt_path(work_dir)

        self._trainable = TrainableModel(model, criterion, train_metrics, eval_metrics, output_transform)
        if vparse(tf.__version__) >= vparse("2.8"):
            self._trainable.compile(optimizer, steps_per_execution=steps_per_loop, jit_compile=self.jit_compile)
        else:
            self._trainable.compile(optimizer, steps_per_execution=steps_per_loop)

        self._verbose = True
        self._epoch = -1

        self.metric_history = MetricHistory(["train", "eval", "test"])
        self._writer = None
        self._train_start = None
        self._max_epochs = None

        self._terminated = False

    def fit(self, ds_train, epochs, ds_val=None, val_freq=1,
            steps_per_epoch=None, val_steps=None, max_epochs=None, save_freq=None,
            local_eval_metrics=None, local_eval_freq=None):

        if max_epochs is None:
            max_epochs = epochs

        self._max_epochs = max_epochs
        start_epoch = self.epoch + 1

        controller = TrainableController(
            self, ds_val, val_steps, val_freq, save_freq, local_eval_metrics, local_eval_freq)

        self._trainable.fit(
            ds_train, epochs=max_epochs, steps_per_epoch=steps_per_epoch, verbose=0,
            initial_epoch=start_epoch,  callbacks=[controller])

    def evaluate(self, ds_val, val_steps=None):
        val_steps = val_steps or len(ds_val)
        for m in self._trainable.eval_metrics.values():
            m.reset_states()
        eval_logs = self._trainable.evaluate(
            ds_val, steps=val_steps, verbose=0,
            return_dict=True, _use_cached_eval_dataset=True)
        log_metrics('eval', eval_logs, self.epoch, self._writer, self.metric_history,
                    stage_name='valid', print_fn=self._print)

    def evaluate_local(self, ds_val, steps, metrics):
        iterator = iter(ds_val)
        for m in metrics.values():
            m.reset_states()
        for step in range(steps):
            y_true, y_pred = self._trainable._local_eval_step(next(iterator))
            for m in metrics.values():
                m.update_state(y_true, y_pred, None)
        metric_results = {}
        for k, m in metrics.items():
            metric_results[k] = m.result().numpy()
        log_metrics('eval', metric_results, self.epoch, stage_name='valid',
                    metric_history=self.metric_history, print_fn=self._print)

    def _print(self, *args, **kwargs):
        if self._verbose:
            print(*args, **kwargs)

    @property
    def epoch(self):
        # Epoch is 0-based, not 1-based
        return self._epoch

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
        optimizers = [self.optimizer]
        if model_only:
            ckpt = tf.train.Checkpoint(model=self.model)
        else:
            ckpt = tf.train.Checkpoint(
                model=self.model, optimizers=optimizers)
        ckpt_options = tf.train.CheckpointOptions(
            experimental_io_device="/job:localhost") if is_distribute_strategy(tf.distribute.get_strategy()) else None
        return ckpt, ckpt_options

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
            self._epoch = epoch
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