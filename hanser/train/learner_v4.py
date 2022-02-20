import tensorflow as tf
from tensorflow.keras.metrics import Mean
from packaging.version import parse as vparse

from hhutil.io import time_now

class TrainableModel(tf.keras.Model):

    def __init__(self, model, train_metrics, eval_metrics, output_transform=lambda x: x):
        super().__init__()
        self.model = model
        self.train_metrics = train_metrics
        self.eval_metrics = eval_metrics
        self.output_transform = output_transform

    def update_metrics(self, metrics, y_true, y_pred, per_example_loss=None):
        y_pred = self.output_transform(y_pred)
        for name, metric in metrics.items():
            if 'loss' in name and type(metric) == Mean:
                num_replicas = tf.distribute.get_strategy().num_replicas_in_sync
                if num_replicas > 1.0:
                    per_example_loss = per_example_loss * per_example_loss
                metric.update_state(per_example_loss)
            else:
                metric.update_state(y_true, y_pred, None)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = self.compiled_loss(y, y_pred)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.update_metrics(self.train_metrics, y, y_pred, loss)
        return {k: m.result() for k, m in self.train_metrics.items()}

    def test_step(self, data):
        x, y = data
        y_pred = self.model(x, training=False)
        self.compiled_loss(y, y_pred)
        self.update_metrics(self.eval_metrics, y, y_pred)
        return {k: m.result() for k, m in self.eval_metrics.items()}


class TrainableController(tf.keras.callbacks.Callback):

    def __init__(self, max_epochs, ds_val=None, val_steps=None, val_freq=1, print_fn=print):
        super().__init__()
        self.max_epochs = max_epochs
        self.ds_val = ds_val
        self.val_steps = val_steps
        self.val_freq = val_freq
        self.print_fn = print_fn

    def on_epoch_begin(self, epoch, logs=None):
        self.print_fn("Epoch %d/%d" % (epoch + 1, self.max_epochs))
        for m in self.model.train_metrics.values():
            m.reset_states()

    def on_epoch_end(self, epoch, logs=None):
        end_at = time_now()
        log_str = "%s %s - " % (end_at, "train")
        metric_logs = []
        for k, v in logs.items():
            metric_logs.append("%s: %.4f" % (k, v))
        log_str += ", ".join(metric_logs)
        self.print_fn(log_str)

        if self.val_freq > 0 and (epoch + 1) % self.val_freq == 0:
            for m in self.model.eval_metrics.values():
                m.reset_states()
            eval_logs = self.model.evaluate(
                self.ds_val, steps=self.val_steps, verbose=0,
                return_dict=True, _use_cached_eval_dataset=True)
            end_at = time_now()
            log_str = "%s %s - " % (end_at, "valid")
            metric_logs = []
            for k, v in eval_logs.items():
                metric_logs.append("%s: %.4f" % (k, v))
            log_str += ", ".join(metric_logs)
            self.print_fn(log_str)


class SuperLearner:

    def __init__(self, model, criterion, optimizer, train_metrics, eval_metrics,
                 steps_per_loop=None, jit_compile=True, output_transform=lambda x: x, work_dir=None):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_metrics = train_metrics
        self.eval_metrics = eval_metrics
        self.steps_per_loop = steps_per_loop
        self.jit_compile = jit_compile
        self.output_transform = output_transform
        self.work_dir = work_dir

        self._trainable = TrainableModel(model, train_metrics, eval_metrics, output_transform)

        if vparse(tf.__version__) >= vparse("2.8"):
            self._trainable.compile(optimizer, self.criterion, steps_per_execution=steps_per_loop,
                                    jit_compile=self.jit_compile)
        else:
            self._trainable.compile(optimizer, self.criterion, steps_per_execution=steps_per_loop)

    def fit(self, ds_train, epochs, ds_val=None, val_freq=1,
            steps_per_epoch=None, val_steps=None, max_epochs=None, save_freq=None):
        self._trainable.fit(
            ds_train, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=0,
            callbacks=[TrainableController(epochs, ds_val, val_steps, val_freq)])
