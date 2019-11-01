import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.python.distribute.input_lib import DistributedDataset

import time

import numpy as np
from hanser.tpu import local_results


def print_results(prefix, elapsed, results):
    s = "%s \tcost: %ds" % (prefix, elapsed)
    for name, val in results:
        s = s + ", %s: %.3f" % (name, val)
    print(s)


def run_epoch(step_fn, iterator, steps, metrics, name):
    start = time.time()
    for step in range(steps):
        step_fn(iterator)
    metric_results = []
    for m in metrics:
        metric_results.append((m.name, m.result()))
        m.reset_states()
    elapsed = time.time() - start
    print_results(name, elapsed, metric_results)


def maybe_cat(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], np.ndarray):
        x = np.concatenate(x)
    return x


def maybe_call(fn, *args, default=None):
    if fn is None:
        return default
    else:
        return fn(*args)


class Trainer:

    def __init__(self, model, criterion, optimizer, lr_schedule, metrics=(), test_metrics=(), model_dir=None, tpu=None,
                 strategy=None, weight_decay=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_schedule = lr_schedule
        self.metrics = metrics
        self.test_metrics = test_metrics
        self.model_dir = model_dir
        self.tpu = tpu
        self.strategy = strategy

        self.weight_decay = weight_decay
        self._get_sample_weight = None

        self._epoch = tf.Variable(0, trainable=False)

    def _make_ckpt(self, name, **kwargs):
        ckpt = tf.train.Checkpoint(**kwargs)
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, directory=self.model_dir + "/" + name, max_to_keep=1)
        return ckpt, ckpt_manager

    @tf.function
    def _train_step(self, iterator):
        def step_fn(images, labels):
            with tf.GradientTape() as tape:
                preds = self.model(images, training=True)
                loss1 = self.criterion(labels, preds)
                if loss1.shape.ndims != 0:
                    loss1 = tf.reduce_mean(loss1)
                if self.weight_decay:
                    loss2 = self.weight_decay * tf.add_n([
                        tf.nn.l2_loss(v)
                        for v in self.model.trainable_variables()
                        if 'batch_normalization' not in v.name
                    ])
                    loss = loss1 + loss2
                else:
                    loss = loss1
                if self.strategy:
                    loss = loss / self.strategy.num_replicas_in_sync
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads, self.model.trainable_variables))

            for metric in self.metrics:
                if 'loss' in metric.name:
                    metric.update_state(loss1)
                else:
                    sample_weight = maybe_call(self._get_sample_weight, labels, preds)
                    metric.update_state(labels, preds, sample_weight)
        self.strategy.experimental_run_v2(step_fn, *next(iterator))

    def _test_step(self, iterator):
        def step_fn(images, labels):
            preds = self.model(images, training=False)
            loss = self.criterion(labels, preds)
            if loss.shape.ndims != 0:
                loss = tf.reduce_mean(loss)

            for metric in self.test_metrics:
                if 'loss' in metric.name:
                    metric.update_state(loss)
                else:
                    sample_weight = maybe_call(self._get_sample_weight, labels, preds)
                    metric.update_state(labels, preds, sample_weight)
        self.strategy.experimental_run_v2(step_fn, *next(iterator))

    def restore(self, checkpoint, manager):
        assert self.model_dir is not None, "`model_dir` should be provided."

        if manager.latest_checkpoint:
            if self.tpu:
                with self.strategy.scope():
                    checkpoint.restore(manager.latest_checkpoint)
            else:
                checkpoint.restore(manager.latest_checkpoint)
            print("Restored from {}".format(manager.latest_checkpoint))
            return True
        else:
            print("Initializing from scratch.")
            return False

    def train_and_evaluate(self, epochs, ds_train, steps_per_epoch, ds_val, val_steps, resume=True,
                           save_per_epochs=None, get_sample_weight=None):
        self._get_sample_weight = get_sample_weight

        if save_per_epochs or resume:
            assert self.model_dir is not None, "`model_dir` should be provided."

            model_ckpt, model_ckpt_manager = self._make_ckpt("model", model=self.model)
            optim_ckpt, optim_ckpt_manager = self._make_ckpt("optim", optimizer=self.optimizer, epoch=self._epoch)

        if self.tpu:
            assert isinstance(ds_train, DistributedDataset)
            assert isinstance(ds_val, DistributedDataset)

        if resume:
            self.restore(model_ckpt, model_ckpt_manager)
            self.restore(optim_ckpt, optim_ckpt_manager)

        epoch = self._epoch.numpy()
        max_epochs = epoch + epochs

        while epoch < max_epochs:
            print('Epoch %s' % (epoch + 1))

            K.set_value(self.optimizer.lr, self.lr_schedule(epoch))

            run_epoch(self._train_step, ds_train, steps_per_epoch, self.metrics, "Train")
            run_epoch(self._test_step, ds_val, val_steps, self.test_metrics, "Val")

            epoch = self._epoch.assign_add(1).numpy()
            if save_per_epochs and epoch % save_per_epochs == 0:
                print("Saved model: %s" % model_ckpt_manager.save(epoch))
                print("Saved optimizer: %s" % optim_ckpt_manager.save(epoch))

    def evaluate(self, ds_test, test_steps, get_sample_weight=None):
        self._get_sample_weight = get_sample_weight

        assert self.model_dir is not None, "`model_dir` should be provided."
        model_ckpt, model_ckpt_manager = self._make_ckpt("model", model=self.model)

        if self.tpu:
            assert isinstance(ds_test, DistributedDataset)

        self.restore(model_ckpt, model_ckpt_manager)

        run_epoch(self._test_step, ds_test, test_steps, self.test_metrics, "Test")

    def evaluate2(self, ds_test, test_steps, metrics,
                  output_transform=lambda x: x, target_transform=lambda x: x, get_sample_weight=None):

        assert self.model_dir is not None, "`model_dir` should be provided."
        model_ckpt, model_ckpt_manager = self._make_ckpt("model", model=self.model)

        assert callable(metrics), "Define metrics as lambda: metrics."
        g_cpu = tf.Graph()
        with g_cpu.as_default():
            metrics = metrics()

        def predict_step(inputs, target):
            output = output_transform(self.model(inputs, training=False))
            # target = target_transform(target)
            return target, output

        ds_test = self.strategy.experimental_distribute_dataset(
            ds_test) if self.strategy else ds_test

        test_it = ds_test.make_initializable_iterator()

        if self.strategy:
            predict_op = local_results(
                self.strategy, self.strategy.experimental_run_v2(predict_step, test_it.get_next()))
        else:
            predict_op = predict_step(*test_it.get_next())

        with tf.Session(target=self._target, config=self._config) as sess:
            all_variables = tf.global_variables()

            sess.run([v.initializer for v in all_variables])
            sess.run(test_it.initializer)

            self.restore(sess, model_ckpt, model_ckpt_manager)

            sess_cpu = tf.Session(graph=g_cpu)
            sess_cpu.run([
                v.initializer
                for m in metrics for v in m.variables
            ])

            for step in range(test_steps):
                target, output = sess.run(predict_op)

                target = maybe_cat(target)
                output = maybe_cat(output)

                with g_cpu.as_default():
                    target = tf.convert_to_tensor(target)
                    output = tf.convert_to_tensor(output)
                    weight = maybe_call(get_sample_weight, target, output)
                    target = target_transform(target_transform)
                    for m in metrics:
                        sess_cpu.run(m.update_state(target, output, weight))
        with g_cpu.as_default():
            results = [sess_cpu.run(m.result()) for m in metrics]
        sess_cpu.close()
        return results


    def collect(self, ds_test, test_steps, output_transform, target_transform):

        assert self.model_dir is not None, "`model_dir` should be provided."
        model_ckpt, model_ckpt_manager = self._make_ckpt("model", model=self.model)

        def predict_step(inputs, target):
            output = output_transform(self.model(inputs, training=False))
            target = target_transform(target)
            return target, output

        ds_test = self.strategy.experimental_distribute_dataset(
            ds_test) if self.strategy else ds_test

        test_it = ds_test.make_initializable_iterator()

        if self.strategy:
            predict_op = local_results(
                self.strategy, self.strategy.experimental_run_v2(predict_step, test_it.get_next()))
        else:
            predict_op = predict_step(*test_it.get_next())

        with tf.Session(target=self._target, config=self._config) as sess:
            all_variables = tf.global_variables()

            sess.run([v.initializer for v in all_variables])
            sess.run(test_it.initializer)

            self.restore(sess, model_ckpt, model_ckpt_manager)

            targets = []
            outputs = []

            for step in range(test_steps):
                target, output = sess.run(predict_op)

                if isinstance(target, np.ndarray):
                    targets.append(target)
                    outputs.append(output)
                else:
                    targets.extend(target)
                    outputs.extend(output)

        target = np.concatenate(targets)
        output = np.concatenate(outputs)
        return target, output
