import tensorflow as tf
import tensorflow.keras.backend as K

import time

import numpy as np
from hanser.tpu import local_results


def print_results(prefix, elapsed, results):
    s = "%s \tcost: %ds" % (prefix, elapsed)
    for name, val in results:
        s = s + ", %s: %.3f" % (name, val)
    print(s)


def run_epoch(sess, op, steps, metrics, metric_resuls_ops, name):
    start = time.time()
    for step in range(steps):
        sess.run(op)
    metric_results = []
    for m, r in zip(metrics, metric_resuls_ops):
        metric_results.append((m.name, sess.run(r)))
        m.reset_states()
    elapsed = time.time() - start
    print_results(name, elapsed, metric_results)


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

        self._target = tpu.master() if tpu else ''
        self._config = tf.ConfigProto(
            allow_soft_placement=True,
            cluster_def=tpu.cluster_spec().as_cluster_def()
        ) if tpu else None

        self.weight_decay = weight_decay
        self._get_sample_weight = None

        self._epoch = tf.Variable(0, trainable=False)

    def _make_ckpt(self, name, **kwargs):
        ckpt = tf.train.Checkpoint(**kwargs)
        ckpt_manager = tf.train.CheckpointManager(
            ckpt, directory=self.model_dir + "/" + name, max_to_keep=1)
        return ckpt, ckpt_manager

    def _train_step(self, images, labels):
        with tf.GradientTape() as tape:
            preds = self.model(images, training=True)
            loss1 = self.criterion(labels, preds)
            if loss1.shape.ndims != 0:
                loss1 = tf.reduce_mean(loss1)
            if self.weight_decay:
                loss2 = self.weight_decay * tf.add_n([
                    tf.nn.l2_loss(v)
                    for v in tf.trainable_variables()
                    if 'batch_normalization' not in v.name
                ])
                loss = loss1 + loss2
            else:
                loss = loss1
            if self.strategy:
                loss = loss / self.strategy.num_replicas_in_sync
            grads = tape.gradient(loss, self.model.trainable_variables)
        update_vars = self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))

        update_ops = [update_vars]
        for metric in self.metrics:
            if 'loss' in metric.name:
                update_op = metric.update_state(loss1)
            else:
                sample_weight = maybe_call(self._get_sample_weight, labels, preds)
                update_op = metric.update_state(labels, preds, sample_weight)
            update_ops.append(update_op)

        with tf.control_dependencies(update_ops):
            return tf.identity(loss)

    def _test_step(self, images, labels):
        preds = self.model(images, training=False)
        loss = self.criterion(labels, preds)
        if loss.shape.ndims != 0:
            loss = tf.reduce_mean(loss)

        update_ops = []
        for metric in self.test_metrics:
            if 'loss' in metric.name:
                update_op = metric.update_state(loss)
            else:
                sample_weight = maybe_call(self._get_sample_weight, labels, preds)
                update_op = metric.update_state(labels, preds, sample_weight)
            update_ops.append(update_op)

        with tf.control_dependencies(update_ops):
            return tf.identity(loss)

    def restore(self, sess, checkpoint, manager):
        assert self.model_dir is not None, "`model_dir` should be provided."

        if manager.latest_checkpoint:
            checkpoint.restore(manager.latest_checkpoint).assert_consumed().run_restore_ops(sess)
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

        if self.tpu is None:
            train_it = ds_train.make_initializable_iterator()
            train_op = self._train_step(*train_it.get_next())

            val_it = ds_val.make_initializable_iterator()
            val_op = self._test_step(*val_it.get_next())

        else:
            ds_train = self.strategy.experimental_distribute_dataset(ds_train)
            train_it = ds_train.make_initializable_iterator()
            train_op = self.strategy.experimental_local_results(
                self.strategy.experimental_run_v2(
                    self._train_step, train_it.get_next()))

            ds_val = self.strategy.experimental_distribute_dataset(ds_val)
            val_it = ds_val.make_initializable_iterator()
            val_op = self.strategy.experimental_local_results(
                self.strategy.experimental_run_v2(
                    self._test_step, val_it.get_next()))

        with tf.Session(target=self._target, config=self._config) as sess:
            # all_variables = self.model.variables + self.optimizer.variables()
            all_variables = tf.global_variables()
            for metric in self.metrics:
                all_variables.extend(metric.variables)
            for metric in self.test_metrics:
                all_variables.extend(metric.variables)

            sess.run([v.initializer for v in all_variables + [train_it, val_it]])

            if resume:
                self.restore(sess, model_ckpt, model_ckpt_manager)
                self.restore(sess, optim_ckpt, optim_ckpt_manager)

            metric_result_ops = [m.result() for m in self.metrics]
            test_metric_result_ops = [m.result() for m in self.test_metrics]

            epoch = sess.run(self._epoch)
            max_epochs = epoch + epochs
            epoch_inc_op = self._epoch.assign_add(1)

            while epoch < max_epochs:
                print('Epoch %s' % (epoch + 1))

                K.set_value(self.optimizer.lr, self.lr_schedule(epoch))
                run_epoch(sess, train_op, steps_per_epoch, self.metrics, metric_result_ops, "Train")

                run_epoch(sess, val_op, val_steps, self.test_metrics, test_metric_result_ops, "Val")

                epoch = sess.run(epoch_inc_op)
                if save_per_epochs and epoch % save_per_epochs == 0:
                    print("Saved model: %s" % model_ckpt_manager.save(epoch))
                    print("Saved optimizer: %s" % optim_ckpt_manager.save(epoch))

    def evaluate(self, ds_test, test_steps, get_sample_weight=None):
        self._get_sample_weight = get_sample_weight

        assert self.model_dir is not None, "`model_dir` should be provided."
        model_ckpt, model_ckpt_manager = self._make_ckpt("model", model=self.model)

        if self.tpu is None:
            test_it = ds_test.make_initializable_iterator()
            test_op = self._test_step(*test_it.get_next())
        else:
            ds_test = self.strategy.experimental_distribute_dataset(ds_test)
            test_it = ds_test.make_initializable_iterator()

            test_op = self.strategy.experimental_local_results(
                self.strategy.experimental_run_v2(
                    self._test_step, test_it.get_next()))

        with tf.Session(target=self._target, config=self._config) as sess:
            all_variables = tf.global_variables()
            for metric in self.test_metrics:
                all_variables.extend(metric.variables)

            sess.run([v.initializer for v in all_variables])
            sess.run(test_it.initializer)

            test_metric_result_ops = [m.result() for m in self.test_metrics]

            self.restore(sess, model_ckpt, model_ckpt_manager)

            run_epoch(sess, test_op, test_steps, self.test_metrics, test_metric_result_ops, "Test")

    def evaluate2(self, ds_test, test_steps, metrics, output_transform, target_transform, get_sample_weight=None):

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

            sess_cpu = tf.Session()
            sess_cpu.run([
                v.initializer
                for m in metrics for v in m.variables
            ])

            for step in range(test_steps):
                target, output = sess.run(predict_op)

                target = maybe_cat(target)
                output = maybe_cat(output)

                target = tf.convert_to_tensor(target)
                output = tf.convert_to_tensor(output)
                weight = maybe_call(get_sample_weight, target, output)
                for m in metrics:
                    sess_cpu.run(m.update_state(target, output, weight))
            return [sess_cpu.run(m.result()) for m in metrics]


def maybe_cat(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], np.ndarray):
        x = np.concatenate(x)
    return x


def maybe_call(fn, *args, default=None):
    if fn is None:
        return default
    else:
        return fn(*args)