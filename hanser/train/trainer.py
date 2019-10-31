import tensorflow as tf

import time

import numpy as np


def print_results(prefix, elapsed, results):
    s = "%s \tcost: %ds" % (prefix, elapsed)
    for name, val in results:
        s = s + ", %s: %.3f" % (name, val)
    print(s)


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

        if self.model_dir:
            self.model_ckpt = tf.train.Checkpoint(model=model)
            self.model_ckpt_manager = tf.train.CheckpointManager(
                self.model_ckpt, directory=model_dir, max_to_keep=1, checkpoint_name='model')

            self.optim_ckpt = tf.train.Checkpoint(optimizer=optimizer, epoch=self._epoch)
            self.optim_ckpt_manager = tf.train.CheckpointManager(
                self.optim_ckpt, directory=model_dir, max_to_keep=1, checkpoint_name='optim')


    def _train_step(self, inputs):
        images, labels = inputs
        with tf.GradientTape() as tape:
            preds = self.model(images, training=True)
            loss1 = self.criterion(labels, preds)
            if loss1.shape.ndims != 0:
                loss1 = tf.reduce_mean(loss1)
            if self.weight_decay is not None:
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
                if self._get_sample_weight is None:
                    sample_weight = None
                else:
                    sample_weight = self._get_sample_weight(labels, preds)
                update_op = metric.update_state(labels, preds, sample_weight)
            update_ops.append(update_op)
        #     update_accuracy = training_accuracy.update_state(labels, logits)
        with tf.control_dependencies(update_ops):
            return tf.identity(loss)

    def _test_step(self, inputs):
        images, labels = inputs
        preds = self.model(images, training=False)
        loss = self.criterion(labels, preds)
        if loss.shape.ndims != 0:
            loss = tf.reduce_mean(loss)

        update_ops = []
        for metric in self.test_metrics:
            if 'loss' in metric.name:
                update_op = metric.update_state(loss)
            else:
                if self._get_sample_weight is None:
                    sample_weight = None
                else:
                    sample_weight = self._get_sample_weight(labels, preds)
                update_op = metric.update_state(labels, preds, sample_weight)
            update_ops.append(update_op)
        #     update_accuracy = training_accuracy.update_state(labels, logits)
        with tf.control_dependencies(update_ops):
            return tf.identity(loss)

    def restore_model(self, sess, checkpoint, manager):
        assert self.model_dir is not None, "`model_dir` should be provided."

        if manager.latest_checkpoint:
            checkpoint.restore(manager.latest_checkpoint).assert_consumed().run_restore_ops(sess)
            print("Restored from {}".format(manager.latest_checkpoint))
            return True
        else:
            print("Initializing from scratch.")
            return False

    def restore(self, sess):
        assert self.model_dir is not None, "`model_dir` should be provided."

        if self.model_ckpt_manager.latest_checkpoint:
            self.model_ckpt.restore(self.model_ckpt_manager.latest_checkpoint).assert_consumed().run_restore_ops(sess)
            print("Restored model from {}".format(self.model_ckpt_manager.latest_checkpoint))
        else:
            print("Initializing model from scratch.")

        if self.optim_ckpt_manager.latest_checkpoint:
            self.optim_ckpt.restore(self.optim_ckpt_manager.latest_checkpoint).assert_consumed().run_restore_ops(sess)
            print("Restored optimizer from {}".format(self.optim_ckpt_manager.latest_checkpoint))
        else:
            print("Initializing optimizer from scratch.")

    def train_and_evaluate(self, epochs, ds_train, steps_per_epoch, ds_val, val_steps, resume=True,
                           save_per_epochs=None, get_sample_weight=None):
        self._get_sample_weight = get_sample_weight

        if save_per_epochs or resume:
            assert self.model_dir is not None, "`model_dir` should be provided."

        if self.tpu is None:
            train_it = ds_train.make_initializable_iterator()
            train_op = self._train_step(train_it.get_next())

            val_it = ds_val.make_initializable_iterator()
            val_op = self._test_step(val_it.get_next())

            target = ''
            config = None
        else:
            ds_train = self.strategy.experimental_distribute_dataset(ds_train)
            train_it = ds_train.make_initializable_iterator()
            train_op = self.strategy.experimental_local_results(
                self.strategy.experimental_run_v2(
                    self._train_step, args=(train_it.get_next(),)))

            ds_val = self.strategy.experimental_distribute_dataset(ds_val)
            val_it = ds_val.make_initializable_iterator()
            val_op = self.strategy.experimental_local_results(
                self.strategy.experimental_run_v2(
                    self._test_step, args=(val_it.get_next(),)))

            target = self.tpu.master()
            config = tf.ConfigProto(
                allow_soft_placement=True,
                cluster_def=self.tpu.cluster_spec().as_cluster_def()
            )

        with tf.Session(target=target, config=config) as sess:
            # all_variables = self.model.variables + self.optimizer.variables()
            all_variables = tf.global_variables()
            for metric in self.metrics:
                all_variables.extend(metric.variables)
            for metric in self.test_metrics:
                all_variables.extend(metric.variables)

            sess.run([v.initializer for v in all_variables])
            sess.run(train_it.initializer)
            sess.run(val_it.initializer)
            #     checkpoint.restore(manager.latest_checkpoint)

            metric_result_ops = [m.result() for m in self.metrics]
            test_metric_result_ops = [m.result() for m in self.test_metrics]

            if resume:
                self.restore(sess)

            epoch = sess.run(self._epoch)
            max_epochs = epoch + epochs
            epoch_inc_op = self._epoch.assign_add(1)

            while epoch < max_epochs:
                print('Epoch %s' % (epoch + 1))
                start = time.time()

                lr = self.lr_schedule(epoch)
                tf.keras.backend.set_value(self.optimizer.lr, lr)
                print("Set lr")
                for step in range(steps_per_epoch):
                    # lr = self.lr_schedule(epoch + float(step) / steps_per_epoch)
                    # tf.keras.backend.set_value(self.optimizer.lr, lr)
                    sess.run(train_op)
                    print("Step %d" % step)

                metric_results = []
                for m, r in zip(self.metrics, metric_result_ops):
                    metric_results.append((m.name, sess.run(r)))
                    m.reset_states()
                    print("Reset %s" % m)
                elapsed = time.time() - start
                print_results("Train", elapsed, metric_results)

                start = time.time()
                for step in range(val_steps):
                    sess.run(val_op)
                metric_results = []
                for m, r in zip(self.test_metrics, test_metric_result_ops):
                    metric_results.append((m.name, sess.run(r)))
                    m.reset_states()
                elapsed = time.time() - start
                print_results("Val", elapsed, metric_results)

                epoch = sess.run(epoch_inc_op)

                if save_per_epochs and epoch % save_per_epochs == 0:
                    print("Saved checkpoint: %s" % self.model_ckpt_manager.save(epoch))
                    print("Saved optimizer: %s" % self.optim_ckpt_manager.save(epoch))

    def evaluate(self, ds_test, test_steps, get_sample_weight=None):
        self._get_sample_weight = get_sample_weight

        assert self.model_dir is not None, "`model_dir` should be provided."

        if self.tpu is None:
            test_it = ds_test.make_initializable_iterator()
            train_op = self._train_step(test_it.get_next())
            test_op = self._test_step(test_it.get_next())

            target = ''
            config = None
        else:
            ds_test = self.strategy.experimental_distribute_dataset(ds_test)
            test_it = ds_test.make_initializable_iterator()
            train_op = self.strategy.experimental_local_results(
                self.strategy.experimental_run_v2(
                    self._train_step, args=(test_it.get_next(),)))

            test_op = self.strategy.experimental_local_results(
                self.strategy.experimental_run_v2(
                    self._test_step, args=(test_it.get_next(),)))

            target = self.tpu.master()
            config = tf.ConfigProto(
                allow_soft_placement=True,
                cluster_def=self.tpu.cluster_spec().as_cluster_def()
            )

        with tf.Session(target=target, config=config) as sess:
            # all_variables = self.model.variables + self.optimizer.variables()
            all_variables = tf.global_variables()
            for metric in self.test_metrics:
                all_variables.extend(metric.variables)

            sess.run([v.initializer for v in all_variables])
            sess.run(test_it.initializer)

            test_metric_result_ops = [m.result() for m in self.test_metrics]

            self.restore(sess)

            start = time.time()
            for step in range(test_steps):
                sess.run(test_op)
            metric_results = []
            for m, r in zip(self.test_metrics, test_metric_result_ops):
                metric_results.append((m.name, sess.run(r)))
                m.reset_states()
            elapsed = time.time() - start
            print_results("Test", elapsed, metric_results)

    def evaluate2(self, ds_test, test_steps, metrics, output_transform, target_transform, get_sample_weight=None):

        assert self.model_dir is not None, "`model_dir` should be provided."

        if self.tpu is None:

            test_it1 = ds_test.make_initializable_iterator()
            test_it2 = ds_test.make_initializable_iterator()
            predict_op = output_transform(self.model(test_it1.get_next()[0], training=False))
            target_op = target_transform(test_it2.get_next()[1])

            target = ''
            config = None
        else:
            ds_test = self.strategy.experimental_distribute_dataset(ds_test)
            test_it1 = ds_test.make_initializable_iterator()
            test_it2 = ds_test.make_initializable_iterator()
            predict_op = self.strategy.experimental_local_results(
                self.strategy.experimental_run_v2(
                    lambda x: output_transform(self.model(x, training=False)), args=(test_it1.get_next()[0],)))

            target_op = self.strategy.experimental_local_results(
                self.strategy.experimental_run_v2(
                    lambda x: target_transform(x[1]), args=(test_it2.get_next(),)))

            target = self.tpu.master()
            config = tf.ConfigProto(
                allow_soft_placement=True,
                cluster_def=self.tpu.cluster_spec().as_cluster_def()
            )

        checkpoint = tf.train.Checkpoint(model=self.model)
        manager = tf.train.CheckpointManager(
            checkpoint, directory=self.model_dir, max_to_keep=1, checkpoint_name='model')

        with tf.Session(target=target, config=config) as sess:
            # all_variables = self.model.variables + self.optimizer.variables()
            all_variables = tf.global_variables()

            sess.run([v.initializer for v in all_variables])
            sess.run(test_it1.initializer)
            sess.run(test_it2.initializer)

            self.restore_model(sess, checkpoint, manager)

            sess_cpu = tf.Session()
            sess_cpu.run([
                v.initializer
                for m in metrics
                for v in m.variables
            ])

            for step in range(test_steps):
                pred = sess.run(predict_op)
                target = sess.run(target_op)
                if not isinstance(pred, np.ndarray):
                    pred = np.concatenate(pred)
                if not isinstance(target, np.ndarray):
                    target = np.concatenate(target)

                pred = tf.convert_to_tensor(pred)
                target = tf.convert_to_tensor(target)
                if get_sample_weight is not None:
                    weight = get_sample_weight(target, pred)
                else:
                    weight = None
                for m in metrics:
                    sess_cpu.run(m.update_state(target, pred, weight))
            return [sess_cpu.run(m.result()) for m in metrics]
