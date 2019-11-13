import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.python.distribute.input_lib import DistributedDataset

import time

from hanser.tpu import local_results


@tf.function
def identity(x):
    return x


def print_results(prefix, elapsed, results):
    s = "%s \tcost: %ds" % (prefix, elapsed)
    for name, val in results:
        s = s + ", %s: %.3f" % (name, val)
    print(s)


def run_epoch(step_fn, iterator, steps, metrics, name="Train"):
    start = time.time()
    for m in metrics:
        m.reset_states()
    for step in range(steps):
        step_fn(iterator)
    metric_results = []
    for m in metrics:
        metric_results.append((m.name, m.result()))
    elapsed = time.time() - start
    print_results(name, elapsed, metric_results)


def maybe_cat(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], tf.Tensor):
        x = tf.concat(x, 0)
    return x


def maybe_call(fn, *args, default=None):
    if fn is None:
        return default
    else:
        return fn(*args)


class Trainer:

    def __init__(self, model, criterion, optimizer, metrics=(), test_metrics=(), model_dir=None,
                 strategy=None, weight_decay=None, bfloat16=False):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metrics = metrics
        self.test_metrics = test_metrics
        self.model_dir = model_dir
        self.strategy = strategy
        if strategy and model_dir:
            assert model_dir.startswith('gs'), "Use gs://... as `model_dir` on tpu mode"

        self.bfloat16 = bfloat16
        self.weight_decay = weight_decay
        self._get_sample_weight = None

        self._epoch = tf.Variable(0, trainable=False)

    def _maybe_cat(self, values):
        if self.strategy:
            return misc_concat(values)
        return values

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
                if self.bfloat16:
                    preds = tf.cast(preds, tf.float32)
                loss1 = self.criterion(labels, preds)
                if loss1.shape.ndims != 0:
                    loss1 = tf.reduce_mean(loss1)
                if self.weight_decay:
                    loss2 = self.weight_decay * tf.add_n([
                        tf.nn.l2_loss(v)
                        for v in self.model.trainable_variables
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

        if self.strategy:
            self.strategy.experimental_run_v2(step_fn, args=next(iterator))
        else:
            step_fn(*next(iterator))

    @tf.function
    def _test_step(self, iterator):
        def step_fn(images, labels):
            preds = self.model(images, training=False)
            if self.bfloat16:
                preds = tf.cast(preds, tf.float32)
            loss = self.criterion(labels, preds)
            if loss.shape.ndims != 0:
                loss = tf.reduce_mean(loss)

            for metric in self.test_metrics:
                if 'loss' in metric.name:
                    metric.update_state(loss)
                else:
                    sample_weight = maybe_call(self._get_sample_weight, labels, preds)
                    metric.update_state(labels, preds, sample_weight)

        if self.strategy:
            self.strategy.experimental_run_v2(step_fn, args=next(iterator))
        else:
            step_fn(*next(iterator))

    def _get_predict_step(self, debug=False):

        @tf.function
        def predict_step(iterator):
            def step_fn(inputs, target):
                output = self.model(inputs, training=debug)
                if self.bfloat16:
                    output = tf.cast(output, tf.float32)
                # output = output_transform(output)
                # target = target_transform(target)
                return target, output

            if self.strategy:
                return local_results(
                    self.strategy, self.strategy.experimental_run_v2(step_fn, args=next(iterator)))
            else:
                return step_fn(*next(iterator))

        return predict_step

    def restore(self, checkpoint, manager):
        assert self.model_dir is not None, "`model_dir` should be provided."

        if manager.latest_checkpoint:
            if self.strategy:
                with self.strategy.scope():
                    checkpoint.restore(manager.latest_checkpoint)
            else:
                checkpoint.restore(manager.latest_checkpoint)
            print("Restored from {}".format(manager.latest_checkpoint))
            return True
        else:
            print("Initializing from scratch.")
            return False

    def resume(self):
        model_ckpt, model_ckpt_manager = self._make_ckpt("model", model=self.model)
        optim_ckpt, optim_ckpt_manager = self._make_ckpt("optim", optimizer=self.optimizer, epoch=self._epoch)

        self.restore(model_ckpt, model_ckpt_manager)
        self.restore(optim_ckpt, optim_ckpt_manager)

    def fit(self, epochs, ds_train, steps_per_epoch,
            ds_val, val_steps, val_freq=1,
            save_per_epochs=None, get_sample_weight=None,
            extra_metrics=(), extra_eval_freq=1,
            target_transform=identity, output_transform=identity,
            debug=False):
        self._get_sample_weight = get_sample_weight

        if self.strategy:
            assert isinstance(ds_train, DistributedDataset)
            assert isinstance(ds_val, DistributedDataset)

        train_it = iter(ds_train)
        val_it = iter(ds_val)

        if save_per_epochs:
            assert self.model_dir is not None, "`model_dir` should be provided."

            model_ckpt, model_ckpt_manager = self._make_ckpt("model", model=self.model)
            optim_ckpt, optim_ckpt_manager = self._make_ckpt("optim", optimizer=self.optimizer, epoch=self._epoch)

        epoch = self._epoch.numpy()
        max_epochs = epoch + epochs

        if extra_metrics:
            assert target_transform
            assert output_transform
            assert extra_eval_freq
            predict_step = self._get_predict_step(debug)

        while epoch < max_epochs:
            print('Epoch %s' % (epoch + 1))

            run_epoch(self._train_step, train_it, steps_per_epoch, self.metrics, "Train")

            epoch = self._epoch.assign_add(1).numpy()

            if epoch % val_freq == 0:
                run_epoch(self._test_step, val_it, val_steps, self.test_metrics, "Val")

            if extra_metrics and epoch % extra_eval_freq == 0:
                start = time.time()
                for step in range(val_steps):
                    target, output = predict_step(val_it)

                    target = self._maybe_cat(target)
                    output = self._maybe_cat(output)

                    weight = maybe_call(get_sample_weight, target, output)
                    output = output_transform(output)
                    target = target_transform(target)

                    for m in extra_metrics:
                        m.update_state(target, output, weight)
                metric_results = []
                for m in extra_metrics:
                    metric_results.append((m.name, m.result()))
                    m.reset_states()
                elapsed = time.time() - start
                print_results("Eval", elapsed, metric_results)

            if save_per_epochs and epoch % save_per_epochs == 0:
                print("Saved model: %s" % model_ckpt_manager.save(epoch))
                print("Saved optimizer: %s" % optim_ckpt_manager.save(epoch))

    def evaluate(self, ds_test, test_steps, get_sample_weight=None):
        self._get_sample_weight = get_sample_weight

        if self.strategy:
            assert isinstance(ds_test, DistributedDataset)
        test_it = iter(ds_test)

        run_epoch(self._test_step, test_it, test_steps, self.test_metrics, "Test")

    def evaluate2(self, ds_test, test_steps, metrics,
                  output_transform=identity, target_transform=identity, get_sample_weight=None, debug=False):

        if self.strategy:
            assert isinstance(ds_test, DistributedDataset)

        predict_step = self._get_predict_step(debug)

        test_it = iter(ds_test)

        for step in range(test_steps):
            target, output = predict_step(test_it)

            target = self._maybe_cat(target)
            output = self._maybe_cat(output)

            weight = maybe_call(get_sample_weight, target, output)
            output = output_transform(output)
            target = target_transform(target)

            for m in metrics:
                m.update_state(target, output, weight)
        results = {m.name: m.result().numpy() for m in metrics}
        return results

    def collect(self, ds_test, test_steps, output_transform=identity, target_transform=identity):

        if self.strategy:
            assert isinstance(ds_test, DistributedDataset)
        test_it = iter(ds_test)

        targets = []
        outputs = []

        predict_step = self._get_predict_step()

        for step in range(test_steps):
            target, output = predict_step(test_it)

            if isinstance(target, (tuple, list)):
                targets.extend(target)
                outputs.extend(output)
            else:
                targets.append(target)
                outputs.append(output)

        target = misc_concat(targets)
        output = misc_concat(outputs)

        output = output_transform(output)
        target = target_transform(target)

        return target, output


def misc_concat(values):
    if isinstance(values, (tuple, list)):
        val = values[0]
        if tf.is_tensor(val):
            return tf.concat(values, 0)
        elif isinstance(val, dict):
            d = {}
            for k in val.keys():
                d[k] = misc_concat([v[k] for v in values])
            return d
        elif isinstance(val, (tuple, list)):
            return val.__class__(v for l in values for v in l)
        else:
            return values
    elif isinstance(values, dict):
        return {k: misc_concat(v) for k, v in values.items()}
    else:
        return values
