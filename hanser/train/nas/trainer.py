import tensorflow as tf
from tensorflow.python.distribute.input_lib import DistributedDataset
from tensorflow.python.keras.callbacks import CallbackList

from hanser.tpu import local_results
from hanser.train.trainer import misc_concat, cast_fp32, parse_strategy, is_global_bfloat16, identity, validate_dataset, \
    print_results


def run_epoch(step_fn, args, steps, metrics, stage="Train"):
    for m in metrics:
        m.reset_states()
    for step in range(steps):
        step_fn(*args)
    metric_results = []
    for m in metrics:
        metric_results.append((m.name, m.result().numpy()))
    print_results(stage, steps, metric_results)


class Trainer:

    def __init__(self, model, criterion, optimizer_arch, optimizer_model,
                 metrics=(), test_metrics=(), grad_clip_norm=None,
                 arch_weight_decay=None, model_weight_decay=None,
                 strategy='auto', metric_transform=identity):
        self.model = model
        self.criterion = criterion
        self.optimizer_arch = optimizer_arch
        self.optimizer_model = optimizer_model
        self.metrics = metrics
        self.test_metrics = test_metrics
        self.grad_clip_norm = grad_clip_norm
        self.arch_weight_decay = arch_weight_decay
        self.model_weight_decay = model_weight_decay
        self.metric_transform = metric_transform

        self.strategy = parse_strategy(strategy)
        self.bfloat16 = is_global_bfloat16()
        self._epoch = tf.Variable(0, trainable=False)

    @tf.function
    def _train_step(self, train_it, search_it, model_only):

        def step_fn(data, data_search):
            input, target = data
            input_search, target_search = data_search

            if not model_only:
                arch_parameters = self.model.arch_parameters()
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(arch_parameters)
                    logits = self.model(input_search, training=False)
                    if self.bfloat16:
                        logits = cast_fp32(logits)
                    per_example_loss = self.criterion(target_search, logits)
                    loss = tf.reduce_mean(per_example_loss)

                    if self.arch_weight_decay:
                        loss = loss + self.arch_weight_decay * tf.add_n([
                            tf.nn.l2_loss(v)
                            for v in arch_parameters
                        ])
                    if self.strategy:
                        loss = loss / self.strategy.num_replicas_in_sync

                grads = tape.gradient(loss, arch_parameters)
                self.optimizer_arch.apply_gradients(zip(grads, arch_parameters))

            model_parameters = self.model.model_parameters()
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(model_parameters)
                logits = self.model(input, training=True)
                if self.bfloat16:
                    logits = cast_fp32(logits)
                per_example_loss = self.criterion(target, logits)
                loss = tf.reduce_mean(per_example_loss)

                if self.model_weight_decay:
                    loss = loss + self.model_weight_decay * tf.add_n([
                        tf.nn.l2_loss(v)
                        for v in model_parameters
                    ])
                if self.strategy:
                    loss = loss / self.strategy.num_replicas_in_sync

            grads = tape.gradient(loss, model_parameters)
            if self.grad_clip_norm:
                grads = tf.clip_by_global_norm(grads, self.grad_clip_norm)[0]
            self.optimizer_model.apply_gradients(zip(grads, model_parameters))

            preds = self.metric_transform(logits)
            for metric in self.metrics:
                if 'loss' in metric.name:
                    metric.update_state(per_example_loss)
                else:
                    metric.update_state(target, preds, None)

        if self.strategy:
            self.strategy.run(step_fn, args=(next(train_it), next(search_it)))
        else:
            step_fn(next(train_it), next(search_it))

    @tf.function
    def _test_step(self, iterator):

        def step_fn(data):
            inputs, target = data
            preds = self.model(inputs, training=False)
            if self.bfloat16:
                preds = cast_fp32(preds)

            preds = self.metric_transform(preds)
            for metric in self.test_metrics:
                metric.update_state(target, preds, None)

        if self.strategy:
            self.strategy.run(step_fn, args=(next(iterator),))
        else:
            step_fn(next(iterator))

    @tf.function
    def _predict_step(self, iterator, debug=False):

        def step_fn(data):
            inputs, target = data
            output = self.model(inputs, training=debug)
            if self.bfloat16:
                output = cast_fp32(output)
            return target, output

        if self.strategy:
            return local_results(
                self.strategy, self.strategy.run(step_fn, args=(next(iterator),)))
        else:
            return step_fn(next(iterator))

    def fit(self, epochs, ds_train, ds_search, steps_per_epoch,
            ds_val, val_steps, val_freq=0, epochs_model_only=0,
            callbacks=None):

        callbacks = CallbackList(callbacks, model=self.model)

        validate_dataset(self.strategy, ds_train, ds_search, ds_val)

        train_it = iter(ds_train)
        search_it = iter(ds_search)
        val_it = iter(ds_val)

        epoch = self._epoch.numpy()
        max_epochs = epoch + epochs

        callbacks.on_train_begin()
        while epoch < max_epochs:
            print('Epoch %d/%d' % (epoch + 1, epochs))

            callbacks.on_epoch_begin(epoch)
            if epochs_model_only and epoch < epochs_model_only:
                run_epoch(self._train_step, (train_it, search_it, True), steps_per_epoch, self.metrics, "Train")
            else:
                run_epoch(self._train_step, (train_it, search_it, False), steps_per_epoch, self.metrics, "Train")
            callbacks.on_epoch_end(epoch)

            epoch = self._epoch.assign_add(1).numpy()

            if val_freq and epoch % val_freq == 0:
                run_epoch(self._test_step, val_it, val_steps, self.test_metrics, "Valid")

        callbacks.on_train_end()

    def evaluate(self, ds_test, test_steps):

        validate_dataset(self.strategy, ds_test)
        test_it = iter(ds_test)

        run_epoch(self._test_step, test_it, test_steps, self.test_metrics, "Test")

    def collect(self, ds_test, test_steps, output_transform=tf.identity, target_transform=tf.identity):

        if self.strategy:
            assert isinstance(ds_test, DistributedDataset)
        test_it = iter(ds_test)

        targets = []
        outputs = []

        for step in range(test_steps):
            target, output = self._predict_step(test_it)

            targets.append(target)
            outputs.append(output)

        target = misc_concat(targets)
        output = misc_concat(outputs)

        output = output_transform(output)
        target = target_transform(target)

        return target, output
