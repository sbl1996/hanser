import tensorflow as tf

from hanser.train.learner import Learner, cast

class SuperLearner(Learner):


    def __init__(self, model, criterion, optimizer, grad_clip_norm=0.0,
                 batch_transform=None, **kwargs):
        self.grad_clip_norm = grad_clip_norm
        self.batch_transform = batch_transform
        super().__init__(model, criterion, optimizer, **kwargs)

    def train_batch(self, batch):
        model = self.model
        optimizer = self.optimizers[0]

        inputs, target = batch
        if self.batch_transform is not None:
            inputs, target = self.batch_transform(inputs, target)
        with tf.GradientTape() as tape:
            inputs = cast(inputs, self.dtype)
            preds = model(inputs, training=True)
            preds = cast(preds, tf.float32)
            per_example_loss = self.criterion(target, preds)
            loss = self.reduce_loss(per_example_loss)
            if self.dtype == tf.float16:
                loss = optimizer.get_scaled_loss(loss)
        self.minimize(tape, optimizer, loss, model.trainable_variables, self.grad_clip_norm)
        self.update_metrics(self.train_metrics, target, preds, per_example_loss)

        if hasattr(self, "_ema") and self._ema is not None:
            self._ema.apply(self._ema_vars)

        return_metrics = {}
        for name, metric in self.train_metrics.items():
            return_metrics[name] = metric.result()
        return return_metrics


    def train_batches(self, *batches):
        batch = tuple(tf.concat(xs, axis=0) for xs in zip(*batches))
        return self.train_batch(batch)

    def _eval_batch(self, batch):
        inputs, target = batch
        inputs = cast(inputs, self.dtype)
        preds = self.model(inputs, training=False)
        preds = cast(preds, tf.float32)
        return target, preds

    def eval_batch(self, batch):
        target, preds = self._eval_batch(batch)
        self.update_metrics(self.eval_metrics, target, preds)

        return_metrics = {}
        for name, metric in self.eval_metrics.items():
            return_metrics[name] = metric.result()
        return return_metrics

    def local_eval_batch(self, batch):
        return self._eval_batch(batch)

CNNLearner = SuperLearner