import tensorflow as tf

from hanser.train.learner import Learner, cast

class OneShotLearner(Learner):

    def __init__(self, model, criterion, optimizer,
                 weight_decay=0.0, grad_clip_norm=0.0, **kwargs):
        self.weight_decay = weight_decay
        self.grad_clip_norm = grad_clip_norm
        super().__init__(model, criterion, optimizer, **kwargs)

    def train_batch(self, batch):
        model = self.model
        optimizer = self.optimizers[0]

        inputs, target = batch
        with tf.GradientTape() as tape:
            inputs = cast(inputs, self.dtype)
            preds = model(inputs, training=True)
            preds = cast(preds, tf.float32)
            per_example_loss = self.criterion(target, preds)
            loss = self.reduce_loss(per_example_loss)

            l2_loss = model.l2_loss()
            if self._strategy:
                l2_loss = l2_loss / self._strategy.num_replicas_in_sync
            loss = loss + self.weight_decay * l2_loss

            if self.dtype == tf.float16:
                loss = optimizer.get_scaled_loss(loss)
        self.minimize(tape, optimizer, loss, model.trainable_variables, self.grad_clip_norm)
        self.update_metrics(self.train_metrics, target, preds, per_example_loss)
        if hasattr(self, "_ema") and self._ema is not None:
            self._ema.apply(self._ema_vars)

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

    def local_eval_batch(self, batch):
        return self._eval_batch(batch)
