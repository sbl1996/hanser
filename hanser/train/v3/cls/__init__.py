import tensorflow as tf

from hanser.train.trainer import cast
from hanser.train.v3.learner import Learner


class CNNLearner(Learner):

    def __init__(self, model, criterion, optimizer, **kwargs):
        super().__init__(model, criterion, optimizer, **kwargs)

    @tf.function(experimental_compile=True)
    def xla_train_batch(self, batch):
        model = self.model
        optimizer = self.optimizers[0]

        def local_step(inputs, target):
            with tf.GradientTape() as tape:
                inputs = cast(inputs, self.dtype)
                preds = model(inputs, training=True)
                preds = cast(preds, tf.float32)
                per_example_loss = self.criterion(target, preds)
                loss = self.reduce_loss(per_example_loss)
                if self.dtype == tf.float16:
                    loss = optimizer.get_scaled_loss(loss)
            grads = tape.gradient(loss, model.trainable_variables)
            return preds, per_example_loss, grads

        inputs, target = batch
        preds, per_example_loss, grads = local_step(inputs, target)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        self.update_metrics(self.train_metrics, target, preds, per_example_loss)


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
            if self.dtype == tf.float16:
                loss = optimizer.get_scaled_loss(loss)

        self.minimize(tape, optimizer, loss, model.trainable_variables)
        self.update_metrics(self.train_metrics, target, preds, per_example_loss)

    def eval_batch(self, batch):
        model = self.model

        inputs, target = batch
        inputs = cast(inputs, self.dtype)
        preds = model(inputs, training=False)
        preds = cast(preds, tf.float32)
        self.update_metrics(self.eval_metrics, target, preds)

    def test_batch(self, inputs):
        model = self.model

        inputs = cast(inputs, self.dtype)
        preds = model(inputs, training=False)
        preds = cast(preds, self.dtype)
        return preds
