import tensorflow as tf

from hanser.train.learner import Learner, cast


class SuperLearner(Learner):

    def __init__(self, model, criterion, optimizer,
                 batch_transform=None, batches_transform=None,
                 **kwargs):
        self.batch_transform = batch_transform
        self.batches_transform = batches_transform
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
        self.minimize(tape, optimizer, loss, model.trainable_variables)
        self.update_metrics(self.train_metrics, target, preds, per_example_loss)

    def train_batches(self, *batches):
        batch = self.batches_transform(*batches)
        return self.train_batch(batch)

    def eval_batch(self, batch):
        model = self.model

        inputs, target = batch
        inputs = cast(inputs, self.dtype)
        preds = model(inputs, training=False)
        preds = cast(preds, tf.float32)
        self.update_metrics(self.eval_metrics, target, preds)

    def simple_eval_batch(self, batch):
        model = self.model

        inputs, target = batch
        inputs = cast(inputs, self.dtype)
        preds = model(inputs, training=False)
        preds = cast(preds, tf.float32)
        return target, preds

    def test_batch(self, inputs):
        model = self.model

        inputs = cast(inputs, self.dtype)
        preds = model(inputs, training=False)
        preds = cast(preds, tf.float32)
        return preds

CNNLearner = SuperLearner