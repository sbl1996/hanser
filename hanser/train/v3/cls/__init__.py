import tensorflow as tf

from hanser.train.trainer import cast_fp32
from hanser.train.v3.learner import Learner


class CNNLearner(Learner):

    def __init__(self, model, criterion, optimizer, **kwargs):
        super().__init__(model, criterion, optimizer, **kwargs)

    def train_batch(self, batch):
        model = self.model
        optimizer = self.optimizers[0]

        inputs, target = batch
        with tf.GradientTape() as tape:
            preds = model(inputs, training=True)
            if self.fp16:
                preds = cast_fp32(preds)
            per_example_loss = self.criterion(target, preds)
            loss = self.reduce_loss(per_example_loss)

        self.minimize(tape, optimizer, loss, model.trainable_variables)
        self.update_metrics(self.train_metrics, target, preds, per_example_loss)

    def eval_batch(self, batch):
        model = self.model

        inputs, target = batch
        preds = model(inputs, training=False)
        if self.fp16:
            preds = cast_fp32(preds)
        self.update_metrics(self.eval_metrics, target, preds)

    # def test_batch(self, batch):
    #     state = self._state['test']
    #     model = self.model
    #
    #     model.eval()
    #     input = convert_tensor(batch, self.device)
    #     with torch.no_grad():
    #         output = model(input)
    #
    #     state.update({
    #         "batch_size": input.size(0),
    #         "y_pred": output,
    #     })