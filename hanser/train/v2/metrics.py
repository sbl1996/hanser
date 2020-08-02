from toolz.curried import get
from tensorflow.keras.metrics import Metric, Mean


class TrainLoss(Mean):

    def __init__(self, name='loss', dtype=None):
        super().__init__(name, dtype)

    @staticmethod
    def output_transform(output):
        per_example_losses = get("per_example_losses", output)
        return per_example_losses


class Loss(Mean):

    def __init__(self, criterion, name="loss", dtype=None):
        self.criterion = criterion
        super().__init__(name=name, dtype=dtype)

    def output_transform(self, output):
        y_true, y_pred = get(["y_true", "y_pred"], output)
        loss = self.criterion(y_pred, y_true)
        return loss