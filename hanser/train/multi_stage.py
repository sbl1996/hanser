import tensorflow as tf
from hanser.train.callbacks import Callback
from hanser.hpo.repeat import repeat

class StageEndException(RuntimeError):
    pass

class TrainEndException(RuntimeError):
    pass

class StageControl(Callback):

    priority = -1

    def __init__(self, epochs, n_stages):
        super().__init__()
        self.epochs = epochs
        self.n_stages = n_stages

    def after_eval(self, state):
        epoch = self.learner.epoch + 1
        if epoch % (self.epochs // self.n_stages) == 0 and epoch != self.epochs:
            raise StageEndException
        if epoch == self.epochs:
            raise TrainEndException


def train_multi_stage(train_fn):
    train_fn_ = lambda _i: train_fn()
    repeat(train_fn_, 1000, catch=(tf.errors.UnavailableError, StageEndException))