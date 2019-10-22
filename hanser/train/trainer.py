import tensorflow as tf

class Trainer:

    def __init__(self, model_fn, train_batch_size, eval_batch_size,
                 steps_per_epoch, eval_steps_per_epoch, save_per_epochs=1,
                 tpu=None, model_params=None, model_dir=None):

        self.model_fn = model_fn
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.steps_per_epoch = steps_per_epoch
        self.eval_steps_per_epoch = eval_steps_per_epoch
        self.save_per_epochs = save_per_epochs
        self.tpu = tpu
        self.use_tpu = bool(tpu)
        self.model_params = model_params
        self.model_dir = model_dir

        training_config = tf.contrib.tpu.RunConfig(
            cluster=tpu,
            model_dir=model_dir,
            save_summary_steps=steps_per_epoch,
            save_checkpoints_steps=steps_per_epoch * save_per_epochs,
            keep_checkpoint_max=1,
            log_step_count_steps=steps_per_epoch,
            tpu_config=tf.contrib.tpu.TPUConfig(steps_per_epoch))

        self.estimator = tf.contrib.tpu.TPUEstimator(
            model_fn=model_fn,
            model_dir=model_dir,
            params={ **model_params, 'steps_per_epoch': steps_per_epoch },
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            predict_batch_size=eval_batch_size,
            config=training_config,
            use_tpu=self.use_tpu,
            export_to_tpu=False)

    def train(self, input_fn, epochs):
        return self.estimator.train(input_fn, steps=epochs * self.steps_per_epoch)

    def evaluate(self, input_fn):
        return self.estimator.evaluate(input_fn, steps=self.eval_steps_per_epoch)
