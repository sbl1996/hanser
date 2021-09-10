import tensorflow as tf


class SelfEnsembleCrossEntropy:

    def __init__(self, aug_repeats, kd_temp=1, label_smoothing=0.0):
        self.aug_repeats = aug_repeats
        self.kd_temp = float(kd_temp)
        self.label_smoothing = label_smoothing

    def __call__(self, y_true, y_pred):
        R = self.aug_repeats
        T = self.kd_temp
        shape = tf.shape(y_pred)
        probs = tf.nn.softmax(y_pred, axis=1)
        teacher_label = tf.reduce_mean(tf.reshape(probs, [shape[0] // R, R, shape[1]]), axis=1, keepdims=True)
        teacher_label = tf.reshape(tf.tile(teacher_label, [1, R, 1]), shape)
        loss = tf.keras.losses.categorical_crossentropy(
            y_true, y_pred, from_logits=True, label_smoothing=self.label_smoothing)
        loss_kd = T * T * tf.keras.losses.categorical_crossentropy(
            teacher_label, y_pred / T, from_logits=False)
        return loss + loss_kd
