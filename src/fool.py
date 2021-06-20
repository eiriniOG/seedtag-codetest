import tensorflow as tf


class Fool:

    EPSILON = 0.2

    def __init__(self, idx_thing: int, image, probs, model):
        self.index = idx_thing
        self.pic = image
        self.pic_pred = probs
        self.model = model

    def _get_label(self):
        return tf.reshape(tf.one_hot(self.index, self.pic_pred.shape[-1]),
                          (1, self.pic_pred.shape[-1]))

    def _make_noise(self):
        loss_object = tf.keras.losses.CategoricalCrossentropy()

        with tf.GradientTape() as tape:
            tape.watch(self.pic)
            prediction = self.model(self.pic)
            loss = loss_object(self._get_label(), prediction)

        return tf.sign(tape.gradient(loss, self.pic))

    def _get_fake_pic(self):
        return tf.clip_by_value(self.pic + self.EPSILON * self._make_noise(),
                                -1, 1)

    def predict(self):
        return self.model.predict(self._get_fake_pic())
