import tensorflow as tf


class AtariModel(tf.keras.Model):
    def __init__(self, action_space=9):
        super().__init__()
        self.base_net = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=16,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    activation="relu",
                    input_shape=(42, 42, 4),
                ),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=4,
                    strides=2,
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.Conv2D(
                    filters=256,
                    kernel_size=11,
                    strides=1,
                    padding="valid",
                    activation="relu",
                ),
                tf.keras.layers.Flatten(),
            ]
        )
        self.values_net = tf.keras.layers.Dense(1)
        self.logits_net = tf.keras.layers.Dense(action_space)

    def call(self, state):
        # transpose [None, 42, 42, 4] into [None, 4, 42, 42]
        # images = tf.transpose(images, perm=[0, 3, 1, 2])
        hidden = self.base_net(state["image"])
        values = tf.squeeze(self.values_net(hidden), axis=1)
        logits = self.logits_net(hidden)
        probs = tf.nn.softmax(logits, axis=1)
        actions = tf.random.categorical(
            tf.math.log(probs),
            num_samples=1,
        )
        actions = tf.squeeze(actions, axis=1)
        return values, logits, actions


if __name__ == "__main__":
    model = AtariModel()
    print(model({"image": tf.random.normal((2, 42, 42, 4))}))
    model.summary()
