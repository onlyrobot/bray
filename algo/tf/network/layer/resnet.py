import tensorflow.keras as keras
import tensorflow as tf

class Block(keras.Model):
    def __init__(
        self, channel_out, use_projection=False
    ):
        super(Block, self).__init__()
        self.use_projection = use_projection
        initializer = keras.initializers.TruncatedNormal(stddev=0.02)
        if use_projection:
            self.conv1 = keras.layers.Conv2D(channel_out, kernel_size=3, strides=1, padding="same", kernel_initializer=initializer, use_bias=False, activation=None)
            self.gn1 = GN(channel_out, channel_out)

        # 3 * 3
        self.conv2 = keras.layers.Conv2D(channel_out, kernel_size=3, strides=1, padding="same", kernel_initializer=initializer, use_bias=False, activation=None)
        self.gn2 = GN(channel_out, channel_out)


        self.relu2 = tf.nn.relu

        # 3 * 3
        self.conv3 = keras.layers.Conv2D(channel_out, kernel_size=3, strides=1, padding="same", kernel_initializer=initializer, use_bias=False, activation=None)
        self.gn3 = GN(channel_out, channel_out)

        self.relu3 = tf.nn.relu

    def call(self, x):
        shortcut = x
        if self.use_projection:
            shortcut = self.conv1(x)
            shortcut = self.gn1(shortcut)

        h = self.conv2(x)
        h = self.gn2(h)
        h = self.relu2(h)

        h = self.conv3(h)
        h = self.gn3(h)
        h = h + shortcut
        y = self.relu3(h)
        return y


class GN(keras.Model):
    def __init__(
        self, channel, G, esp=1e-5
    ):
        super(GN, self).__init__()
    
        self.gamma = tf.Variable(tf.constant(1.0, shape=[channel]), dtype=tf.float32, name='gamma')
        self.beta = tf.Variable(tf.constant(0.0, shape=[channel]), dtype=tf.float32, name='beta')
        self.esp = esp
        self.G = G

    def call(self, x):
        x = tf.transpose(x, [0, 3, 1, 2])
        N, C, H, W = x.get_shape().as_list()
        G = min(self.G, C)
        x = tf.reshape(x, [-1, G, C // G, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keepdims=True)
        # print('mean', mean)
        # print('var', var)
        x = (x - mean) / tf.sqrt(var + self.esp)
        # per channel gamma and beta
        gamma = tf.reshape(self.gamma, [1, C, 1, 1])
        beta = tf.reshape(self.beta, [1, C, 1, 1])

        output = tf.reshape(x, [-1, C, H, W]) * gamma + beta
        # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
        output = tf.transpose(output, [0, 2, 3, 1])
        return output