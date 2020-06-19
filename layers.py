import tensorflow as tf


class UpSampleBlock(tf.keras.layers.Layer):
    """Conv2DTranspose => Batchnorm => Dropout => Relu"""
    def __init__(self, filters, size, apply_dropout=False):
        super(UpSampleBlock, self).__init__()
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.conv_transpose = tf.keras.layers.Conv2DTranspose(
                                      filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=self.initializer,
                                      use_bias=False
                                      )
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.relu = tf.nn.relu

