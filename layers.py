import tensorflow as tf


class UpSampleBlock(tf.keras.layers.Layer):
    """Conv2DTranspose => Batchnorm => Dropout => Relu"""
    def __init__(self, filters, size, apply_dropout=False):
        super(UpSampleBlock, self).__init__()
        self.apply_dropout = apply_dropout
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.conv_transpose = tf.keras.layers.Conv2DTranspose(
                                      filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=self.initializer,
                                      use_bias=False
                                      )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.relu = tf.nn.relu

    def call(self, inputs):
        x = self.conv_transpose(inputs)
        x = self.batch_norm(x)
        if self.apply_dropout:
            x = self.dropout(x)
        return self.relu(x)


class OutBlock(tf.keras.layers.Layer):
    """Conv2DTranspose => Conv2D => softmax"""
    def __init__(self, filters, size, num_classes=2):
        super(OutBlock, self).__init__()
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.conv_transpose = tf.keras.layers.Conv2DTranspose(
                                      filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=self.initializer,
                                      use_bias=False
                                      )
        self.conv2D = tf.keras.layers.Conv2D(num_classes, 1, activation='softmax')

    def call(self, inputs):
        x = self.conv_transpose(inputs)
        return self.conv2D(x)