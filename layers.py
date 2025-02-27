import tensorflow as tf
from tensorflow.keras.applications import MobileNet, ResNet50


class UpSampleBlock(tf.keras.layers.Layer):
    """Conv2DTranspose => Batchnorm => Dropout => Relu"""

    def __init__(self, filters, size, apply_dropout=False, deeper=False, separable=False):
        super(UpSampleBlock, self).__init__()
        self.separable = separable
        self.apply_dropout = apply_dropout
        self.deeper = deeper
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.conv_transpose = tf.keras.layers.Conv2DTranspose(
            filters, size, strides=2,
            padding='same',
            kernel_initializer=self.initializer,
            use_bias=False
        )
        self.conv = tf.keras.layers.Conv2D(
            filters,
            size,
            activation='relu',
            padding='same',
            kernel_initializer='he_normal')
        self.sep_conv = tf.keras.layers.SeparableConv2D(
            filters,
            size,
            activation='relu',
            padding='same', )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.relu = tf.nn.relu

    def call(self, inputs, training=False):
        x = self.conv_transpose(inputs)
        x = self.batch_norm(x, training=training)
        if self.apply_dropout:
            x = self.dropout(x)
        x = self.relu(x)
        if self.deeper:
            if self.separable:
                x = self.sep_conv(x)
                x = self.sep_conv(x)
                x = self.sep_conv(x)
            else:
                x = self.conv(x)
                x = self.conv(x)
        return x


class OutBlock(tf.keras.layers.Layer):
    """Conv2DTranspose => Conv2D => softmax"""

    def __init__(self, filters, size, num_classes=2, separable=False):
        super(OutBlock, self).__init__()
        self.separable = separable
        self.initializer = tf.random_normal_initializer(0., 0.02)
        self.conv_transpose = tf.keras.layers.Conv2DTranspose(
            filters, size, strides=2,
            padding='same',
            kernel_initializer=self.initializer,
            use_bias=False
        )
        self.sep_conv = tf.keras.layers.SeparableConv2D(
            filters,
            size,
            activation='relu',
            padding='same', )
        self.conv2D = tf.keras.layers.Conv2D(num_classes, 1, activation='softmax')

    def call(self, inputs):
        x = self.conv_transpose(inputs)
        if self.separable:
            x = self.sep_conv(x)
            x = self.sep_conv(x)
        return self.conv2D(x)


def mobilenet_encoder(input_shape=[224, 224, 3]):
    mn = MobileNet(weights='imagenet',
                   include_top=False,
                   input_shape=input_shape)
    mn.trainable = False
    layer_names = [
        'conv_pw_1_relu',
        'conv_pw_3_relu',
        'conv_pw_5_relu',
        'conv_pw_11_relu',
        'conv_pw_13_relu',
    ]
    layers = [mn.get_layer(name).output for name in layer_names]
    down_stack = tf.keras.Model(inputs=mn.input, outputs=layers)
    down_stack.trainable = False
    return down_stack


def resnet50_encoder(input_shape=[224, 224, 3]):
    resnet50 = ResNet50(weights='imagenet',
                        include_top=False,
                        input_shape=(224, 224, 3))

    layers_names = [
        'conv1_relu',
        'conv2_block3_out',
        'conv3_block4_out',
        'conv4_block6_out',
        'conv5_block3_out'
    ]
    layers = [resnet50.get_layer(name).output for name in layers_names]
    down_stack = tf.keras.Model(inputs=resnet50.input, outputs=layers)
    down_stack.trainable = False
    return down_stack
