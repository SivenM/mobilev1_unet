import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from layers import UpSampleBlock, OutBlock, get_encoder


class MobileUnet(tf.keras.Model):
#need fix
    def __init__(self, input_size=(224, 224, 3)):
        super(MobileUnet, self).__init__()
        self.input_size = input_size
        self.encoder = self._get_encoder()
        self.up_stack = [
            UpSampleBlock(512, 3),  # (bs, 16, 16, 1024)
            UpSampleBlock(256, 3),  # (bs, 32, 32, 512)
            UpSampleBlock(128, 3),  # (bs, 64, 64, 256)
            UpSampleBlock(64, 3),  # (bs, 128, 128, 128)
        ]
        self.output_layer = OutBlock(32, 3)

    def _get_encoder(self):
        mobile_net = MobileNet(weights='imagenet',
                               include_top=False,
                               input_shape=self.input_shape)
        layer_names = [
            'conv_pw_1_relu',
            'conv_pw_3_relu',
            'conv_pw_5_relu',
            'conv_pw_11_relu',
            'conv_pw_13_relu',
        ]
        layers = [mobile_net.get_layer(name).output for name in layer_names]
        down_stack = tf.keras.Model(inputs=mobile_net.input, outputs=layers)
        down_stack.trainable = False
        return down_stack

    def call(self, inputs):
        x = inputs
        skips = self.encoder(inputs)
        x = skips[-1]
        skips = reversed(skips[:-1])
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            concat = tf.keras.layers.Concatenate()
            x = concat([x, skip])
        return self.output_layer(x)


def mobile_unet(input_shape=(224, 224, 3), deeper=False, separable=False):
    encoder = get_encoder(input_shape)
    up_stack = [
        UpSampleBlock(512, 3, deeper=deeper, separable=separable),  # (bs, 16, 16, 1024)
        UpSampleBlock(256, 3, deeper=deeper, separable=separable),  # (bs, 32, 32, 512)
        UpSampleBlock(128, 3, deeper=deeper, separable=separable),  # (bs, 64, 64, 256)
        UpSampleBlock(64, 3, deeper=deeper, separable=separable),  # (bs, 128, 128, 128)
    ]
    inputs = tf.keras.layers.Input(shape=input_shape)
    # Вход
    x = inputs
    # Энкодер
    skips = encoder(x)
    x = skips[-1]
    skips = reversed(skips[:-1])
    # Дкодер и конкатенация фичемапов
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])
    # Выходной блок
    x = OutBlock(32, 3)(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model
