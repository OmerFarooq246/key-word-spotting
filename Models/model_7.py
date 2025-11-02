import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Conv1D, GlobalAveragePooling1D
from config import config

class Model(tf.keras.Model):
    def __init__(self, name="model_7"):
        super().__init__(name=name)
        self.conv_1 = Conv1D(
            filters=16,
            kernel_size=3,
            activation='relu'
        )
        self.conv_2 = Conv1D(
            filters=32,
            kernel_size=3,
            activation='relu'
        )
        self.conv_3 = Conv1D(
            filters=64,
            kernel_size=3,
            activation='relu'
        )
        self.dropout = Dropout(0.5)
        self.gap = GlobalAveragePooling1D()
        self.output_layer = Dense(config.NUM_CLASSES)

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.dropout(x)
        x = self.gap(x)
        x = self.output_layer(x)
        return x