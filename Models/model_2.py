import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout
from config import config

class Model(tf.keras.Model):
    def __init__(self, name="model_2"):
        super().__init__(name=name)
        self.flatten = Flatten()
        self.dense_1 = Dense(32, activation='relu')
        self.dense_2 = Dense(64, activation='relu')
        self.dense_3 = Dense(16, activation='relu')
        self.dropout = Dropout(0.5)
        self.output_layer = Dense(config.NUM_CLASSES)

    def call(self, inputs):
        flatten_inputs = self.flatten(inputs)
        x = self.dense_1(flatten_inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x