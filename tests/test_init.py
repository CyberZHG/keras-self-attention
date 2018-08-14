import unittest
import keras
from keras_global_self_attention import Attention


class TestInit(unittest.TestCase):

    def test_init(self):
        model = keras.models.Sequential()
        model.add(keras.layers.Embedding(input_dim=10000,
                                         output_dim=300,
                                         mask_zero=True))
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=128,
                                                               return_sequences=True)))
        model.add(Attention())
        model.add(keras.layers.Dense(units=5))
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['categorical_accuracy'],
        )
        model.summary()

    def test_return_attention(self):
        inputs = keras.layers.Input(shape=(None,))
        embd = keras.layers.Embedding(input_dim=10000,
                                      output_dim=300,
                                      mask_zero=True)(inputs)
        lstm = keras.layers.Bidirectional(keras.layers.LSTM(units=128,
                                                            return_sequences=True))(embd)
        att, weights = Attention(return_attention=True,
                                 kernel_regularizer=keras.regularizers.l2(1e-4),
                                 bias_regularizer=keras.regularizers.l1(1e-4))(lstm)
        dense = keras.layers.Dense(units=5)(att)
        model = keras.models.Model(inputs=inputs, outputs=[dense, weights])
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            loss_weights=[1.0, 0.0],
            metrics=['categorical_accuracy'],
        )
        model.summary()
