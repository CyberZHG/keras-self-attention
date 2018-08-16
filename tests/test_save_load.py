import unittest
import os
import random
import tempfile
import keras
from keras_self_attention import Attention


class TestSaveLoad(unittest.TestCase):

    def test_save_load(self):
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
        model_path = os.path.join(tempfile.gettempdir(), 'keras_global_self_att_test_save_load_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={'Attention': Attention})
        model.summary()
