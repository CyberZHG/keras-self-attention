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

        model_path = os.path.join(tempfile.gettempdir(), 'keras_self_att_test_save_load_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={'Attention': Attention})
        model.summary()
        self.assertTrue(model is not None)

    def test_save_load_with_loss(self):
        inputs = keras.layers.Input(shape=(None,))
        embd = keras.layers.Embedding(input_dim=1000,
                                      output_dim=16,
                                      mask_zero=True)(inputs)
        lstm = keras.layers.Bidirectional(keras.layers.LSTM(units=16,
                                                            return_sequences=True))(embd)
        att, weights = Attention(return_attention=True,
                                 attention_width=5,
                                 attention_type=Attention.ATTENTION_TYPE_MUL,
                                 kernel_regularizer=keras.regularizers.l2(1e-4),
                                 bias_regularizer=keras.regularizers.l1(1e-4),
                                 name='Attention')(lstm)
        dense = keras.layers.Dense(units=5, name='Dense')(att)
        model = keras.models.Model(inputs=inputs, outputs=[dense, weights])
        model.compile(
            optimizer='adam',
            loss={'Dense': 'sparse_categorical_crossentropy', 'Attention': Attention.loss(1e-2)},
            metrics={'Dense': 'categorical_accuracy'},
        )

        model.summary()
        model_path = os.path.join(tempfile.gettempdir(), 'keras_self_att_test_sl_with_loss_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={
            'Attention': Attention,
            'attention_regularizer': Attention.loss(1e-2),
        })
        model.summary()
        self.assertTrue(model is not None)
