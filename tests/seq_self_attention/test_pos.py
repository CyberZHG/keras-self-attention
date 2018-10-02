import keras
import random
import numpy as np
from keras_self_attention import SeqSelfAttention
from .util import TestMaskShape


class TestPos(TestMaskShape):

    @staticmethod
    def get_pos_model(attention, token_dict, pos_num):
        input_data = keras.layers.Input(shape=(None,), name='Input-Data')
        input_pos = keras.layers.Input(shape=(pos_num,), name='Input-Pos')
        embd = keras.layers.Embedding(input_dim=len(token_dict),
                                      output_dim=32,
                                      mask_zero=True,
                                      name='Embedding')(input_data)
        lstm = keras.layers.Bidirectional(keras.layers.LSTM(units=16,
                                                            return_sequences=True),
                                          name='Bi-LSTM')(embd)
        att, weights = attention([lstm, input_pos])
        dense = keras.layers.Dense(units=5, name='Dense')(att)
        model = keras.models.Model(inputs=[input_data, input_pos], outputs=[dense, weights])
        model.compile(
            optimizer='adam',
            loss={
                'Dense': 'sparse_categorical_crossentropy',
                'Attention': 'mse',
            },
            metrics={'Dense': 'sparse_categorical_accuracy'},
        )
        model.summary(line_length=120)
        return model

    def test_position(self):
        pos_num = random.randint(1, 4)
        attention_layer = SeqSelfAttention(return_attention=True,
                                           name='Attention')
        sentences, input_data, token_dict = self.get_input_data()
        model = self.get_pos_model(attention_layer, token_dict, pos_num)
        input_pos = np.asarray([[random.randint(0, len(s) - 1) for _ in range(pos_num)] for s in sentences])
        outputs, attention = model.predict([input_data, input_pos])
        self.assertEqual(pos_num, outputs.shape[1])
        self.assertEqual(pos_num, attention.shape[1])
