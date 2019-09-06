import unittest
import random
import numpy
import keras
from keras_self_attention import SelfAttention


class TestMaskShape(unittest.TestCase):

    @staticmethod
    def get_input_data():
        sentences = [
            ['All', 'work', 'and', 'no', 'play'],
            ['makes', 'Jack', 'a', 'dull', 'boy', '.'],
            ['From', 'that', 'day', 'forth', 'my', 'arm', 'changed'],
        ]
        token_dict = {
            '': 0,
            '<UNK>': 1,
        }
        sentence_len = max(map(len, sentences))
        input_data = [[0] * sentence_len for _ in range(len(sentences))]
        for i, sentence in enumerate(sentences):
            for j, token in enumerate(sentence):
                if token in token_dict:
                    input_data[i][j] = token_dict[token]
                elif random.randint(0, 5) == 0:
                    input_data[i][j] = token_dict['<UNK>']
                else:
                    input_data[i][j] = len(token_dict)
                    token_dict[token] = len(token_dict)
        return sentences, numpy.asarray(input_data), token_dict

    @staticmethod
    def get_model(attention, token_dict, return_attention=False):
        inputs = keras.layers.Input(shape=(None,), name='Input')
        embd = keras.layers.Embedding(input_dim=len(token_dict),
                                      output_dim=16,
                                      mask_zero=True,
                                      name='Embedding')(inputs)
        lstm = keras.layers.Bidirectional(keras.layers.LSTM(units=16,
                                                            return_sequences=True),
                                          name='Bi-LSTM')(embd)
        if attention.return_attention:
            att, weights = attention(lstm)
        else:
            att = attention(lstm)
        dense = keras.layers.Dense(units=5, name='Dense')(att)
        if attention.return_attention:
            model = keras.models.Model(inputs=inputs, outputs=[dense, weights])
        else:
            model = keras.models.Model(inputs=inputs, outputs=dense)
        model.compile(
            optimizer='adam',
            loss={'Dense': 'sparse_categorical_crossentropy'},
            metrics={'Dense': 'categorical_accuracy'},
        )
        model.summary(line_length=100)
        return model

    def check_mask_shape(self, attention):
        sentences, input_data, token_dict = self.get_input_data()
        model = self.get_model(attention, token_dict)
        outputs = model.predict(input_data)
        if attention.attention_width is None:
            attention_width = 1e9
        else:
            attention_width = attention.attention_width // 2
        attention = outputs[1]
        sentence_len = input_data.shape[1]
        for i, sentence in enumerate(sentences):
            for j in range(sentence_len):
                for k in range(sentence_len):
                    if j < len(sentence) and k < len(sentence) and abs(j - k) <= attention_width:
                        self.assertGreater(attention[i][j][k], 0.0)
                    else:
                        self.assertEqual(attention[i][j][k], 0.0)
                if j < len(sentence):
                    self.assertTrue(abs(numpy.sum(attention[i][j]) - 1.0) < 1e-6)
