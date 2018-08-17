import os
import sys
import unittest
import random
import numpy
import keras
import keras.backend as K
from keras_self_attention import Attention
if sys.version_info[0] >= 3:
    import importlib


class TestLoss(unittest.TestCase):

    def setUp(self):
        self.backend = K.backend()

    def tearDown(self):
        TestLoss.set_keras_backend(self.backend)

    @staticmethod
    def set_keras_backend(backend):
        if K.backend() != backend:
            os.environ['KERAS_BACKEND'] = backend
            if sys.version_info[0] >= 3:
                import importlib
                importlib.reload(K)
            else:
                reload(K)

    def check_loss(self):
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
        input_data = numpy.asarray(input_data)
        inputs = keras.layers.Input(shape=(None,))
        embd = keras.layers.Embedding(input_dim=len(token_dict),
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
        model.summary(line_length=100)
        model.fit(
            x=input_data,
            y=[
                numpy.zeros((len(sentences), sentence_len, 1)),
                numpy.zeros((len(sentences), sentence_len, sentence_len))
            ],
            epochs=10,
        )

    def test_tensorflow_loss(self):
        TestLoss.set_keras_backend(Attention.BACKEND_TYPE_TENSORFLOW)
        self.check_loss()

    def test_theano_loss(self):
        TestLoss.set_keras_backend(Attention.BACKEND_TYPE_THEANO)
        self.check_loss()
