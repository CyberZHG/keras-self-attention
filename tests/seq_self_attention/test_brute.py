import random
import keras
import keras.backend as K
import numpy
import tensorflow as tf
import unittest
from keras_self_attention import SeqSelfAttention


class SelfAttentionBrute(keras.engine.Layer):

    def __init__(self, units=32, **kwargs):
        """
        :param attention_dim: The dimension of h_a.
        :param kwargs: Parameters for parent class.
        """
        self.units = units
        self.Wt, self.Wx, self.bh = None, None, None
        self.Wa, self.ba = None, None
        super(SelfAttentionBrute, self).__init__(** kwargs)

    def build(self, input_shape):
        feature_dim = input_shape[2]

        self.Wt = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wt'.format(self.name),
                                  initializer=keras.initializers.get('glorot_normal'))
        self.Wx = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wx'.format(self.name),
                                  initializer=keras.initializers.get('glorot_normal'))
        self.bh = self.add_weight(shape=(self.units,),
                                  name='{}_Add_bh'.format(self.name),
                                  initializer=keras.initializers.get('zeros'))

        self.Wa = self.add_weight(shape=(self.units, 1),
                                  name='{}_Add_Wa'.format(self.name),
                                  initializer=keras.initializers.get('glorot_normal'))
        self.ba = self.add_weight(shape=(1,),
                                  name='{}_Add_ba'.format(self.name),
                                  initializer=keras.initializers.get('zeros'))
        super(SelfAttentionBrute, self).build(input_shape)

    def call(self, x, mask=None):
        v = K.map_fn(self.calc_sample, x)
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            v = v * mask
        return v

    def calc_sample(self, s):
        return K.map_fn(lambda ht: self.calc_time(s, ht), s)

    def calc_time(self, h, ht):
        time_len = K.shape(h)[0]
        k, q = K.dot(h, self.Wx), K.dot(K.expand_dims(ht, 0), self.Wt)
        ha = K.tanh(k + q + self.bh)
        e = K.exp(K.dot(ha, self.Wa) + self.ba)
        a = e / (K.sum(e) + K.epsilon())
        v = K.sum(K.transpose(h) * K.reshape(a, (time_len,)), axis=1)
        return v


class TestBrute(unittest.TestCase):

    @staticmethod
    def _reset_seed(seed):
        random.seed(seed)
        numpy.random.seed(seed)
        tf.set_random_seed(seed)

    def test_same_as_brute(self):
        batch_size, sentence_len, feature_dim, units = 2, 3, 5, 7
        test_x = numpy.random.rand(batch_size, sentence_len, feature_dim)

        seed = random.randint(0, 1000)
        self._reset_seed(seed)
        inp = keras.layers.Input((sentence_len, feature_dim))
        att = SeqSelfAttention(units=units, kernel_initializer='glorot_normal', bias_initializer='zeros')
        out = att(inp)
        model = keras.models.Model(inp, out)
        predict_1 = model.predict(test_x)
        self.assertEqual((batch_size, sentence_len, feature_dim), predict_1.shape)

        self._reset_seed(seed)
        inp = keras.layers.Input((sentence_len, feature_dim))
        att = SelfAttentionBrute(units=units)
        out = att(inp)
        model = keras.models.Model(inp, out)
        predict_2 = model.predict(test_x)
        self.assertEqual((batch_size, sentence_len, feature_dim), predict_2.shape)

        self.assertTrue(numpy.allclose(predict_1, predict_2))
