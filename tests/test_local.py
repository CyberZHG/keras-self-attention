import keras
from keras_self_attention import SelfAttention
from .util import TestMaskShape


class TestLocal(TestMaskShape):

    def check_local_range(self, attention_type):
        attention = SelfAttention(return_attention=True,
                                  attention_width=5,
                                  attention_type=attention_type,
                                  kernel_regularizer=keras.regularizers.l2(1e-4),
                                  bias_regularizer=keras.regularizers.l1(1e-4),
                                  name='Attention')
        self.check_mask_shape(attention)

    def test_add(self):
        self.check_local_range(attention_type=SelfAttention.ATTENTION_TYPE_ADD)

    def test_mul(self):
        self.check_local_range(attention_type=SelfAttention.ATTENTION_TYPE_MUL)

    def test_not_implemented(self):
        with self.assertRaises(NotImplementedError):
            inputs = keras.layers.Input(shape=(None,))
            embd = keras.layers.Embedding(input_dim=40,
                                          output_dim=18,
                                          mask_zero=True)(inputs)
            lstm = keras.layers.Bidirectional(keras.layers.LSTM(units=18,
                                                                return_sequences=True))(embd)
            att = SelfAttention(attention_width=15)
            att._backend = 'random'
            att = att(lstm)
            dense = keras.layers.Dense(units=5)(att)
            model = keras.models.Model(inputs=inputs, outputs=dense)
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['categorical_accuracy'],
            )
