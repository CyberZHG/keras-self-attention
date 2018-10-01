import keras
import numpy as np
from keras_self_attention import SeqSelfAttention
from .util import TestMaskShape


class TestHistory(TestMaskShape):

    def test_history(self):
        attention = SeqSelfAttention(return_attention=True,
                                     attention_width=3,
                                     history_only=True,
                                     kernel_regularizer=keras.regularizers.l2(1e-4),
                                     bias_regularizer=keras.regularizers.l1(1e-4),
                                     use_additive_bias=False,
                                     use_attention_bias=False,
                                     attention_activation='relu',
                                     name='Attention')
        self.check_mask_shape(attention)

    def check_mask_shape(self, attention):
        sentences, input_data, token_dict = self.get_input_data()
        model = self.get_model(attention, token_dict)
        attention_width = attention.attention_width
        outputs = model.predict(input_data)
        attention = outputs[1]
        sentence_len = input_data.shape[1]
        for i, sentence in enumerate(sentences):
            for j in range(sentence_len):
                for k in range(sentence_len):
                    if j < len(sentence) and k < len(sentence) and 0 <= j - k < attention_width:
                        self.assertGreater(attention[i][j][k], 0.0)
                    else:
                        self.assertEqual(attention[i][j][k], 0.0)
                if j < len(sentence):
                    self.assertTrue(abs(np.sum(attention[i][j]) - 1.0) < 1e-6)
