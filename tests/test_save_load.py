import os
import tempfile
import random
import keras
from keras_self_attention import Attention
from .util import TestMaskShape


class TestSaveLoad(TestMaskShape):

    def test_save_load(self):
        _, _, token_dict = self.get_input_data()
        model = self.get_model(Attention(name='Attention'), token_dict)
        model_path = os.path.join(tempfile.gettempdir(), 'keras_self_att_test_save_load_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={'Attention': Attention})
        model.summary()
        self.assertTrue(model is not None)

    def test_save_load_with_loss(self):
        attention = Attention(return_attention=True,
                              attention_width=7,
                              attention_type=Attention.ATTENTION_TYPE_MUL,
                              kernel_regularizer=keras.regularizers.l2(1e-4),
                              bias_regularizer=keras.regularizers.l1(1e-4),
                              attention_regularizer_weight=1e-3,
                              name='Attention')
        _, _, token_dict = self.get_input_data()
        model = self.get_model(attention, token_dict, return_attention=True)
        model_path = os.path.join(tempfile.gettempdir(), 'keras_self_att_test_sl_with_loss_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={'Attention': Attention})
        model.summary()
        self.assertTrue(model is not None)
