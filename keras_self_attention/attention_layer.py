import keras
import keras.backend as K


class Attention(keras.layers.Layer):

    ATTENTION_TYPE_ADD = 'additive'
    ATTENTION_TYPE_MUL = 'multiplicative'

    BACKEND_TYPE_TENSORFLOW = 'tensorflow'
    BACKEND_TYPE_THEANO = 'theano'

    def __init__(self,
                 units=32,
                 attention_width=None,
                 attention_type=ATTENTION_TYPE_ADD,
                 return_attention=False,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 use_additive_bias=True,
                 use_attention_bias=True,
                 attention_activation=None,
                 attention_regularizer_weight=0.0,
                 **kwargs):
        """Layer initialization.

        :param units: The dimension of the vectors that used to calculate the attention weights.
        :param attention_width: The width of local attention.
        :param attention_type: 'additive' or 'multiplicative'.
        :param return_attention: Whether to return the attention weights for visualization.
        :param kernel_regularization: The regularization for weight matrices.
        :param bias_regularization: The regularization for biases.
        :param use_additive_bias: Whether to use bias while calculating the relevance of inputs features
                                  in additive mode.
        :param use_attention_bias: Whether to use bias while calculating the weights of attention.
        :param attention_activation: The activation used for calculating the weights of attention.
        :param attention_regularizer_weight: The weights of attention regularizer.
        :param kwargs: Parameters for parent class.
        """
        self.supports_masking = True
        self.units = units
        self.attention_width = attention_width
        self.attention_type = attention_type
        self.return_attention = return_attention

        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.attention_activation = keras.activations.get(attention_activation)
        self.attention_regularizer_weight = attention_regularizer_weight
        self._backend = keras.backend.backend()

        if attention_type == Attention.ATTENTION_TYPE_ADD:
            self.Wx, self.Wt, self.bh = None, None, None
            self.Wa, self.ba = None, None
        elif attention_type == Attention.ATTENTION_TYPE_MUL:
            self.Wa, self.ba = None, None
        else:
            raise NotImplementedError('No implementation for attention type : ' + attention_type)

        super(Attention, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'units': self.units,
            'attention_width': self.attention_width,
            'attention_type': self.attention_type,
            'return_attention': self.return_attention,
            'use_additive_bias': self.use_additive_bias,
            'use_attention_bias': self.use_attention_bias,
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'attention_activation': keras.activations.serialize(self.attention_activation),
            'attention_regularizer_weight': self.attention_regularizer_weight,
        }
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if self.attention_type == Attention.ATTENTION_TYPE_ADD:
            self._build_additive_attention(input_shape)
        elif self.attention_type == Attention.ATTENTION_TYPE_MUL:
            self._build_multiplicative_attention(input_shape)
        super(Attention, self).build(input_shape)

    def _build_additive_attention(self, input_shape):
        feature_dim = input_shape[2]

        self.Wt = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wt'.format(self.name),
                                  initializer=keras.initializers.get('glorot_normal'),
                                  regularizer=self.kernel_regularizer)
        self.Wx = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wx'.format(self.name),
                                  initializer=keras.initializers.get('glorot_normal'),
                                  regularizer=self.kernel_regularizer)
        if self.use_additive_bias:
            self.bh = self.add_weight(shape=(self.units,),
                                      name='{}_Add_bh'.format(self.name),
                                      initializer=keras.initializers.get('zeros'),
                                      regularizer=self.bias_regularizer)

        self.Wa = self.add_weight(shape=(self.units, 1),
                                  name='{}_Add_Wa'.format(self.name),
                                  initializer=keras.initializers.get('glorot_normal'),
                                  regularizer=self.kernel_regularizer)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Add_ba'.format(self.name),
                                      initializer=keras.initializers.get('zeros'),
                                      regularizer=self.bias_regularizer)

    def _build_multiplicative_attention(self, input_shape):
        feature_dim = input_shape[2]

        self.Wa = self.add_weight(shape=(feature_dim, feature_dim),
                                  name='{}_Mul_Wa'.format(self.name),
                                  initializer=keras.initializers.get('glorot_normal'),
                                  regularizer=self.kernel_regularizer)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Mul_ba'.format(self.name),
                                      initializer=keras.initializers.get('zeros'),
                                      regularizer=self.bias_regularizer)

    def call(self, inputs, mask=None, **kwargs):
        if self._backend not in {Attention.BACKEND_TYPE_TENSORFLOW, Attention.BACKEND_TYPE_THEANO}:
            raise NotImplementedError('No implementation for backend : ' + K.backend())

        input_len = K.shape(inputs)[1]

        if self.attention_type == Attention.ATTENTION_TYPE_ADD:
            e = self._call_additive_emission(inputs)
        elif self.attention_type == Attention.ATTENTION_TYPE_MUL:
            e = self._call_multiplicative_emission(inputs)

        if self.attention_activation is not None:
            e = self.attention_activation(e)
        e = K.exp(e)
        if self.attention_width is not None:
            if self._backend == Attention.BACKEND_TYPE_TENSORFLOW:
                import tensorflow as tf
                ones = tf.ones((input_len, input_len))
                local = tf.matrix_band_part(ones, self.attention_width // 2, (self.attention_width - 1) // 2)
            elif self._backend == Attention.BACKEND_TYPE_THEANO:
                import theano.tensor as T
                ones = T.ones((input_len, input_len))
                local = T.triu(ones, -(self.attention_width // 2)) * T.tril(ones, (self.attention_width - 1) // 2)
            e = e * K.expand_dims(local, 0)
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask)
            e = K.permute_dimensions(K.permute_dimensions(e * mask, (0, 2, 1)) * mask, (0, 2, 1))

        # a_{t} = \text{softmax}(e_t)
        s = K.sum(e, axis=-1)
        s = self._tile(K.expand_dims(s, axis=-1), K.stack([1, 1, input_len]), ndim=3)
        a = e / (s + K.epsilon())

        # l_t = \sum_{t'} a_{t, t'} x_{t'}
        inputs = K.permute_dimensions(inputs, (0, 2, 1))
        v = K.permute_dimensions(K.batch_dot(inputs, K.permute_dimensions(a, (0, 2, 1))), (0, 2, 1))
        if self.attention_regularizer_weight > 0.0:
            self.add_loss(self._attention_regularizer(a))
        if self.return_attention:
            return [v, a]
        return v

    def _call_additive_emission(self, inputs):
        input_shape = K.shape(inputs)
        batch_size, input_len = input_shape[0], input_shape[1]

        # h_{t, t'} = \tanh(x_t^T W_t + x_{t'}^T W_x + b_h)
        q, k = K.dot(inputs, self.Wt), K.dot(inputs, self.Wx)
        q = self._tile(K.expand_dims(q, 2), K.stack([1, 1, input_len, 1]), ndim=4)
        k = self._tile(K.expand_dims(k, 1), K.stack([1, input_len, 1, 1]), ndim=4)
        if self.use_additive_bias:
            h = K.tanh(q + k + self.bh)
        else:
            h = K.tanh(q + k)

        # e_{t, t'} = W_a h_{t, t'} + b_a
        if self.use_attention_bias:
            e = K.reshape(K.dot(h, self.Wa) + self.ba, (batch_size, input_len, input_len))
        else:
            e = K.reshape(K.dot(h, self.Wa), (batch_size, input_len, input_len))
        return e

    def _call_multiplicative_emission(self, inputs):
        input_len = K.shape(inputs)[1]

        # e_{t, t'} = x_t^T W_a x_{t'} + b_a
        e = K.batch_dot(K.dot(inputs, self.Wa), K.permute_dimensions(inputs, (0, 2, 1)))
        if self.use_attention_bias:
            if self._backend == Attention.BACKEND_TYPE_THEANO:
                e = K.bias_add(e, K.reshape(K.repeat(K.expand_dims(self.ba, 0), input_len), (input_len,)))
            else:
                e = e + self.ba
        return e

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [input_shape, (input_shape[0], input_shape[1], input_shape[1])]
        return input_shape

    def compute_mask(self, inputs, mask=None):
        if self.return_attention:
            return [mask, None]
        return mask

    def _tile(self, x, n, ndim=None):
        if self._backend == Attention.BACKEND_TYPE_THEANO:
            import theano.tensor as T
            return T.tile(x, n, ndim=ndim)
        return K.tile(x, n)

    def _attention_regularizer(self, attention):
        batch_size = K.cast(K.shape(attention)[0], K.floatx())
        input_len = K.shape(attention)[-1]
        if K.backend() == Attention.BACKEND_TYPE_TENSORFLOW:
            import tensorflow as tf
            return self.attention_regularizer_weight * K.sum(K.square(K.batch_dot(
                attention,
                K.permute_dimensions(attention, (0, 2, 1))) - tf.eye(input_len))) / batch_size
        if K.backend() == Attention.BACKEND_TYPE_THEANO:
            import theano.tensor as T
            ones = T.ones((input_len, input_len))
            eye = T.triu(ones, 0) * T.tril(ones, 0)
            return self.attention_regularizer_weight * K.sum(K.square(K.batch_dot(
                attention,
                K.permute_dimensions(attention, (0, 2, 1))) - K.expand_dims(eye, 0))) / batch_size
