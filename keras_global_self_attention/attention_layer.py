import keras
import keras.backend as K


class Attention(keras.layers.Layer):

    def __init__(self,
                 units=32,
                 return_attention=False,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        """Layer initialization.

        :param units: Dimension of the vectors that used to calculate the attention weights.
        :param return_attention: Whether return the attention weights for visualization.
        :param kernel_regularization: The regularization for weight matrices.
        :param bias_regularization: The regularization for biases.
        :param kwargs: Parameters for parent class.
        """
        self.supports_masking = True
        self.units = units
        self.return_attention = return_attention

        self.kernel_regularizer, self.bias_regularizer = kernel_regularizer, bias_regularizer

        self.Wx, self.Wt, self.bh = None, None, None
        self.Wa, self.ba = None, None

        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        feature_dim = input_shape[2]

        self.Wt = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Wt'.format(self.name),
                                  initializer=keras.initializers.get('glorot_normal'),
                                  regularizer=self.kernel_regularizer)
        self.Wx = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Wx'.format(self.name),
                                  initializer=keras.initializers.get('glorot_normal'),
                                  regularizer=self.kernel_regularizer)
        self.bh = self.add_weight(shape=(self.units,),
                                  name='{}_bh'.format(self.name),
                                  initializer=keras.initializers.get('zeros'),
                                  regularizer=self.bias_regularizer)

        self.Wa = self.add_weight(shape=(self.units, 1),
                                  name='{}_Wa'.format(self.name),
                                  initializer=keras.initializers.get('glorot_normal'),
                                  regularizer=self.kernel_regularizer)
        self.ba = self.add_weight(shape=(1,),
                                  name='{}_ba'.format(self.name),
                                  initializer=keras.initializers.get('zeros'),
                                  regularizer=self.bias_regularizer)

        self.trainable_weights = [self.Wx, self.Wt, self.bh, self.Wa, self.ba]
        super(Attention, self).build(input_shape)

    def call(self, inputs, mask=None, **kwargs):
        input_shape = K.shape(inputs)
        batch_size, input_len = input_shape[0], input_shape[1]
        # h_{t, t'} = \tanh(x_t^T W_t + x_{t'}^T W_x + b_h)
        q, k = K.dot(inputs, self.Wt), K.dot(inputs, self.Wx)
        q = K.tile(K.expand_dims(q, 2), K.stack([1, 1, input_len, 1]))
        k = K.tile(K.expand_dims(k, 1), K.stack([1, input_len, 1, 1]))
        h = K.tanh(q + k + self.bh)
        # e_{t, t'} = \sigma(W_a h_{t, t'} + b_a)
        e = K.sigmoid(K.reshape(K.dot(h, self.Wa) + self.ba, (batch_size, input_len, input_len)))
        # a_{t} = \text{softmax}(e_t)
        e = K.exp(e)
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask)
            e = K.permute_dimensions(K.permute_dimensions(e * mask, (0, 2, 1)) * mask, (0, 2, 1))
        s = K.sum(e, axis=-1)
        s = K.tile(K.expand_dims(s, axis=-1), K.stack([1, 1, input_len]))
        a = e / (s + K.epsilon())
        # l_t = \sum_{t'} a_{t, t'} x_{t'}
        inputs = K.permute_dimensions(inputs, (0, 2, 1))
        v = K.permute_dimensions(K.batch_dot(inputs, K.permute_dimensions(a, (0, 2, 1))), (0, 2, 1))
        if self.return_attention:
            return [v, a]
        return v

    def compute_output_shape(self, input_shape):
        if self.return_attention:
            return [input_shape, (input_shape[0], input_shape[1], input_shape[1])]
        return input_shape

    def compute_mask(self, inputs, mask=None):
        if self.return_attention:
            return [mask, None]
        return mask
