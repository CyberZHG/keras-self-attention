# Keras自注意力

[![Version](https://img.shields.io/pypi/v/keras-self-attention.svg)](https://pypi.org/project/keras-self-attention/)
![License](https://img.shields.io/pypi/l/keras-self-attention.svg)

\[[中文](https://github.com/CyberZHG/keras-self-attention/blob/master/README.zh-CN.md)|[English](https://github.com/CyberZHG/keras-self-attention/blob/master/README.md)\]

Attention mechanism for processing sequential data that considers the context for each timestamp.

* ![](https://user-images.githubusercontent.com/853842/44248592-1fbd0500-a21e-11e8-9fe0-52a1e4a48329.gif)
* ![](https://user-images.githubusercontent.com/853842/44248591-1e8bd800-a21e-11e8-9ca8-9198c2725108.gif)
* ![](https://user-images.githubusercontent.com/853842/44248590-1df34180-a21e-11e8-8ff1-268217f466ba.gif)
* ![](https://user-images.githubusercontent.com/853842/44249018-8ba06d00-a220-11e8-80e3-802677b658ed.gif)

## 安装

```bash
pip install keras-self-attention
```

## 使用

### 基本

默认情况下，注意力层使用加性注意力机制，并使用全部上下文进行计算。下面的代码根据页首的公式创建了一个注意力层（`attention_activation`是注意力权重`e_{t, t'}`）：

```python
import keras
from keras_self_attention import SeqSelfAttention


model = keras.models.Sequential()
model.add(keras.layers.Embedding(input_dim=10000,
                                 output_dim=300,
                                 mask_zero=True))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=128,
                                                       return_sequences=True)))
model.add(SeqSelfAttention(attention_activation='sigmoid'))
model.add(keras.layers.Dense(units=5))
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy'],
)
model.summary()
```

### 局部注意力

参数`attention_width`控制着局部注意力的宽度：

```python
from keras_self_attention import SeqSelfAttention

SeqSelfAttention(
    attention_width=15,
    attention_activation='sigmoid',
    name='Attention',
)
```

### 乘性注意力

用`attention_type`来改变注意力机制的计算方法：

![](https://user-images.githubusercontent.com/853842/44253887-a03a3080-a233-11e8-9d49-3fd7e622a0f7.gif)

```python
from keras_self_attention import SeqSelfAttention

SeqSelfAttention(
    attention_width=15,
    attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
    attention_activation=None,
    kernel_regularizer=keras.regularizers.l2(1e-6),
    use_attention_bias=False,
    name='Attention',
)
```

### 正则化

![](https://user-images.githubusercontent.com/853842/44250188-f99b6300-a225-11e8-8fab-8dcf0d99616e.gif)

通过将`attention_regularizer_weight`设置为一个正数来使用正则化：

```python
import keras
from keras_self_attention import SeqSelfAttention

inputs = keras.layers.Input(shape=(None,))
embd = keras.layers.Embedding(input_dim=32,
                              output_dim=16,
                              mask_zero=True)(inputs)
lstm = keras.layers.Bidirectional(keras.layers.LSTM(units=16,
                                                    return_sequences=True))(embd)
att = SeqSelfAttention(attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
                       kernel_regularizer=keras.regularizers.l2(1e-4),
                       bias_regularizer=keras.regularizers.l1(1e-4),
                       attention_regularizer_weight=1e-4,
                       name='Attention')(lstm)
dense = keras.layers.Dense(units=5, name='Dense')(att)
model = keras.models.Model(inputs=inputs, outputs=[dense])
model.compile(
    optimizer='adam',
    loss={'Dense': 'sparse_categorical_crossentropy'},
    metrics={'Dense': 'categorical_accuracy'},
)
model.summary(line_length=100)
```

### 加载模型

Make sure to add `SeqSelfAttention` to custom objects:

```python
import keras

keras.models.load_model(model_path, custom_objects=SeqSelfAttention.get_custom_objects())
```

### 只使用历史进行计算

对于decoder等场景，为了保持输出固定只能使用上文的信息：

```python
SeqSelfAttention(
    attention_width=3,
    history_only=True,
    name='Attention',
)
```

### 多头注意力

参考[keras-multi-head](https://github.com/CyberZHG/keras-multi-head)。
