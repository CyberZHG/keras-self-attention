# Keras Global Self-Attention

[![Travis](https://travis-ci.org/PoWWoP/keras-global-self-attention.svg)](https://travis-ci.org/PoWWoP/keras-global-self-attention)
[![Coverage](https://coveralls.io/repos/github/PoWWoP/keras-global-self-attention/badge.svg?branch=master)](https://coveralls.io/github/PoWWoP/keras-global-self-attention)


Attention mechanism for processing sequence data that considers the global context for each timestamp.

* ![](http://latex.codecogs.com/gif.latex?h_{t,&space;t'}&space;=&space;\tanh(x_t^T&space;W_t&space;&plus;&space;x_{t'}^T&space;W_x&space;&plus;&space;b_t))
* ![](http://latex.codecogs.com/gif.latex?e_{t,&space;t'}&space;=&space;\sigma(W_a&space;h_{t,&space;t'}&space;&plus;&space;b_a))
* ![](http://latex.codecogs.com/gif.latex?a_{t}&space;=&space;\text{softmax}(e_t))

## Install

```bash
pip install keras-global-self-attention
```

## Usage

```python
import keras
from keras_global_self_attention import Attention


model = keras.models.Sequential()
model.add(keras.layers.Embedding(input_dim=10000,
                                 output_dim=300,
                                 mask_zero=True))
model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=128,
                                                       return_sequences=True)))
model.add(Attention())
model.add(keras.layers.Dense(units=5))
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy'],
)
model.summary()
```
