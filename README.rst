
Keras Self-Attention
====================


.. image:: https://travis-ci.org/CyberZHG/keras-self-attention.svg
   :target: https://travis-ci.org/CyberZHG/keras-self-attention
   :alt: Travis


.. image:: https://coveralls.io/repos/github/CyberZHG/keras-self-attention/badge.svg?branch=master
   :target: https://coveralls.io/github/CyberZHG/keras-self-attention
   :alt: Coverage


.. image:: https://img.shields.io/pypi/pyversions/keras-self-attention.svg
   :target: https://pypi.org/project/keras-self-attention/
   :alt: PyPI


.. image:: https://api.codacy.com/project/badge/Grade/5a99d0419bec42cfb73c4af06d746c8a
   :target: https://www.codacy.com/project/CyberZHG/keras-self-attention/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=CyberZHG/keras-self-attention&amp;utm_campaign=Badge_Grade_Dashboard
   :alt: Codacy Badge


Attention mechanism for processing sequence data that considers the context for each timestamp.


* 
  .. image:: https://user-images.githubusercontent.com/853842/44248592-1fbd0500-a21e-11e8-9fe0-52a1e4a48329.gif
     :target: https://user-images.githubusercontent.com/853842/44248592-1fbd0500-a21e-11e8-9fe0-52a1e4a48329.gif
     :alt: 

* 
  .. image:: https://user-images.githubusercontent.com/853842/44248591-1e8bd800-a21e-11e8-9ca8-9198c2725108.gif
     :target: https://user-images.githubusercontent.com/853842/44248591-1e8bd800-a21e-11e8-9ca8-9198c2725108.gif
     :alt: 

* 
  .. image:: https://user-images.githubusercontent.com/853842/44248590-1df34180-a21e-11e8-8ff1-268217f466ba.gif
     :target: https://user-images.githubusercontent.com/853842/44248590-1df34180-a21e-11e8-8ff1-268217f466ba.gif
     :alt: 

* 
  .. image:: https://user-images.githubusercontent.com/853842/44249018-8ba06d00-a220-11e8-80e3-802677b658ed.gif
     :target: https://user-images.githubusercontent.com/853842/44249018-8ba06d00-a220-11e8-80e3-802677b658ed.gif
     :alt: 

Install
-------

.. code-block:: bash

   pip install keras-self-attention

Usage
-----

Basic
^^^^^

By default, the attention layer uses additive attention and considers the whole context while calculating the relevance. The following code creates an attention layer that follows the equations in the first section (\ ``attention_activation`` is the activation function of ``e_{t, t'}``\ ):

.. code-block:: python

   import keras
   from keras_self_attention import Attention


   model = keras.models.Sequential()
   model.add(keras.layers.Embedding(input_dim=10000,
                                    output_dim=300,
                                    mask_zero=True))
   model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=128,
                                                          return_sequences=True)))
   model.add(Attention(attention_activation='sigmoid'))
   model.add(keras.layers.Dense(units=5))
   model.compile(
       optimizer='adam',
       loss='categorical_crossentropy',
       metrics=['categorical_accuracy'],
   )
   model.summary()

Local Attention
^^^^^^^^^^^^^^^

The global context may be too broad for one piece of data. The parameter ``attention_width`` controls the width of the local context:

.. code-block:: python

   from keras_self_attention import Attention

   Attention(
       attention_width=15,
       attention_activation='sigmoid',
       name='Attention',
   )

Multiplicative Attention
^^^^^^^^^^^^^^^^^^^^^^^^

You can use multiplicative attention by setting ``attention_type``\ :


.. image:: https://user-images.githubusercontent.com/853842/44253887-a03a3080-a233-11e8-9d49-3fd7e622a0f7.gif
   :target: https://user-images.githubusercontent.com/853842/44253887-a03a3080-a233-11e8-9d49-3fd7e622a0f7.gif
   :alt: 


.. code-block:: python

   from keras_self_attention import Attention

   Attention(
       attention_width=15,
       attention_type=Attention.ATTENTION_TYPE_MUL,
       attention_activation=None,
       kernel_regularizer=keras.regularizers.l2(1e-6),
       use_attention_bias=False,
       name='Attention',
   )

Regularizer
^^^^^^^^^^^


.. image:: https://user-images.githubusercontent.com/853842/44250188-f99b6300-a225-11e8-8fab-8dcf0d99616e.gif
   :target: https://user-images.githubusercontent.com/853842/44250188-f99b6300-a225-11e8-8fab-8dcf0d99616e.gif
   :alt: 


To use the regularizer, set ``attention_regularizer_weight`` to a positive number:

.. code-block:: python

   import keras
   from keras_self_attention import Attention

   inputs = keras.layers.Input(shape=(None,))
   embd = keras.layers.Embedding(input_dim=32,
                                 output_dim=16,
                                 mask_zero=True)(inputs)
   lstm = keras.layers.Bidirectional(keras.layers.LSTM(units=16,
                                                       return_sequences=True))(embd)
   att, weights = Attention(attention_type=Attention.ATTENTION_TYPE_MUL,
                            kernel_regularizer=keras.regularizers.l2(1e-4),
                            bias_regularizer=keras.regularizers.l1(1e-4),
                            attention_regularizer_weight=1e-4,
                            name='Attention')(lstm)
   dense = keras.layers.Dense(units=5, name='Dense')(att)
   model = keras.models.Model(inputs=inputs, outputs=[dense, weights])
   model.compile(
       optimizer='adam',
       loss={'Dense': 'sparse_categorical_crossentropy'},
       metrics={'Dense': 'categorical_accuracy'},
   )
   model.summary(line_length=100)
   model.fit(
       x=x,
       y=numpy.zeros((batch_size, sentence_len, 1)),,
       epochs=10,
   )

Load the Model
^^^^^^^^^^^^^^

Make sure to add ``Attention`` to custom objects:

.. code-block:: python

   import keras

   keras.models.load_model(model_path, custom_objects={
       'Attention': Attention,
   })
