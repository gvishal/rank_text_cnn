
# coding: utf-8

# In[54]:

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.layers import Input, Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers.merge import Concatenate, Dot
from keras import regularizers
import numpy as np


# In[55]:

ques_max_len = 20
ques_vocab_size = 1000
ans_max_len = 40
ans_vocab_size = 1000
embedding = np.random.rand(1000, 50)


# In[56]:

# Prepare layers for Question
input_q = Input(shape=(ques_max_len,))
# print input_q.name
# we will load embedding values from corpus here.
embed_q = Embedding(input_dim=ques_vocab_size, output_dim=50, input_length=ques_max_len,
                    weights=[embedding], trainable=False)(input_q)
# Padding means, if input size is 32x32, output will also be 32x32, i.e, the dimensions
# will not reduce
conv_q = Conv1D(filters=100, kernel_size=5, strides=1, padding='same',
                activation='relu', kernel_regularizer=regularizers.l2(1e-5))(embed_q)
# also referenced as x(q) in paper
pool_q = GlobalMaxPooling1D()(conv_q)


# In[57]:

# Prepare layers for Answer
input_a = Input(shape=(ans_max_len,))
# print input_a.name
embed_a = Embedding(input_dim=ans_vocab_size, output_dim=50, input_length=ans_max_len,
                    weights=[embedding], trainable=False)(input_a)
conv_a = Conv1D(filters=100, kernel_size=5, strides=1, padding='same',
                activation='relu', kernel_regularizer=regularizers.l2(1e-5))(embed_a)
pool_a = GlobalMaxPooling1D()(conv_a)


# In[58]:

# M or the similarity layer here
# Paper: x_d_dash = M.x_d
x_a = Dense(100, use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(pool_a)
sim = Dot(axes=-1)([pool_q, x_a])


# In[59]:

# Combine Question, sim and Answer pooled outputs.
join_layer = keras.layers.concatenate([pool_q, sim, pool_a])


# In[60]:

# Using relu here too? Not mentioned in the paper.
hidden_layer = Dense(201, kernel_regularizer=regularizers.l2(1e-4))(join_layer)
hidden_layer = Dropout(0.5)(hidden_layer)

# Final Softmax Layer, add regularizer here too?
softmax = Dense(1, activation='softmax')(hidden_layer)


# In[61]:

model = Model(inputs=[input_q, input_a], outputs=softmax)
print model.summary()

model.compile(optimizer='Adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# model.fit([data_a, data_b], labels, epochs=10)


# In[ ]:



