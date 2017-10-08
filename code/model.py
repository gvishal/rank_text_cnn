from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D


input_q = Input(shape=(ques_max_len,))
embed_q = Embedding(input_dim=)