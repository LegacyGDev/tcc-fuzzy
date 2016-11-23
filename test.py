import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.layers import Dense, Activation
from keras.layers import Input, Embedding, merge
from keras.models import Model
from keras.utils.visualize_util import plot

look_back = 1
ativacao = 'relu'

temp_max_input = Input(shape=(look_back,),name='temp_max_input')
temp_min_input = Input(shape=(look_back,),name='temp_min_input')

merged = merge([temp_max_input,temp_min_input],mode='concat')
x = Dense(8,activation=ativacao)(merged)

temp_max_output = Dense(1,activation=ativacao, name='temp_max_output')(x)
temp_min_output = Dense(1,activation=ativacao, name='temp_min_output')(x)

model = Model(input=[temp_max_input,temp_min_input], output=[temp_max_output,temp_min_output])

plot(model, to_file='model.png')
