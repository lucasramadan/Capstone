# ----------------------------- IMPORTS -----------------------------

# general 
from sys import argv
import numpy as np
import json

# keras specific
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, Masking, Activation 
from keras.layers import SimpleRNN, GRU, LSTM, Dropout
from keras.layers.advanced_activations import ELU, SReLU
from keras.layers.normalization import BatchNormalization

# ----------------------------- PRE-PROCESSING -----------------------------

script, data_file, bs = argv
model_name = script.split('.')[0]
bs = int(bs)

# data reading 
with open(data_file) as f:
	data = np.load(f)
	labels = data['y']
	data = data['x']

n_input = tuple(data.shape[1:])
n_hidden = 1000
n_output = labels.shape[2]

# ----------------------------- BEGIN MODELING -----------------------------

# instantiate model
model = Sequential()

# first (output) layer
model.add(Masking(mask_value = 0, input_shape = n_input))
model.add(SimpleRNN(n_output, return_sequences = True))
model.add(Activation('softmax'))

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# ----------------------------- BEGIN TRAINING -----------------------------

h = model.fit(data, labels, batch_size = bs, validation_split = 0.2)

# save history of training
with open(model_name + '.json', 'w') as f:
	json.dump(h.history, f)
