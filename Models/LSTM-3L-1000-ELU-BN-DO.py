# ----------------------------- IMPORTS -----------------------------

# general 
from sys import argv
import numpy as np
import json

# pre-processing
from sklearn.cross_validation import train_test_split

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

X_train, X_test, Y_train, Y_test = train_test_split(data, labels, random_state=4)

n_input = tuple(data.shape[1:])
n_hidden = 1000
n_output = labels.shape[2]

# ----------------------------- BEGIN MODELING -----------------------------

# instantiate model
model = Sequential()

# first (input) layer
model.add(TimeDistributed(Dense(n_hidden), input_shape=n_input))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(ELU())

# second (hidden) layer
model.add(LSTM(n_hidden, return_sequences=True, consume_less='gpu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(ELU())

# third (output) layer
model.add(TimeDistributed(Dense(n_output)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Activation('softmax'))

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# ----------------------------- BEGIN TRAINING -----------------------------

# NOTE: Average epoch training time ~ 7 hours

h = model.fit(data, labels, batch_size = bs, validation_split = 0.2)

# save history of training
with open(model_name + '.json', 'w') as f:
	json.dump(h.history, f)
