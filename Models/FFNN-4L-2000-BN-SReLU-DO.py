# ----------------------------- IMPORTS -----------------------------

# general
from sys import argv
import pandas as pd
import numpy as np
import json

# processing
from sklearn.cross_validation import train_test_split

# Keras specific imports
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import SReLU

# ----------------------------- PRE-PROCESSING -----------------------------

# unpack arguments
script, data_file, labels_file, epochs = argv

# get the script name
model_name = script.split('.')[0]
epochs = int(epochs)

# get data in 
data = pd.read_csv(data_file)
labels = pd.read_csv(labels_file)

# find 'C' labels
ind = [i for i, l in enumerate(labels['C'].values) if l == 1]

# remove 'C' labels since they are unknown designations
data.drop(ind, inplace = True)
labels.drop(ind, inplace = True)

# now drop the 'C' class
labels.drop('C', axis = 1, inplace = True)

X_train, X_test, y_train, y_test = train_test_split(data.values, labels.values, random_state=4)

# dimensions for models
n_input = X_train.shape[1]
n_hidden = 2000
n_output = y_train.shape[1]


# ----------------------------- BEGIN MODELING ----------------------------- 

# instantiate model
model_4LW = Sequential()

# first layer, n_input nodes, BatchNormalized, SReLU, Dropout
model_4LW.add(Dense(input_dim = n_input, output_dim = n_hidden))
model_4LW.add(BatchNormalization())
model_4LW.add(SReLU())
model_4LW.add(Dropout(0.5))

# second layer, n_hidden nodes, BatchNormalized, SReLU, Dropout
model_4LW.add(Dense(input_dim = n_hidden, output_dim = n_hidden))
model_4LW.add(BatchNormalization())
model_4LW.add(SReLU())
model_4LW.add(Dropout(0.5))

# third layer, n_hidden nodes, BatchNormalized, SReLU, Dropout
model_4LW.add(Dense(input_dim = n_hidden, output_dim = n_hidden))
model_4LW.add(BatchNormalization())
model_4LW.add(SReLU())
model_4LW.add(Dropout(0.5))

# fourth layer, 6 nodes, BatchNormalized, SoftMax
model_4LW.add(Dense(input_dim = n_hidden, output_dim = n_output))
model_4LW.add(BatchNormalization())
model_4LW.add(Activation("softmax"))

model_4LW.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# ----------------------------- BEGIN TRAINING -----------------------------

h = model_4LW.fit(X_train, y_train, validation_data = (X_test, y_test), nb_epoch = epochs)

# save history of training
with open(model_name + '.json', 'w') as f:
	json.dump(h.history, f)