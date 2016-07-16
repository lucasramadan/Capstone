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
from keras.layers.advanced_activations import ELU

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
n_hidden = 1000
n_output = y_train.shape[1]


# ----------------------------- BEGIN MODELING ----------------------------- 

# instantiate model
model_3L = Sequential()

# first layer, 1000 nodes, BatchNormalized, ELU and Dropout
model_3L.add(Dense(input_dim = n_input, output_dim = n_hidden))
model_3L.add(BatchNormalization())
model_3L.add(ELU(alpha=0.9))
model_3L.add(Dropout(0.5))

# second layer, 1000 nodes, BatchNormalized, ELU and Dropout
model_3L.add(Dense(input_dim = n_hidden, output_dim = n_hidden))
model_3L.add(BatchNormalization())
model_3L.add(ELU(alpha=0.9))
model_3L.add(Dropout(0.5))

# third layer, 7 nodes, BatchNormalized, SoftMax
model_3L.add(Dense(input_dim = n_hidden, output_dim = n_output))
model_3L.add(BatchNormalization())
model_3L.add(Activation("softmax"))

model_3L.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# ----------------------------- BEGIN TRAINING -----------------------------

h = model_3L.fit(X_train, y_train, validation_data = (X_test, y_test), nb_epoch = epochs)

# save history of training
with open(model_name + '.json', 'w') as f:
	json.dump(h.history, f)
