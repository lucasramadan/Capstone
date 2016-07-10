#!/usr/bin/python
from __future__ import print_function
from sys import argv
import pandas as pd
import numpy as np
import os

# unpack command line arguments
# data_dir must contain verbose files, labels_dir must contain one_hot files
script, data_dir, labels_dir, mask_dir, spacing = argv

# get directories 
data_top, data_inner, data_files = os.walk(data_dir).next()
labels_top, labels_inner, labels_files = os.walk(labels_dir).next()

# check to see if mask_dir already exists
if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)

# helper function for filenames
def get_fn(fn):
    s = fn.index('_') + 1
    return fn[s:]

# subset the data_files corresponding to the provided spacing
match_str = '({0})_'.format(str(spacing))
data_files = [fi for fi in data_files if fi.startswith(match_str)]

# hard check to make sure we have the right files incoming
assert len(data_files) == len(labels_files)
assert all([get_fn(df) == lf for df, lf in zip(data_files, labels_files)])

# first loop over and determine the maximum length protein in the dataset
n_files = len(labels_files) - 1
max_len = 0

for i, lf in enumerate(labels_files):
	prog = str(round(i*100.0/n_files, 2))
	print('\rprogress: '+prog+'%', end='')

	label_data = pd.read_csv(labels_top + lf)
	n_obs, n_labels = label_data.shape

	if n_obs > max_len:
		max_len = n_obs

print()

# get the feature space size 
# TODO: make this smarter for scaling up to full dataset
data_fs = pd.read_csv(data_top+data_files[0]).shape[1]

# now go through and mask all of the data and label files based on max_len
# first construct zero rows to work with 
zero_data = [0] * data_fs
zero_label = [0] * n_labels

complete_dataset = []
complete_labels = []

for i, (df, lf) in enumerate(zip(data_files, labels_files)):
    prog = str(round(i*100.0/n_files, 2))
    print('\rprogress: '+prog+'%', end='')

    # another safeguard for our files
    assert get_fn(df) == lf

    data = pd.read_csv(data_top + df).values
    label = pd.read_csv(labels_top + lf).values

    # conditions to catch anomolies or already max_len file
    if (data.shape[0] == max_len):
        filename = mask_dir + df[:df.index('.csv')] + '.npz'
        np.savez(filename, x=data, y=label)
        continue
    
    # calculate difference
    diff = max_len - data.shape[0]

    # construct required zero rows for masking
    data_req = np.asarray([zero_data for _ in xrange(diff)])
    label_req = np.asarray([zero_label for _ in xrange(diff)])

    # construct masked data and label files
    d = np.append(data_req, data, axis=0)
    l = np.append(label_req, label, axis=0)

    # continue constructing the full dataset
    complete_dataset.append(d)
    complete_labels.append(l)

    # construct the filename for the data
    filename = mask_dir + df[:df.index('.csv')] + '.npz'

    # save file
    np.savez(filename, x=d, y=l)

print()

# finally save the complete
complete_dataset = np.asarray(complete_dataset)
complete_labels = np.asarray(complete_labels)

print(complete_dataset.shape, complete_labels.shape)

np.savez('complete_masked_verbose.npz', x=complete_dataset, y=complete_labels)
