# imports
from __future__ import print_function
from sys import argv
import pandas as pd
import numpy as np
import os

__author__ = 'Lucas Ramadan'

# unpack arguments
_, input_dir, output_dir = argv

# check if output_dir actually exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# get files
topdir, _, files = next(os.walk(input_dir))
n_files = len(files) - 1 # save this for progress printing

# run through all the files
for i, fi in enumerate(files):
    # calculate fraction of total
    per = round(i*100.0/n_files, 2)

    # print progress
    print('\rprogress: '+str(per)+'%', end='')

    # get window spacing
    w = int(fi[fi.find('(') + 1: fi.find(')')])

    # get the data
    df = pd.read_csv(topdir+fi)
    data = df.values

    # get dimensions
    num_r, num_c = data.shape

    # create the tensor
    data = data.reshape(num_r, num_c/w, w)

    # protein id
    p_id = fi[:fi.index('.')]

    # make filename for raw tensor
    fn = output_dir + p_id + '.npy'

    # finally, save the raw tensor
    np.save(fn, data)

    # calculate mean and std
    mu = data.mean()
    sigma = data.std()

    # scale, using numpy's broadcasting behavior
    data = (data-mu)/sigma

    # save the scaled tensor
    fn = 'scaled_' + output_dir + p_id + '.npy'
    np.save(fn, data)

print('\ncompleted')
