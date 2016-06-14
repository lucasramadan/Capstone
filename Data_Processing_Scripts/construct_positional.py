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

# helper function to make positional proteins
def construct_positional(seq, w=5, header=True):
    """
    INPUT: sequence of amino acids (seq), window size (w) and header (bool)
    OUTPUT: 2D numpy array, known as a positional protein

    Given an sequence (array) of amino acids
    construct a windowed representation of each amino acid.
    Defaults: window size (w) of 5, include a header row

    Contains helper function (construct_positional_header) to make header row
    """

    s = w//2
    pad = ['-']*s + list(seq) + ['-']*s
    n = len(seq)
    rows = []

    def construct_positional_header():
        return np.asarray(['AA'+str(i) for i in xrange(-s, s+1)])

    if header:
        rows.append(construct_positional_header())

    for i in xrange(s, n+s):
        row = pad[i-s:i+s+1]
        rows.append(row)

    return np.asarray(rows)

# get files
topdir, _, files = next(os.walk(input_dir))
n_files = len(files) - 1 # save this for progress printing

# run through the files and conver them to positional
for i, fi in enumerate(files):
    # calculate fraction of total
    per = round(i*100.0/n_files, 2)

    # print progress
    print('\rprogress: '+str(per)+'%', end='')

    # get the data
    df = pd.read_csv(topdir+fi)

    # get seq
    seq = df['AA'].values

    # construct the positional df for each spacing
    for spacing in xrange(5, 20, 2):

        # construct positional array
        positional = construct_positional(seq, w=spacing, header=True)

        # create new pos df
        pos_df = pd.DataFrame(positional[1:], columns=positional[0])

        # construct filename
        fn = output_dir+'(' + str(spacing) + ')_' + fi

        # write to file
        pos_df.to_csv(fn, index=False)

print('\ncompleted')
