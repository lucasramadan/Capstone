# imports
from __future__ import print_function
from multiprocessing import Pool
from sys import argv
import pandas as pd
import numpy as np
import os
import string

__author__ = 'Lucas Ramadan'

# unpack arguments
_, input_dir, output_dir, n_processes = argv
n_processes = int(n_processes)

# check if output_dir actually exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# helper functions
def construct_verbose(row):
    """
    INPUT: Positional row (ex: ['-', '-', 'A', 'E', 'K'] )
    OUTPUT: Verbose row (ex: ['--2', '--1', 'A0', 'E1', 'K2'] )
    """

    m = len(row) // 2
    pos = xrange(-m, m+1)

    verb = [v + str(p) for v, p in zip(row, pos)]

    return verb

def construct_features(n=5):
    """
    Constructs the feature space for a verbose protein, given the spacing (n)
    """
    m = n // 2
    pos = xrange(-m, m+1)

    # hard coded possible amino acids from data
    aminos = ['-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
              'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y']

    # can also have cystein bridges, represented by lowercase letters
    aminos = aminos + list(string.ascii_lowercase)

    # construct full feature space
    fs = np.asarray([[aa+str(p) for p in pos] for aa in aminos]).ravel()

    return fs

def construct_full(verb, fs):
    """
    This function converts a verbose row to a fully binarized row,
    according to the feature space that was provided (fs)

    INPUT: A verbose row (verb) and the feature space (fs)
    OUTPUT: A fully binarized row
    """
    # infer window size from the verbose row
    n = len(verb)
    m = n // 2
    pos = xrange(-m, m+1)

    # start with empty array
    full = np.zeros(len(fs), dtype=int)

    # get the indices where each entry in the verbose row matches the full row
    ind = [np.where(v == fs)[0][0] for v in verb]

    # set the values of the matching entries to 1
    full[ind] = 1

    return full

def construct_verbose_df(data, n=5):
    """
    This function will convert a positional df to a verbose df

    INPUT: A positionally-formatted pandas DataFrame (data) and window size (n)
    OUTPUT: A fully verbose pandas DataFrame (full_df)
    """

    fs = construct_features(n=n)

    full_ar = []

    for row in data.values:
        vb = construct_verbose(row)
        full = construct_full(vb, fs)
        full_ar.append(full)

    full_df = pd.DataFrame(full_ar, columns=fs)

    return full_df

def write_verbose(fi):
    w = int(fi[fi.find('(') + 1: fi.find(')')])
    data = pd.read_csv(topdir+fi)
    full_df = construct_verbose_df(data, n=w)
    fn = output_dir + fi
    full_df.to_csv(fn, index=False)
    # snag number of completed files
    n_comp = len(next(os.walk(output_dir))[2])
    perc = round(n_comp*100.0/n_total, 2)
    print('\rprogress: '+str(perc), end='')

# get files
topdir, _, files = next(os.walk(input_dir))
n_total = len(files)

# multiprocessing
pool = Pool(n_processes)
pool.map(write_verbose, files)
pool.close()
pool.join()
