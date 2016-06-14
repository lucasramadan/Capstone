# imports
from __future__ import print_function
from sys import argv
import pandas as pd
import numpy as np
import os
import warnings

# skip the warning from copy setting in pandas
warnings.filterwarnings('ignore')

__author__ = 'Lucas Ramadan'

# unpack arguments
_, input_dir, output_dir = argv

# check if output_dir actually exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# helper function to fix HHTHH structure pattern
def replace_pi(l):
    """
    This function will go through all entries in SS column
    And replace occurances of HHTHH with IIIII
    """
    i = 0
    while i < len(l)-4:
        window = ''.join(l[i:i+5])
        if window == 'HHTHH':
            l[i:i+5] = 'I'
        i += 1
    return l

# get files
topdir, _, files = next(os.walk(input_dir))
n_files = len(files) - 1 # save this for progress printing

# go through and clean
for i, fi in enumerate(files):
    # calc percent
    perc_complete = round(i*100.0/n_files, 2)
    # show progress
    print('\rProgress: ' + str(perc_complete) + '%', end='')

    # read in the csv
    df = pd.read_csv(topdir+fi)

    # replace NaN with ?
    df.fillna(value='?', inplace=True)

    # drop !* rows
    df = df[df['AA'] != '!']

    # replace "HHTHH" pi helix pattern
    df['SS'] = replace_pi(df['SS'])

    # replace '?' SS with 'C'
    df['SS'] = df['SS'].apply(lambda aa: 'C' if aa == '?' else aa)

    # write to csv
    df.to_csv(output_dir+fi, index=False)
