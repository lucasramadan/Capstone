# imports
from __future__ import print_function
from sys import argv
import pandas as pd
import numpy as np
import os

# unpack command line arguments
_, input_dir, output_dir = argv

# check if output_dir actually exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# unpack files
topdir, _, files = next(os.walk(input_dir))

# hard code spacings in order
spacing = [11, 13, 15, 17, 19, 5, 7, 9]

# calculate number of files, for printing and relevant files
n = len(files)

# outer for each spacing
for i, s in enumerate(spacing):

    rel_files = files[(n*i):(n*(i+1))]

    # for matrices
    combo = pd.read_csv(topdir+rel_files[0])
    header = combo.columns
    combo = combo.values

#     for tensors
#     combo = np.load(topdir+rel_files[0])

    # inner for each file after
    for ii, rf in enumerate(rel_files[1:]):

        prog = (ii+1) + (i*347)
        print('\rprogress: ' + str(int(prog*100.0/(n-1))) + '%', end='')

        # for matrices
        df = pd.read_csv(topdir+rf)
        combo = np.concatenate((combo, df.values), axis=0)

#         # for tensors
#         d = np.load(topdir+rf)
#         combo = np.append(combo, d, axis=0)

    # for matrices, write the combo
    fn = '../combined_verbose_dssp_csv/(' + str(s) + ')_combined_verbose.csv'
    combo = pd.DataFrame(combo, columns=header)
    combo.to_csv(fn, index=False)

#     # for tensors write the combo
#     fn = '../combined_scaled_tensor_dssp_npy/(' + str(s) + ')_combined_scaled.npy'
#     np.save(fn, combo)
