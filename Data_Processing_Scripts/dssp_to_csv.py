# imports
from __future__ import print_function # for progress printing
import pandas as pd
import numpy as np
import os
from sys import argv

__author__ = 'Lucas Ramadan'

# unpacking of sys arguments
_, dssp_dir, output_dir = argv

# check if output_dir actually exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# helper functions
def row_splitter(l):
    """
    This function parses each row of the tabular section of the DSSP file
    and returns a row (list) that is comma seperated for each field
    """
    ind = [(0, 5),     # known as 'DSSP RESIDUE #'
           (5, 10),    # known as 'PDB RESIDUE #'
           (10, 12),   # known as 'CHAIN ID'
           (12, 14),   # known as 'AA'
           (14, 17),   # known as 'SECONDARY STRUCTURE'
           (17, 19),   # known as '3-HELIX'
           (19, 20),   # known as '4-HELIX'
           (20, 21),   # known as '5-HELIX'
           (21, 22),   # known as 'BEND'
           (22, 23),   # known as 'CHIRALITY'
           (23, 24),   # known as 'BETA BRIDGE 1'
           (24, 25),   # known as 'BETA BRIDGE 2'
           (25, 29),   # known as 'BP1'
           (29, 33),   # known as 'BP2'
           (33, 34),   # known as 'BSL'
           (34, 38),   # known as 'ACC'
           (38, 45),   # known as N-H-->O BF1 I
           (46, 50),   # known as N-H-->O BF1 E
           (50, 56),   # known as O-->H-N BF1 I
           (57, 61),   # known as O-->H-N BF1 E
           (61, 67),   # known as N-H-->O BF2 I
           (68, 72),   # known as N-H-->O BF2 E
           (72, 78),   # known as O-->H-N BF2 I
           (79, 83),   # known as O-->H-N BF2 E
           (83, 91),   # known as TCO
           (91, 97),   # known as KAPPA
           (97, 103),  # known as ALPHA
           (103, 109), # known as PHI
           (109, 115), # known as PSI
           (115, 122), # known as X-CA
           (122, 129), # known as Y-CA
           (129, 136)] # known as Z-CA

    row = []

    for s, e in ind:
        entry = l[s:e]
        entry = entry.replace(' ', '')
        row.append(entry)

    return row

def make_header_row():
    """
    This function constructs the corresponding header row for a parsed dssp file
    """
    row = []
    row.append('DSSP') # known as 'DSSP RESIDUE #'
    row.append('PDB') # known as 'PDB RESIDUE #'
    row.append('CHAIN') # known as 'CHAIN ID'
    row.append('AA') # known as 'AA'
    row.append('SS') # known as 'SECONDARY STRUCTURE'
    row.append('3H') # known as '3-HELIX'
    row.append('4H') # known as '4-HELIX'
    row.append('5H') # known as '5-HELIX'
    row.append('BEND') # known as 'BEND'
    row.append('CHIR') # known as 'CHIRALITY'
    row.append('BB1') # known as 'BETA BRIDGE 1'
    row.append('BB2') # known as 'BETA BRIDGE 2'
    row.append('BP1') # known as 'BP1'
    row.append('BP2') # known as 'BP2'
    row.append('BSL') # known as 'BETA SHEET LABEL'
    row.append('ACC') # known as 'ACC'
    row.append('NO1I') # known as N-H-->O BF1 I
    row.append('NO1E') # known as N-H-->O BF1 E
    row.append('ON1I') # known as O-->H-N BF1 I
    row.append('ON1E') # known as O-->H-N BF1 E
    row.append('NO2I') # known as N-H-->O BF2 I
    row.append('NO2E') # known as N-H-->O BF2 E
    row.append('ON2I') # known as O-->H-N BF2 I
    row.append('ON2E') # known as O-->H-N BF2 E
    row.append('TCO') # known as TCO
    row.append('KAPPA') # known as KAPPA
    row.append('ALPHA') # known as ALPHA
    row.append('PHI') # known as PHI
    row.append('PSI') # known as PSI
    row.append('X-CA') # known as X-CA
    row.append('Y-CA') # known as Y-CA
    row.append('Z-CA') # known as Z-CA
    return row

# get the list of files that we are going to convert
topdir, _, files = next(os.walk(dssp_dir))
n_files = len(files) - 1 # save this for progress printing

# construct header row outside of the iterations
header = make_header_row()

# go through each file
for i, fi in enumerate(files):
    # show progress
    perc_complete = round(i*100.0/n_files, 2)
    print('\rProgress: ' + str(perc_complete) + '%', end='')
    
    # open the .dssp file
    with open(topdir+fi) as f:
        data = f.read()
    
    # find the sequence section
    protein = data[data.find('#  RESIDUE'):].split('\n')
    
    # go through and get each column of the row of data
    # skip the first row (old header) and last row (artifact '\n')
    clean_protein = [row_splitter(r) for r in protein[1:-1]]
    
    # construct the dataframe
    protein_df = pd.DataFrame(clean_protein, columns=header)
    
    # construct filename
    fn = output_dir + fi[:fi.find('.')] + '.csv'
    
    # write to csv
    protein_df.to_csv(fn, index=False)
