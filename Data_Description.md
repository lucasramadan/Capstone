# Data Description

**dssp.data/**
* original dssp files from PDB

![DSSP Example](imgs/dssp_ex.png)

**raw_dssp_csv/**
* parsed dssp files, in csv format

**clean_dssp_csv/**
* pi-helix substitutions (HHTHH -> IIIII)
* blank SS entries replaced with C

**positional_dssp_csv**
* converted proteins to have context
* filenames begin wth '(#)' for the relative spacing
* used for Probablistic Graphical Model approaches

**verbose_dssp_csv**
* converted proteins to have context
* full feature space utilized 
* for use with Neural Network approaches

**tensor_dssp_npy**
* convert proteins to context
* full feature space utlized
* feature space restructured as "amino images"
* saved as .npy files
