# Predicting Protein Secondary Structure

<img src='imgs/seq_structure.png'>

### Description
> This is a continuation of my <a href='https://github.com/lucasramadan/Protein_Prediction'>previous semester's work</a>, and the basis for my Capstone (Thesis) project at GalvanizeU

### Goal
> Given a sequence of amino acids, can we predict the structure of the protein? For each amino acid, there is a corresponding "fold-state" which we would like to infer. 

### Data
> Data originally comes from the <a href='http://www.rcsb.org/pdb/home/home.do'>Protein Data Bank</a>, however files are in .dssp format, which is the output of the Pascal program <a href='https://en.wikipedia.org/wiki/DSSP_(hydrogen_bond_estimation_algorithm)'>DSSP</a>. DSSP files were then parsed into CSV and further processed. 

### Models
* Hidden Markov Model (<a href='http://hmmlearn.readthedocs.io/en/latest/'>HMM</a>) 
* Continuous Random Field (<a href='https://cran.r-project.org/web/packages/CRF/index.html'>CRF</a>)
* Convolutional Neural Network (<a href='http://keras.io/layers/convolutional/'>CNN</a>)
* Bidirectional Long-Short Term Memory Recurrent Neural Network (<a href='http://keras.io/layers/recurrent/#lstm'>LSTM</a>)

More to come
