## Motivation
### Excerpt from [Wikipedia](https://en.wikipedia.org/wiki/Protein_secondary_structure)

> 
Predicting protein tertiary structure from only its amino acid sequence is a very challenging problem (see protein structure prediction), but using the simpler secondary structure definitions is more tractable.
> 
Early methods of secondary-structure prediction were restricted to predicting the three predominate states: helix, sheet, or random coil. These methods were based on the helix- or sheet-forming propensities of individual amino acids, sometimes coupled with rules for estimating the free energy of forming secondary structure elements. Such methods were typically ~60% accurate in predicting which of the three states (helix/sheet/coil) a residue adopts. The first widely used technique to predict protein secondary structure from the amino acid sequence was the Chou–Fasman method.
> 
A significant increase in accuracy (to nearly ~80%) was made by exploiting multiple sequence alignment; knowing the full distribution of amino acids that occur at a position (and in its vicinity, typically ~7 residues on either side) throughout evolution provides a much better picture of the structural tendencies near that position. For illustration, a given protein might have a glycine at a given position, which by itself might suggest a random coil there. However, multiple sequence alignment might reveal that helix-favoring amino acids occur at that position (and nearby positions) in 95% of homologous proteins spanning nearly a billion years of evolution. Moreover, by examining the average hydrophobicity at that and nearby positions, the same alignment might also suggest a pattern of residue solvent accessibility consistent with an α-helix. Taken together, these factors would suggest that the glycine of the original protein adopts α-helical structure, rather than random coil. Several types of methods are used to combine all the available data to form a 3-state prediction, including neural networks, hidden Markov models and support vector machines. Modern prediction methods also provide a confidence score for their predictions at every position.
> 
Secondary-structure prediction methods are continuously benchmarked, e.g., EVA (benchmark). Based on these tests, the most accurate methods at present are Psipred, SAM, PORTER, PROF, and SABLE. The chief area for improvement appears to be the prediction of β-strands; residues confidently predicted as β-strand are likely to be so, but the methods are apt to overlook some β-strand segments (false negatives). There is likely an upper limit of ~90% prediction accuracy overall, due to the idiosyncrasies of the standard method (DSSP) for assigning secondary-structure classes (helix/strand/coil) to PDB structures, against which the predictions are benchmarked. 
Accurate secondary-structure prediction is a key element in the prediction of tertiary structure, in all but the simplest (homology modeling) cases. For example, a confidently predicted pattern of six secondary structure elements βαββαβ is the signature of a ferredoxin fold.

## Resources
> Overall: http://swift.cmbi.ru.nl/gv/dssp/ <br>
> Output: http://swift.cmbi.ru.nl/gv/dssp/DSSP_2.html <br>
> More: http://www.cmbi.ru.nl/dssp.html <br>
> Further: http://swift.cmbi.ru.nl/gv/dssp/HTML/descrip.html <br>
> PDB: http://www.wwpdb.org/documentation/file-format-content/format33/sect5.html

## Additional Info
* Class grouping: helix (G, H and I), strand (E and B) and loop (S, T, and C, where C sometimes is represented also as blank space).
> 
| HELIX   | STRAND | LOOP    |
|---------|--------|---------|
| G, H, I | E, B   | S, T, C |

* Label I ([pi helix](https://en.wikipedia.org/wiki/Pi_helix)) is very rare, due to an artifact of the original 1983 implementation of DSSP
  * A quick solution to this is to replace structural sequences of 'HHTHH' with 'IIIII' as described  [here](http://swift.cmbi.ru.nl/gv/dssp/DSSP_2.html)
