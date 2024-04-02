# IDRWalker

A tool for complementing missing regions in proteins based on random walk.

## Installation

Clone the repository, then build and install:

```bash
pip install .
```

## Usage

First import the package:

```python
from IDRWalker import *
```

Load the PDB files with missing regions and the FASTA files containing full-length sequences, each file should contain information of only one chain.

```python
chainA = Chain(seq='A.fasta', PDB='A.pdb', chainID='A')
chainB = Chain(seq='B.fasta', PDB='B.pdb', chainID='B')
```

Create a box, and add the chains to the box:

```python
box = Box(box_size=(100,100,100), grid_size=2)
box.add_chain(chainA)
box.add_chain(chainB)
```

Run IDRWalker

```python
box.run()
```

Save the results:

```python
chainA.write('A_out.pdb')
chainB.write('B_out.pdb')
```
