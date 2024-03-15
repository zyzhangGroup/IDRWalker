# IDRWalker

A tool for complementing missing parts in proteins based on random walk.

## Installation

Clone the repository, then build and install:

```bash
python -m build --sdist
python -m pip install dist/IDRWalker-0.0.1.tar.gz
```

## Usage

First import the package:

```python
from IDRWalker import *
```

Read the PDB files with missing regions and the FASTA files with full-length sequences, each file should contain information of only one chain.

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
