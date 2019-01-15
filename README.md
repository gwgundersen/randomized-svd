## Randomized SVD

This repository contains a Python 3.X implementation of randomized SVD as described in

> [_Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions_](https://arxiv.org/abs/0909.4061) by
Nathan Halko, Per-Gunnar Martinsson, Joel A. Tropp (2010).

Furthermore, the repository contains several scripts:
 
- `gen_fig72.py`: This script reproduces the numerical results from Figure 7.2 in the paper, demonstrating that the implementation produces an actual error that is close to the theoretical minimum implied by Theorem 1.1 in the paper.
- `gen_fig76.py`: This script reproduces the numerical results from Figure 7.6 in the paper, demonstrating that as the number of subspace iterations increases, the low-rank approximation of a given matrix improves.
- `recon_images.py`: This script uses randomized SVD to reconstruct a photograph and demonstrates qualitatively that randomized SVD is working as expected.
- `gen_subspace_fig.py`: This script demonstrates that subspace iteration modifies the relative weights of the singular spectrum of a matrix.

### Installation and Usage

To install and reproduce the figures, first clone the repository and then run:

```bash
$ cd /path/to/repo
$ conda create -n rsvd python=3.7 -f=requirements.txt
$ source activate rsvd
```

Then just run the scripts, e.g.:

```bash
$ python gen_fig72.py
```