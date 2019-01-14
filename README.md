### Randomized SVD

This repository contains a Python 3.X implementation of randomized SVD as described in

> [_Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions_](https://arxiv.org/abs/0909.4061) by
Nathan Halko, Per-Gunnar Martinsson, Joel A. Tropp (2010).

Furthermore, the repository contains a script for reproducing numerical results from Figure 7.2 in the paper, demonstrating that the implementation produces an actual error that is close to the theoretical minimum implied by Theorem 1.1 in the paper.

### Installation and Usage

To install and reproduce the figures, first clone the repository and then run:

```bash
conda create -n rsvd python=3.7 -f=requirements.txt
```

To generate Figure 7.2, run:

```bash
python gen_fig72.py
```