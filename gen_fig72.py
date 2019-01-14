"""=============================================================================
Reproducing numerical results in Halko et al's Figure 7.2. The main idea is to
demonstrate that on a matrix with rapidly decaying singular values, randomized
SVD captures low-rank structure with a reconstruction loss that is close to the
theoretical minimum.
============================================================================="""

import matplotlib.pyplot as plt
from   matplotlib import rc
import numpy as np
from   rsvd import rsvd

rc('font', **{
    'family': 'sans-serif',
    'sans-serif': 'Arial',
    'size' : 16
})
rc('text', usetex=True)

# ------------------------------------------------------------------------------

def generate_decaying_matrix(m, n):
    U = np.eye(m)
    # Rapidly decaying singular values with some noise.
    S = [-0.75 ** i * np.random.randn() for i in range(n)]
    Vt = np.eye(n)
    return U @ np.diag(S) @ Vt

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(15, 5)
    m = 200
    n = 200
    A = generate_decaying_matrix(m, n)

    mins = []
    errs = []

    ls = range(1, 151, 5)
    for l in ls:
        _, _, _, Q = rsvd(A, l, n_oversamples=0, return_range=True)
        err = np.linalg.norm((np.eye(m) - Q @ Q.T) @ A, 2)
        errs.append(np.log10(err))

        S = np.linalg.svd(A, compute_uv=False)
        min_ = S[l + 1]
        mins.append(np.log10(min_))

    ax.scatter(ls, mins, color='#11accd', s=20,
               label=r'$\log_{10}(\sigma_{k+1})$', marker='v')
    ax.scatter(ls, errs, color='#807504', s=20, label=r'$\log_{10}(e_l)$',
               marker='o')
    ax.plot(ls, mins, color='#11accd', linewidth=1)
    ax.plot(ls, errs, color='#807504', linewidth=1)

    ax.set_ylabel('Order of magnitude of errors')
    ax.set_xlabel(r'$l$')
    ax.set_title('Exponentially decaying singular values')

    plt.legend()
    plt.tight_layout()
    plt.show()
