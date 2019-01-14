"""=============================================================================
Reproducing numerical results in Halko et al's Figure 7.6.
============================================================================="""

import matplotlib.pyplot as plt
from   matplotlib import rc
import numpy as np
from   scipy.misc import face
from   rsvd import rsvd

rc('font', **{
    'family': 'sans-serif',
    'sans-serif': 'Arial',
    'size' : 16
})
rc('text', usetex=True)

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(15, 5)

    A = face(gray=True)
    m, _ = A.shape

    mins = []
    q2list = {
        0: ([], '#11accd', 'o'),
        1: ([], '#807504', 'v'),
        2: ([], '#bc2612', 'd'),
        3: ([], '#236040', '*')
    }

    ls = [20, 40, 60, 80, 100]
    qs = [0, 1, 2, 3]
    for l in ls:

        # Compute theoretical minimum error for each l.
        S = np.linalg.svd(A, compute_uv=False)
        min_ = S[l + 1]
        mins.append(min_)

        # Compute error for each number of subspace iterations.
        for q in qs:
            _, _, _, Q = rsvd(A, l, n_oversamples=0, n_subspace_iters=q,
                              return_range=True)
            err = np.linalg.norm((np.eye(m) - Q @ Q.T) @ A, 2)
            q2list[q][0].append(err)

    ax.scatter(ls, mins, color='gray', s=30, label='Minimum error', marker='s')
    ax.plot(ls, mins, color='gray', linewidth=1)

    for q in qs:
        data, color, marker = q2list[q]
        ax.scatter(ls, data, s=30, label=r'$q = %s$' % q, marker=marker,
                   color=color)
        ax.plot(ls, data, linewidth=1, color=color)

    ax.set_ylabel('Magnitude')
    ax.set_xlabel(r'Random samples $\ell$')
    ax.set_title(r'Approximation error $e_{\ell}$')
    plt.legend()
    plt.tight_layout()
    plt.show()
