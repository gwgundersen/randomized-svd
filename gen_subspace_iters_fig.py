"""=============================================================================
============================================================================="""

import matplotlib.pyplot as plt
from   matplotlib import rc
import numpy as np

# ------------------------------------------------------------------------------

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
rc('text', usetex=True)

# ------------------------------------------------------------------------------

def gen_linear_decaying_spectrum_matrix(m, n, k):
    U = np.eye(m)
    # Rapidly decaying singular values with some noise.
    S = [i if i <= k else 0 for i in range(n)]
    Vt = np.eye(n)
    return U @ np.diag(S) @ Vt

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(15, 5)

    k = 30
    A = gen_linear_decaying_spectrum_matrix(50, 50, k)

    qs = [(1, '#11accd'), (2, '#807504'), (3, '#bc2612')]
    errs = []
    min_errs = []

    S_true = np.linalg.svd(A, compute_uv=False)
    S_true = [s / S_true.max() for s in S_true]
    x = range(len(S_true))
    ax.scatter(x, S_true, color='gray')
    ax.plot(x, S_true, label='True singular values', color='gray')

    for q, color in qs:
        A_new = (A @ A.T) ** q @ A
        S = np.linalg.svd(A_new, compute_uv=False)
        S_new = [s / S.max() for s in S]
        x = range(len(S_new))
        ax.scatter(x, S_new, color=color)
        ax.plot(x, S_new, label=r'$q = %s$' % q, color=color)

    ax.set_title('Normalized singular values per $q$ power iterations')

    plt.legend()
    plt.tight_layout()
    plt.show()
