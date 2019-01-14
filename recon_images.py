"""=============================================================================
Qualitative experiment showing better image reconstructions given more sampling.
============================================================================="""

from   imageio import imwrite
import numpy as np
from   scipy.misc import face
from   rsvd import rsvd

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    A = face(gray=True)
    imwrite('figures/raccoon.jpg', A.astype(np.uint8))
    rank = 10

    for n_oversamples in [1, 10, 100]:
        U, S, Vt = rsvd(A, rank, n_oversamples, n_subspace_iters=0)
        M_recon = (U * S) @ Vt
        fname = 'figures/raccoon_k%s_p%s.jpg' % (rank, n_oversamples)
        imwrite(fname, M_recon.astype(np.uint8))

    U, S, Vt = np.linalg.svd(A)
    U, S, Vt = U[:, :rank], S[:rank], Vt[:rank, :]
    M_recon = (U * S) @ Vt
    fname = 'figures/raccoon_actual_k%s.jpg' % rank
    imwrite(fname, M_recon.astype(np.uint8))
