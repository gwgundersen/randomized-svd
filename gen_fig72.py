"""=============================================================================
Reproducing results in Figure 7.2.

Randomized SVD on a matrix with rapidly decaying singular values.
============================================================================="""

from matplotlib import rc
rc('font', **{'family':'sans-serif', 'sans-serif': ['Arial']})
rc('text', usetex=True)

import matplotlib.pyplot as plt
import numpy as np
from   rsvd import rsvd

# ------------------------------------------------------------------------------

def generate_decaying_matrix(m, n):
	U  = np.eye(m)
	# Rapidly decaying singular values with some noise.
	S = [-0.75**i * np.random.randn() for i in range(n)]
	Vt = np.eye(n)
	return U @ np.diag(S) @ Vt

# ------------------------------------------------------------------------------

def generate_full_rank_matrix(m, n):
	return np.random.randn(m, n)

# ------------------------------------------------------------------------------

def actual_error(A, m, l):
	_, _, _, Q = rsvd(A, l, l, return_Q=True)
	return np.linalg.norm((np.eye(m) - Q @ Q.T) @ A, 2)

# ------------------------------------------------------------------------------

def theoretical_error(A, l):
	U, S, Vt = np.linalg.svd(A)
	return S[l+1]

# ------------------------------------------------------------------------------

def plot(A, ax, title, ylabel):
	theoretical_errors = []
	actual_errors      = []
	estimated_errors   = []

	x = range(1, 151, 5)
	for l in x:
		actual_errors.append(np.log10(actual_error(A, m, l)))
		theoretical_errors.append(np.log10(theoretical_error(A, l)))

	ax.scatter(x, theoretical_errors,
		color='#11accd',
		s=20, 
		label=r'$\log_{10}(\sigma_{k+1})$',
		marker='v')
	ax.scatter(x, actual_errors,
		color='#807504',
		s=20,
		label=r'$\log_{10}(e_l)$',
		marker='o')

	ax.plot(x, theoretical_errors,
		color='#11accd',
		linewidth=1)
	ax.plot(x, actual_errors,
		color='#807504',
		linewidth=1)

	if ylabel:
		ax.set_ylabel('Order of magnitude of errors')
	ax.set_xlabel(r'$l$')
	ax.set_title(title)

# ------------------------------------------------------------------------------

fig, ax = plt.subplots(1, 2)

m = 200
n = 200
A = generate_decaying_matrix(m, n)
plot(A, ax[0], 'Exponentially decaying singular values', True)
A = generate_full_rank_matrix(m, n)
plot(A, ax[1], 'Linearly decaying singular values', False)

plt.legend()
plt.tight_layout(	)
plt.show()
