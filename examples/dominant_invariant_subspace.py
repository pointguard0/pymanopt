"""
This file is based on dominant_invariant_subspace.m from the manopt MATLAB
package.

The optimization is performed on the Grassmann manifold, since only the
space spanned by the columns of X matters. The implementation is short to
show how Manopt can be used to quickly obtain a prototype. To make the
implementation more efficient, one might first try to use the caching
system, that is, use the optional 'store' arguments in the cost, grad and
hess functions. Furthermore, using egrad2rgrad and ehess2rhess is quick
and easy, but not always efficient. Having a look at the formulas
implemented in these functions can help rewrite the code without them,
possibly more efficiently.

See also: dominant_invariant_subspace_complex

Main author: Nicolas Boumal, July 5, 2013

Ported to pymanopt by Jamie Townsend, Nov 24, 2015
"""
import theano.tensor as T
import numpy as np

from pymanopt import Problem
from pymanopt.solvers import ConjugateGradient
from pymanopt.manifolds import Stiefel


def dominant_invariant_subspace(A, p):
    """
    Returns an orthonormal basis of the dominant invariant p-subspace of A.

    Arguments:
        - A
            A real, symmetric matrix A of size nxn
        - p
            integer p < n.
    Returns:
        A real, orthonormal matrix X of size nxp such that trace(X'*A*X)
        is maximized. That is, the columns of X form an orthonormal basis
        of a dominant subspace of dimension p of A. These span the same space
        as  the eigenvectors associated with the largest eigenvalues of A.
        Sign is important: 2 is deemed a larger eigenvalue than -5.
    """
    # Make sure the input matrix is square and symmetric
    n = A.shape[0]
    assert type(A) == np.ndarray, 'A must be a numpy array.'
    assert np.isreal(A).all(), 'A must be real.'
    assert A.shape[1] == n, 'A must be square.'
    assert np.linalg.norm(A-A.T) < n * np.spacing(1), 'A must be symmetric.'
    assert p <= n, 'p must be smaller than n.'

    # Define the cost on the Stiefel manifold
    St = Stiefel(n, p)
    X = T.matrix()
    cost = -T.dot(X.T, T.dot(A, X)).trace()

    # Setup the problem
    problem = Problem(manifold=St, cost=cost, arg=X)

    # Create a solver object
    solver = ConjugateGradient(maxiter=1000, mingradnorm=1e-6)

    # Solve
    Xopt = solver.solve(problem)

    return Xopt


if __name__ == '__main__':
    """
    This demo script will generate a random 128 x 128 symmetric matrix and find
    the dominant invariant 3 dimensional subspace for this matrix, that is, it
    will find the subspace spanned by the three eigenvectors with the largest
    eigenvalues.
    """
    # Generate some random data to test the function
    print('Generating random matrix...')
    n = 100
    A = np.random.randn(n, n)
    A = 0.5 * (A + A.T)

    p = 2

    St = Stiefel(n, p)
    X = T.matrix()
    cost = -T.dot(X.T, T.dot(A, X)).trace()

    # Setup the problem
    problem = Problem(manifold=St, cost=cost, arg=X)

    # Create a solver object
    solver = ConjugateGradient(maxiter=1000, mingradnorm=1e-6)

    # Solve
    Xopt = solver.solve(problem)

    print solver.iters


    # Test function...
    # dominant_invariant_subspace(A, p)

    # print np.logspace(1, 5, 5)

    # for n in np.logspace(10, 10000000, 6):
    #     A = np.random.randn(n, n)
    #     A = 0.5 * (A + A.T)
    #     p = n // 10
    #     dominant_invariant_subspace(A, p)
