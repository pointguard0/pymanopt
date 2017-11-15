from conjugate_gradient import ConjugateGradient

import theano.tensor as T
import numpy as np

from pymanopt import Problem 
from pymanopt.manifolds import Stiefel

import time

import matplotlib.pyplot as plt

if __name__ == '__main__':
    n = 100
    A = np.random.randn(n, n)
    A = 0.5 * (A + A.T)

    p = 5

    true_val = sum(sorted(np.linalg.eig(A)[0])[:p])

    St = Stiefel(n, p)
    X = T.matrix()
    cost = T.dot(X.T, T.dot(A, X)).trace()

    problem = Problem(manifold=St, cost=cost, arg=X)

    solver = ConjugateGradient(maxiter=1000, mingradnorm=1e-6)

    Xopt = solver.solve(problem)

    print solver.iters, time.time() - solver._time

    plt.plot(solver._cost, '-v')
    plt.plot(solver._gradnorm, '-*')
    plt.hlines(y=true_val, xmin=0, xmax=solver.iters + 1)
    plt.show()

