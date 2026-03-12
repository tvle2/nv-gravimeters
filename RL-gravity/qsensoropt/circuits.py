"""
    Module for the simulation of noiseless quantum circuits
    in TF as proposed by J. Kattermole in 
    https://www.kattemolle.com/other/QCinPY.html
"""

from numpy import zeros, array
from tensorflow import reshape, constant
from tensorflow.experimental.numpy import tensordot, moveaxis
from math import sqrt


class Circuit:

    def __init__(self, n, batchsize, prec="complex64"):
        self.n = n
        self.dim = 2**n
        self.bs = batchsize
        self.prec = prec

        self.H_matrix = constant(
            1/sqrt(2)*array([[1.0, 1.0], [1.0, -1.0]]),
            dtype=self.prec,
        )

    def reset(self):
        """Reset the state.

        Prepare initial density operator of shape
        [bs,2**n,2**n], initialized in the |000..0> state.
        """

        rho = zeros(
            [self.bs, self.dim, self.dim],
            dtype=self.prec
        )
        rho[:, 0, 0] = 1
        rho = constant(rho, dtype=self.prec)

        return reshape(
            rho, [self.bs] + [2,]*2*self.n
        )

    def H(self, q_idx, state):
        """Application of the the Hadamart to the qbit identified
        by the index q_idx.

        The index starts from zero.
        """

        state = tensordot(self.H_matrix, state, (1, q_idx))
        return moveaxis(state, 0, q_idx)

    def RotZ(self, theta, q_idx, state):

        RotZ_matrix = tf.stack(
            [[tf.math.exp(-1j*tf.cast(theta, self.prec)/2), self.zero_arr],
             [self.zero_arr, tf.math.exp(1j*tf.cast(theta, self.prec)/2)]],
        )
