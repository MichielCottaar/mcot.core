#!/usr/bin/env python
"""Extracts b-tensor for given sequence"""
import numpy as np
from loguru import logger


class QTrajectory(object):
    """
    Describes a Q-space trajectory
    """
    def __init__(self, gradients, times=None):
        """
        Produces a Q-space trajectory out of N parts

        :param gradients: N (M_n, 3) arrays of the gradients sequences
        :param times: (N + (N-1)) timings for the gradient waveforms and the pauses
        """
        self.gradients = gradients
        self.times = times
        assert len(times) == len(gradients) * 2 - 1

    @property
    def n_parts(self, ):
        return len(self.gradients)

    def qvec(self, ):
        """
        N (M_n, 3) arrays with the q-vector
        """
        start = np.zeros(3)
        res = []
        for grad, time in zip(self.gradients, self.times[::2]):
            dt = time / (grad.shape[0] - 1)
            res.append(np.concatenate((start[None, :], start + np.cumsum(dt * (grad[1:] + grad[:-1]) / 2, 0)), 0))
            assert res[-1].shape == grad.shape
            start = res[-1][-1]
        return res

    def btensor(self, ):
        """
        N (M_n, 3, 3) arrays with the b-tensor trajectory
        """
        start = np.zeros((3, 3))
        res = []
        for idx, qvec in enumerate(self.qvec()):
            dt = self.times[idx * 2] / (qvec.shape[0] - 1)
            #qmean = (qvec[1:] + qvec[:-1]) / 2.
            qmean = qvec
            qmean_tensor = qmean[:, None, :] * qmean[:, :, None]
            res.append(np.concatenate((start[None, :, :], start + np.cumsum(qmean_tensor * dt, 0)), 0))
            if idx != self.n_parts - 1:
                start = res[-1][-1] + qvec[-1, None, :] * qvec[-1, :, None] * self.times[idx * 2 + 1]
        return res

    def plot(self, target, axes=None):
        """
        Plots the target in RGB

        :param target: one of ('gradients', 'qvec', 'bdiag', 'boffdiag')
        """
        if axes is None:
            import matplotlib.pyplot as plt
            axes = plt.gca()
        if target == 'gradients':
            yval = self.gradients
        elif target == 'qvec':
            yval = self.qvec()
        elif target == 'bdiag':
            yval = [val[..., [0, 1, 2], [0, 1, 2]] for val in self.btensor()]
        elif target == 'boffdiag':
            yval = [val[..., [1, 2, 2], [0, 0, 1]] for val in self.btensor()]
        else:
            raise ValueError(f"{target} not recognized")
        time_sum = np.append(0, np.cumsum(self.times))
        xval = [np.linspace(start, start + step, val.shape[0]) for start, step, val in zip(time_sum[::2], self.times[::2], yval)]
        for idx, color in enumerate('ymc' if target == 'boffdiag' else 'rgb'):
            for idx2, xv, yv in zip(range(self.n_parts), xval, yval):
                axes.plot(xv, yv[:, idx], color)
                if idx2 != 0:
                    axes.plot((prev[0][-1], xv[0]), (prev[1][-1, idx], yv[0, idx]), color)
                prev = xv, yv
        axes.axis('off')


def run(sequence, duration):
    """
    Calculates the b-tensor of a sequence

    :param sequence: (N, 3) array of gradient orientations in mT/m
    :param duration: total duration of the sequence in ms
    :return: tuple with:

        - (3, ) array of final q-tensor in rad/m
        - (3, 3) array of final B-tensor in s/mm^2
    """
    if sequence.ndim != 2 or sequence.shape[1] != 3:
        raise ValueError("Expected an (N, 3) array as a sequence")
    logger.debug('sequence shape: %s', sequence.shape)
    gradient_slope = 0.5 * (sequence[1:] - sequence[:-1])
    gradient_intercept = sequence[:-1]
    qvec = np.cumsum(gradient_intercept + gradient_slope, 0)
    qfinal = qvec[-1]
    constant_term = np.concatenate((np.zeros((1, 3)), qvec[:-1]), 0)

    btensor = np.zeros((3, 3))
    for t1, value1 in enumerate((constant_term, gradient_intercept, gradient_slope)):
        for t2, value2 in enumerate((constant_term, gradient_intercept, gradient_slope)):
            btensor += np.sum(value2[:, :, None] * value1[:, None, :] / (t1 + t2 + 1), 0)
    logger.debug('b-tensor before unit conversion: %s', sequence.shape)

    dt = duration / (sequence.shape[0] - 1)
    gamma = 268
    return qfinal * dt * gamma, btensor * 1e-9 * dt ** 3 * gamma ** 2


def run_from_args(args):
    qfinal, btensor = run(
            sequence=np.genfromtxt(args.sequence, skip_header=1) * args.Gmax,
            duration=args.dt,
    )
    print('Final q-value in rad/m:')
    print(qfinal)
    print('Final B-tensor in s/mm^2:')
    print(btensor)
    print('b-value =', np.trace(btensor))


def add_to_parser(parser):
    parser.add_argument("sequence", help='General waveform')
    parser.add_argument("dt", type=float, help='total duration of the sequence')
    parser.add_argument('-G', "--Gmax", default=80, type=float, help='Maximum gradient in mT/m (default: 80)')

