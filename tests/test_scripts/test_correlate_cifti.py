import numpy as np
from scipy import stats
from mcot.core._scripts.cifti import correlate


def test_correlate_arr():
    for _ in range(10):
        a = np.random.randn(10)
        b = np.random.randn(10)
        assert abs(stats.pearsonr(a, b)[0] -
                   correlate.correlate_arr(a[:, None], b[:, None])[0, 0]) < 1e-8

