import math

import cupy
import numpy as np
import unittest

from radical.phasesol import makeJHWJ, makeJHWr
from radical.transforms import ndash

class PhaseSol(unittest.TestCase):
    def test_normalmatrices(self):
        NSRC = 5
        NPARAMS = 2
        MPARAMS = NPARAMS * NSRC
        NANTS = 48
        NROWS = (NANTS * (NANTS - 1)) // 2
        NCHANS = 30

        # Create a data set 1128 rows x 30 channels x 2 x 2
        uvws = np.random.rand(NROWS, 3) - 0.5
        data = (np.random.rand(NROWS, NCHANS, 2, 2) - 0.5) + 1j * (np.random.rand(NROWS, NCHANS, 2, 2) - 0.5)
        weights = np.random.rand(NROWS, NCHANS, 2, 2)

        wavelengths = np.linspace(1, 2, NCHANS)

        # 48 antennas gives 1128 correlations
        ants1, ants2 = [], []
        for ant1 in range(NANTS):
            for ant2 in range(ant1 + 1, NANTS):
                ants1.append(ant1)
                ants2.append(ant2)

        ants1, ants2 = np.array(ants1, dtype=int), np.array(ants2, dtype=int)
        assert(len(ants1) == len(ants2) == NROWS)

        # Set antenna locations and phase offsets randomly
        ddphases = np.random.rand(NANTS, NSRC) + 1j * np.random.rand(NANTS, NSRC)
        X = np.random.rand(NANTS, NPARAMS)

        # 5 sources
        lmndash = 0.1 * (np.random.rand(NSRC, 3) - 0.5)
        lmndash[:, 2] = ndash(lmndash[:, 0], lmndash[:, 1])
        fluxes = (np.random.rand(NSRC, NCHANS, 2, 2) - 0.5) + 1j * (np.random.rand(NSRC, NCHANS, 2, 2) - 0.5)

        J = np.zeros((NROWS * NCHANS * 4, MPARAMS), dtype=np.complex128)

        model = np.zeros((NROWS, NCHANS, 2, 2), dtype=np.complex128)
        for isrc in range(NSRC):
            srcmodel = fluxes[isrc, None, :, :, :] * np.exp(
                2j * np.pi * (
                    uvws[:, 0:1] * lmndash[isrc, 0] +
                    uvws[:, 1:2] * lmndash[isrc, 1] +
                    uvws[:, 2:3] * lmndash[isrc, 2]
                ) /  wavelengths[None, :] + 1j * (ddphases[ants1, isrc] - ddphases[ants2, isrc])[:, None]
            )[:, :, None, None]

            model += srcmodel

            for iparam in range(NPARAMS):
                J[:, isrc * NPARAMS + iparam] = (
                    (1j * (X[ants1, iparam] - X[ants2, iparam]))[:, None, None, None] * srcmodel
                ).reshape(-1)

        expected = np.conj(J.T) @ (weights.reshape(-1, 1) * J)

        JHWJ_d = cupy.zeros((MPARAMS, MPARAMS), dtype=np.complex128)
        makeJHWJ[(math.ceil(NROWS / 256), MPARAMS * MPARAMS), (256, 1)](
            JHWJ_d.view(np.float64).reshape(MPARAMS, MPARAMS, 2),
            cupy.asarray(uvws),
            cupy.asarray(lmndash),
            cupy.asarray(wavelengths),
            cupy.asarray(ants1),
            cupy.asarray(ants2),
            cupy.asarray(fluxes),
            cupy.asarray(ddphases),
            cupy.asarray(weights),
            cupy.asarray(X)
        )

        np.testing.assert_allclose(expected, cupy.asnumpy(JHWJ_d))

        expected = np.conj(J.T) @ (weights * (data - model)).reshape(-1, 1)

        JHWr_d = cupy.zeros((MPARAMS, 1), dtype=np.complex128)
        makeJHWr[(math.ceil(NROWS / 256), MPARAMS), (256, 1)](
            JHWr_d.view(np.float64).reshape(MPARAMS, 2),
            cupy.asarray(uvws),
            cupy.asarray(lmndash),
            cupy.asarray(wavelengths),
            cupy.asarray(ants1),
            cupy.asarray(ants2),
            cupy.asarray(fluxes),
            cupy.asarray(ddphases),
            cupy.asarray(weights),
            cupy.asarray(data),
            cupy.asarray(X)
        )
