import numpy as np
from numba import njit, cuda

@cuda.jit
def predict(data, uvws, wavelengths, ants1, ants2, lmndash, fluxes, ddphases):
    # Accumulation variable
    tmp = cuda.local.array((2, 2), dtype=data.dtype)

    for idx in range(cuda.grid(1), data.shape[0] * data.shape[1], cuda.gridsize(1)):
        i = idx // data.shape[1] # row
        j = idx % data.shape[1]  # channel

        tmp[:] = 0

        u, v, w = uvws[i]
        u /= wavelengths[j]
        v /= wavelengths[j]
        w /= wavelengths[j]

        ant1 = ants1[i]
        ant2 = ants2[i]

        for k in range(lmndash.shape[0]):  # k = source index
            l, m, ndash = lmndash[k]
            flux = fluxes[k, j]

            # Compute complex phasor
            phase = 2 * np.float32(np.pi) * (u * l + v * m + w * ndash) + ddphases[ant1, k] - ddphases[ant2, k]
            phasor = (np.cos(phase) + 1j * np.sin(phase))

            tmp[0, 0] += flux[0, 0] * phasor
            tmp[0, 1] += flux[0, 1] * phasor
            tmp[1, 0] += flux[1, 0] * phasor
            tmp[1, 1] += flux[1, 1] * phasor

        data[i, j, 0, 0] = tmp[0, 0]
        data[i, j, 0, 1] = tmp[0, 1]
        data[i, j, 1, 0] = tmp[1, 0]
        data[i, j, 1, 1] = tmp[1, 1]