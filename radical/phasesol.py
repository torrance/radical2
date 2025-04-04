import math

import cupy
from numba import njit, cuda
import numpy as np

from radical.predict import predict
from radical.util import conj

def phasesol(mset, lmndash, fluxes):
    NANTS = mset.nants
    NSRCS = fluxes.shape[0]
    NPARAMS = 2
    MPARAMS = NPARAMS * NSRCS

    # Load reusbale data into GPU memory
    uvw_d = cupy.asarray(mset.uvw)
    wavelengths_d = cupy.asarray(mset.wavelength)
    lmndash_d = cupy.asarray(lmndash)
    fluxes_d = cupy.asarray(fluxes)
    ants1_d = cupy.asarray(mset.antenna1)
    ants2_d = cupy.asarray(mset.antenna2)
    weights_d = cupy.asarray(mset.weight)
    data_d = cupy.asarray(mset.data)

    # X is a vector that contains the associated polynomial value
    # e.g. x, y, x^2, y^2 .... for each antenna
    antsx, antsy, _ = mset.antennas(0).T
    X = np.empty((NANTS, NPARAMS), dtype=np.float64)
    X[:, 0] = antsx
    X[:, 1] = antsy
    X_d = cupy.asarray(X)

    # Initialize params
    params = np.zeros((NSRCS, NPARAMS), dtype=np.complex64) # srcs, params
    ddphases = makeddphases(params, X)

    print("Initializing residuals...", flush=True, end="")
    r_d = cupy.empty((mset.shape), dtype=np.complex64)
    predict[math.ceil(r_d.shape[0] * r_d.shape[1] / 512), 512](
        r_d, uvw_d, wavelengths_d, ants1_d, ants2_d, lmndash_d,
        fluxes_d, cupy.asarray(ddphases)
    )
    r_d -= data_d
    r_d *= weights_d
    print(" Done.")

    lastcost = np.linalg.norm(r_d.reshape(-1))**2
    print("Initial cost:", lastcost)

    # The Levinbergh-Marquardt damping factor, often denoted lambda
    # This is modified each iteration depending on whether a step
    # increased or decreased the associated cost.
    dampingfactor = 1

    for i in range(50):
        print(f"=== ITERATION {i} ===")

        print("Calculating JHWJ...", flush=True, end="")
        JHWJ_d = cupy.zeros((MPARAMS, MPARAMS), dtype=np.complex128)
        makeJHWJ[(math.ceil(r_d.shape[0] / 256), MPARAMS * MPARAMS), (256, 1)](
            JHWJ_d.view(np.float64).reshape(MPARAMS, MPARAMS, 2),
            uvw_d,
            lmndash_d,
            wavelengths_d,
            ants1_d,
            ants2_d,
            fluxes_d,
            cupy.asarray(ddphases),
            weights_d,
            X_d
        )
        cupy.cuda.runtime.deviceSynchronize()
        print(" Done.")

        print("Calculating JHr...", flush=True, end="")
        JHWr_d = cupy.zeros((MPARAMS, 1), dtype=np.complex128)
        makeJHWr[(math.ceil(r_d.shape[0] / 256), MPARAMS), (256, 1)](
            JHWr_d.view(np.float64).reshape(MPARAMS, 2),
            uvw_d,
            lmndash_d,
            wavelengths_d,
            ants1_d,
            ants2_d,
            fluxes_d,
            cupy.asarray(ddphases),
            weights_d,
            data_d,
            X_d
        )
        cupy.cuda.runtime.deviceSynchronize()
        print(" Done.")

        delta = cupy.asnumpy(cupy.linalg.solve(
            JHWJ_d + dampingfactor * cupy.diag(cupy.diag(JHWJ_d)),
            JHWr_d
        ))

        nextparams = params + delta.reshape(params.shape)
        nextddphases = makeddphases(nextparams, X)

        print("Calculating residuals...", flush=True, end="")
        predict[math.ceil(r_d.shape[0] * r_d.shape[1] / 512), 512](
            r_d, uvw_d, wavelengths_d, ants1_d, ants2_d, lmndash_d,
            fluxes_d, cupy.asarray(nextddphases)
        )
        r_d -= data_d
        r_d *= weights_d
        cost = np.linalg.norm(r_d)**2
        print(" Done.")

        if cost > lastcost:
            print("Rejecting step!")
            dampingfactor *= 2
        else:
            print("Cost:", cost)
            print("Mean step size:", np.mean(np.abs(delta)))
            print("Mean phase delta:", np.mean(np.abs(nextddphases - ddphases)))

            dampingfactor /= 5
            lastcost = cost
            params = nextparams
            ddphases = nextddphases
            print(params)

    return ddphases


def makeddphases(params, X):
    # params [source, params]
    # X [ants, x]
    # ddphases [ ants, source ]
    return np.einsum('kj,ij->ik', params, X, dtype=np.complex128)


@cuda.jit
def makeJHWJ(JHWJ, uvws, lmndash, wavelengths, ants1, ants2, fluxes, ddphases, weights, X):
    """
    JHWJ [params x params x real/imag]
    uvws [rows x 3]
    lmndash [ sources x 3]
    wavelengths [chans]
    ants1, ants2 [rows]
    fluxes [sources x chans x pol x pol]
    ddphases [antenna, src]
    weights [rows x chans x pol x pol]
    X [ants, params]

    """
    shared = cuda.shared.array(256, dtype=np.complex128)

    # Axis X refers to each measurement set row
    xidx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    xstride = cuda.blockDim.x * cuda.gridDim.x

    # Block id Y refers to index in the JHWJ.
    # There are (N^2 + N)/2 unique elements in the matrix.
    yidx = cuda.blockIdx.y

    # Determine which cell we are in
    i = yidx // JHWJ.shape[1]
    k = yidx % JHWJ.shape[1]

    # The params are ordered: major=source, minor=source params
    NPARAMS = X.shape[1]

    isrc = i // NPARAMS
    ksrc = k // NPARAMS

    param_i = i % NPARAMS
    param_k = k % NPARAMS

    l_i, m_i, ndash_i = lmndash[isrc]
    l_k, m_k, ndash_k = lmndash[ksrc]
    acc = 0 + 0j

    for irow in range(xidx, weights.shape[0], xstride):
        u, v, w = uvws[irow]
        ant1, ant2 = ants1[irow], ants2[irow]

        deltaphase_i = ddphases[ant1, isrc] - ddphases[ant2, isrc]
        deltaphase_k = ddphases[ant1, ksrc] - ddphases[ant2, ksrc]

        rowacc = 0 + 0j

        # Along each row, calculate: V_i^* V_k W
        for ichan in range((weights.shape[1])):
            phase_i = 2 * np.pi * (u * l_i + v * m_i + w * ndash_i) / wavelengths[ichan] + deltaphase_i
            phase_k = 2 * np.pi * (u * l_k + v * m_k + w * ndash_k) / wavelengths[ichan] + deltaphase_k

            # deltphase is complex: this handles the real and complex parts of the phase separately
            phasor = (
                math.cos((phase_k - phase_i).real) + 1j * math.sin((phase_k - phase_i).real)
            ) * math.exp(-(phase_k + phase_i).imag)

            rowacc += weights[irow, ichan, 0, 0] * fluxes[ksrc, ichan, 0, 0] * conj(fluxes[isrc, ichan, 0, 0]) * phasor
            rowacc += weights[irow, ichan, 0, 1] * fluxes[ksrc, ichan, 0, 1] * conj(fluxes[isrc, ichan, 0, 1]) * phasor
            rowacc += weights[irow, ichan, 1, 0] * fluxes[ksrc, ichan, 1, 0] * conj(fluxes[isrc, ichan, 1, 0]) * phasor
            rowacc += weights[irow, ichan, 1, 1] * fluxes[ksrc, ichan, 1, 1] * conj(fluxes[isrc, ichan, 1, 1]) * phasor

        rowacc *= (X[ant1, param_i] - X[ant2, param_i]) * (X[ant1, param_k] - X[ant2, param_k])

        acc += rowacc

    # Perform the reduction over the block
    shared[cuda.threadIdx.x] = acc

    s = 128
    while s > 0:
        cuda.syncthreads()
        if cuda.threadIdx.x < s:
            shared[cuda.threadIdx.x] += shared[cuda.threadIdx.x + s]
        s >>= 1

    # And perform final reduction over the grid
    if cuda.threadIdx.x == 0:
        cuda.atomic.add(JHWJ, (i, k, 0), shared[0].real)
        cuda.atomic.add(JHWJ, (i, k, 1), shared[0].imag)

@cuda.jit
def makeJHWr(JHWr, uvws, lmndash, wavelengths, ants1, ants2, fluxes, ddphases, weights, data, X):
    shared = cuda.shared.array(256, dtype=np.complex64)
    tmp = cuda.local.array((2, 2), dtype=np.complex128)

    # Axis X refers to each measurement set row
    xidx = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    xstride = cuda.blockDim.x * cuda.gridDim.x

    # Block id Y refers to index in the JTr
    yidx = cuda.blockIdx.y

    # The params are ordered: major=source, minor=source params
    NPARAMS = X.shape[1]

    isrc = yidx // NPARAMS  # source ID
    param_i = yidx % NPARAMS

    l_i, m_i, ndash_i = lmndash[isrc]

    acc = 0 + 0j

    for irow in range(xidx, weights.shape[0], xstride):
        ant1, ant2 = ants1[irow], ants2[irow]

        rowacc = 0 + 0j

        # Along each row, calculate: W V_i^* (D - V)
        for ichan in range((weights.shape[1])):
            # First, calculate residual
            tmp[0, 0] = data[irow, ichan, 0, 0]
            tmp[0, 1] = data[irow, ichan, 0, 1]
            tmp[1, 0] = data[irow, ichan, 1, 0]
            tmp[1, 1] = data[irow, ichan, 1, 1]

            u = uvws[irow, 0] / wavelengths[ichan]
            v = uvws[irow, 1] / wavelengths[ichan]
            w = uvws[irow, 2] / wavelengths[ichan]

            for k in range(fluxes.shape[0]):
                l, m, ndash = lmndash[k]
                phase = 2 * np.pi * (u * l + v * m + w * ndash) + ddphases[ant1, k] - ddphases[ant2, k]

                phasor = (
                    math.cos(phase.real) + 1j * math.sin(phase.real)
                ) * math.exp(-phase.imag)

                tmp[0, 0] -= fluxes[k, ichan, 0, 0] * phasor
                tmp[0, 1] -= fluxes[k, ichan, 0, 1] * phasor
                tmp[1, 0] -= fluxes[k, ichan, 1, 0] * phasor
                tmp[1, 1] -= fluxes[k, ichan, 1, 1] * phasor

            # Now calculate v_ji
            l, m, ndash = lmndash[isrc]
            phase = 2 * np.pi * (u * l + v * m + w * ndash) + ddphases[ant1, isrc] - ddphases[ant2, isrc]

            phasor = (
                math.cos(phase.real) + 1j * math.sin(phase.real)
            ) * math.exp(-phase.imag)

            # And multiple the residuals by its conjugate and weight
            tmp[0, 0] *= conj(fluxes[isrc, ichan, 0, 0] * phasor) * weights[irow, ichan, 0, 0]
            tmp[0, 1] *= conj(fluxes[isrc, ichan, 0, 1] * phasor) * weights[irow, ichan, 0, 1]
            tmp[1, 0] *= conj(fluxes[isrc, ichan, 1, 0] * phasor) * weights[irow, ichan, 1, 0]
            tmp[1, 1] *= conj(fluxes[isrc, ichan, 1, 1] * phasor) * weights[irow, ichan, 1, 1]

            # Finally sum the row
            rowacc += tmp[0, 0] + tmp[0, 1] + tmp[1, 0] + tmp[1, 1]

        rowacc *= -1j * (X[ant1, param_i] - X[ant2, param_i])

        acc += rowacc

    # Perform the reduction over the block
    shared[cuda.threadIdx.x] = acc

    s = 128
    while s > 0:
        cuda.syncthreads()
        if cuda.threadIdx.x < s:
            shared[cuda.threadIdx.x] += shared[cuda.threadIdx.x + s]
        s >>= 1

    # And perform final reduction over the grid
    if cuda.threadIdx.x == 0:
        cuda.atomic.add(JHWr, (yidx, 0), shared[0].real)
        cuda.atomic.add(JHWr, (yidx, 1), shared[0].imag)
