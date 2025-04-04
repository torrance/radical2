from numba import njit, prange
import numpy as np

from radical.util import matmul_NxNx2x2, matmul_AHxB, matmul_AxBH, trace

def phasesol_scalar(data, model, weights, ddphases, ants1, ants2, fluxes, phases, wavelengths):
    # Convert from phase [rad] to phasor
    ddphases = np.cos(ddphases) + 1j * np.sin(ddphases)

    nants = ddphases.shape[0]
    nsources = ddphases.shape[1]

    numerator = np.zeros_like(ddphases)
    denominator = np.zeros_like(ddphases)

    for rowid, (ant1, ant2) in enumerate(zip(ants1, ants2)):
        # Solving for antenna 1
        # i = ant1
        # j = ant2
        # k = sources
        delta = data[rowid] - model[rowid]  # [chans]
        cik = ddphases[ant1, :]  # [sources]
        cjk = ddphases[ant2, :]  # [sources]

        phasor = np.cos(phases[rowid][:, None] / wavelengths[None, :]) + 1j * np.sin(phases[rowid][:, None] / wavelengths[None, :]) # [sources, chans]

        numerator[ant1, :] += np.nansum(
            (delta[None, :] * cjk[:, None] * fluxes * phasor + (cik * cjk * np.conj(cjk))[:, None] * fluxes**2) * weights[rowid][None, :],
            axis=1
        )

        denominator[ant1, :] += np.nansum(
            (cjk * np.conj(cjk))[:, None] * fluxes**2 * weights[rowid][None, :],
            axis=1
        )

        # Solving for antenna 2
        # i = ant2
        # j = ant1
        delta = np.conj(delta)
        phasor = np.conj(phasor)
        cik = ddphases[ant2, :]
        cjk = ddphases[ant1, :]

        numerator[ant2, :] += np.nansum(
            (delta[None, :] * cjk[:, None] * fluxes * phasor + (cik * cjk * np.conj(cjk))[:, None] * fluxes**2) * weights[rowid][None, :],
            axis=1
        )

        denominator[ant2, :] += np.nansum(
            (cjk * np.conj(cjk))[:, None] * fluxes**2 * weights[rowid][None, :],
            axis=1
        )

    return np.angle(numerator / denominator)  # covert from phasor to phase [rad]

def phasesol(data, model, weights, ddphases, ants1, ants2, fluxes, phases, wavelengths):
    # Convert from phase [rad] to phasor
    ddphases = np.cos(ddphases) + 1j * np.sin(ddphases)

    numerator, denominator = _phasesol(data, model, weights, ddphases, ants1, ants2, fluxes, phases, wavelengths)
    return np.angle(numerator / denominator)  # convert from phasor to phase [rad]

@njit(parallel=True)
def _phasesol(data, model, weights, ddphases, ants1, ants2, fluxes, phases, wavelengths):
    # Numerator = Tr[ c_jk e^(-2pi i phi_ijk) IH_k (V - M) + c_ik IH_k I_k ]
    # Denominator = Tr[ IH_k I_k ]
    numerator = np.zeros_like(ddphases)
    denominator = np.zeros_like(ddphases)

    delta = data - model

    for isrc in prange(ddphases.shape[1]):
        # Threadlocal 2x2 array
        tmp2x2 = np.empty((2, 2), dtype=np.complex128)

        for irow, (ant1, ant2) in enumerate(zip(ants1, ants2)):
            for ichan in range(data.shape[1]):
                # ANTENNA 1
                phase = phases[irow, isrc] / wavelengths[ichan]
                phasor = np.cos(phase) + 1j * np.sin(phase)

                c_ik = ddphases[ant1, isrc]
                c_jk = ddphases[ant2, isrc]

                # ANTENNA 1: c_jk e^(-2pi i phi_ijk) IH_k (V - M)
                matmul_AHxB(fluxes[isrc, ichan], delta[irow, ichan], tmp2x2)
                tmp2x2 *= weights[irow, ichan]
                numerator[ant1, isrc] += c_jk * phasor * trace(tmp2x2)

                # ANTENNA 2: c_ik e^(+2pi i phi_ijk) I_k (V - M)H
                matmul_AxBH(fluxes[isrc, ichan], delta[irow, ichan], tmp2x2)
                tmp2x2 *= weights[irow, ichan]
                numerator[ant2, isrc] += c_ik * np.conj(phasor) * trace(tmp2x2)

                # BOTH ANTENNAS: IH_k I_k
                matmul_AHxB(fluxes[isrc, ichan], fluxes[isrc, ichan], tmp2x2)
                tmp2x2 *= weights[irow, ichan]
                numerator[ant1, isrc] += c_ik * trace(tmp2x2)
                numerator[ant2, isrc] += c_jk * trace(tmp2x2)

                # BOTH ANTENNAS: DENOMINATOR
                denominator[ant1, isrc] += trace(tmp2x2)
                denominator[ant2, isrc] += trace(tmp2x2)

    return numerator, denominator