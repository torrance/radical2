
import numba
from numba import njit, prange
import numpy as np

from radical.util import matmul_Nx2x2, inv_NxNx2x2

def antsol(data, model, weights, ants1, ants2):
    nants = max(ants1.max(), ants2.max()) + 1
    nchans = data.shape[1]

    # [antid, chans, pol, pol]
    jones0 = np.zeros((nants, nchans, 2, 2), dtype=np.complex64)
    jones0[:, :, 0, 0] = 1
    jones0[:, :, 1, 1] = 1

    # Track failed solutions
    ok = np.ones((nants, nchans), dtype=bool)
    distances = np.zeros((nants, nchans))

    for i in range(20):
        # These are accumlation variables pertainging to
        # sum ( D J M^H) and sum ( M J^H J M^H) respectively.
        numerator = np.zeros((nants, nchans, 2, 2), dtype=np.complex64)
        denominator = np.zeros((nants, nchans, 2, 2), dtype=np.complex64)

        for rowid, (ant1, ant2) in enumerate(zip(ants1, ants2)):
            print(f"\rCalibration round {i} {100 * (rowid + 1) / len(ants1):0.2f}%", flush=True, end="")
            # Andre's calibrate: ( D J M^H ) / ( M J^H J M^H )

            # Update ant1
            MH = np.conj(np.transpose(model[rowid, :, :, :], [0, 2, 1]))  # [chan, pol, pol]
            JMH = matmul_Nx2x2(jones0[ant2, :, :, :], MH)  # [chans, pol, pol]
            MJH = np.conj(np.transpose(JMH, [0, 2, 1]))

            numerator[ant1] += matmul_Nx2x2(data[rowid, :, :, :], JMH) * weights[rowid, :, :, :]
            denominator[ant1] += matmul_Nx2x2(MJH, JMH) * weights[rowid, :, :, :]

            # Update ant2
            # We have to take the Heritian adjoint of each of the data the model
            # to account for the swapped order of the correlation
            # ie. ( D^H J M) / (M^H J^H J M)
            DH = np.conj(np.transpose(data[rowid, :, :, :], [0, 2, 1]))
            JM = matmul_Nx2x2(jones0[ant1, :, :, :], model[rowid, :, :, :])
            MHJH = np.conj(np.transpose(JM, [0, 2, 1]))

            numerator[ant2] += matmul_Nx2x2(DH, JM) * weights[rowid, :, :, :]
            denominator[ant2] += matmul_Nx2x2(MHJH, JM) * weights[rowid, :, :, :]

        print("")

        # Calculate numerator / denominator
        # But np.linalg.inv fails if any matrix is singular,
        # so let's do it ourselves
        jones1 = numerator @ inv_NxNx2x2(denominator)

        # Mark any solutions where inversion failed as failed
        ok[~np.isfinite(jones1.sum(axis=(2, 3)))] = False
        print(f"{(~ok).sum()}/{ok.size} solutions failed")

        # If there are only 4 or fewer antenna left in a channel, mark the whole channel as failed
        ok[:, ok.sum(axis=0) <= 4] = False

        # Update jones as mean of jones and new jones
        jones0[:, :, :, :] = np.mean([jones0, jones1], axis=0)
        jones0[~ok] = 0  # This has the effect to ignore the failed solutions without checking for NaN

        # Calculate distance between solutions
        distances[:] = np.mean(
            ((jones1 - jones0) * np.conj(jones1 - jones0)).real, axis=(2, 3)
        )
        mediandistance = np.median(distances[ok])
        print(f"Median distance: {mediandistance}")

        if mediandistance < 1e-7:
            break

    # Set any solutions that haven't converged as failed
    ok[distances > 1e-5] = False
    print(f"{100 * ok.sum() / ok.size:.1f}% of solutions have converged")

    jones0[~ok] = np.nan

    # Invert jones0 so that it can be applied as J D J^H
    return inv_NxNx2x2(jones0)

def antsol_scalar(data, model, weights, ants1, ants2):
    nants = max(ants1.max(), ants2.max()) + 1
    nchans = data.shape[1]

    # [antid, chans]
    jones0 = np.ones((nants, nchans), dtype=np.complex64)

    # Track failed solutions
    ok = np.ones((nants, nchans), dtype=bool)
    distances = np.zeros((nants, nchans))

    for i in range(20):
        print(f"\rCalibration round {i}")

        numerator, denominator = internalloop(data, model, weights, jones0, ants1, ants2)
        jones1 = numerator / denominator

        # # These are accumlation variables pertainging to
        # # sum ( D J M^H) and sum ( M J^H J M^H) respectively.
        # numerator = np.zeros((nants, nchans), dtype=np.complex64)
        # denominator = np.zeros((nants, nchans), dtype=np.complex64)

        # for rowid, (ant1, ant2) in enumerate(zip(ants1, ants2)):
        #     print(f"\rCalibration round {i} {100 * (rowid + 1) / len(ants1):0.2f}%", flush=True, end="")
        #     # Andre's calibrate: ( D J M^H ) / ( M J^H J M^H )

        #     # Update ant1
        #     MH = np.conj(model[rowid, :])  # [chan, pol, pol]
        #     JMH = jones0[ant2, :] * MH  # [chans, pol, pol]
        #     MJH = np.conj(JMH)

        #     numerator[ant1] += data[rowid, :] * JMH * weights[rowid, :]
        #     denominator[ant1] += MJH * JMH * weights[rowid, :]

        #     # Update ant2
        #     # We have to take the Heritian adjoint of each of the data the model
        #     # to account for the swapped order of the correlation
        #     # ie. ( D^H J M) / (M^H J^H J M)
        #     DH = np.conj(data[rowid, :])
        #     JM = jones0[ant1, :] * model[rowid, :]
        #     MHJH = np.conj(JM)

        #     numerator[ant2] += DH * JM * weights[rowid, :]
        #     denominator[ant2] += MHJH * JM * weights[rowid, :]

        # print("")

        # # Calculate numerator / denominator
        # jones1 = numerator / denominator

        # Mark any solutions where inversion failed as failed
        ok[~np.isfinite(jones1)] = False
        print(f"{(~ok).sum()}/{ok.size} solutions failed")

        # If there are only 4 or fewer antenna left in a channel, mark the whole channel as failed
        ok[:, ok.sum(axis=0) <= 4] = False

        # Calculate distance between solutions
        distances[:] = ((jones1 - jones0) * np.conj(jones1 - jones0)).real

        mediandistance = np.median(distances[ok])
        print(f"Median distance: {mediandistance}")

        # Update jones as mean of jones and new jones
        jones0[:, :] = np.mean([jones0, jones1], axis=0)

        jones0[~ok] = 0  # This has the effect to ignore the failed solutions without checking for NaN

        if mediandistance < 1e-7:
            break

    # Set any solutions that haven't converged as failed
    ok[distances > 1e-5] = False
    print(f"{100 * ok.sum() / ok.size:.1f}% of solutions have converged")

    jones0[~ok] = np.nan

    # Invert jones0 so that it can be applied as J D J^H
    return 1 / jones0

@njit
def internalloop(data, model, weights, jones0, ants1, ants2):
    # These are accumlation variables pertaining to
    # sum ( D J M^H) and sum ( M J^H J M^H) respectively.
    numerator = np.zeros_like(jones0)
    denominator = np.zeros_like(jones0)

    for irow in range(data.shape[0]):
        ant1 = ants1[irow]
        ant2 = ants2[irow]

        for ichan in range(data.shape[1]):
            # D J M*
            numerator[ant1, ichan] += data[irow, ichan] * jones0[ant2, ichan] * np.conj(model[irow, ichan]) * weights[irow, ichan]

            # |J| |M|
            denominator[ant1, ichan] += jones0[ant2, ichan] * np.conj(jones0[ant2, ichan]) * model[irow, ichan] * np.conj(model[irow, ichan]) * weights[irow, ichan]

            # Update ant2
            # We have to take the Heritian adjoint of each of the data the model
            # to account for the swapped order of the correlation
            # ie. ( D^H J M) / (M^H J^H J M)

            # D* J M
            numerator[ant2, ichan] += np.conj(data[irow, ichan]) * jones0[ant1, ichan] * model[irow, ichan] * weights[irow, ichan]

            # |J| |M|
            denominator[ant2, ichan] += jones0[ant1, ichan] * np.conj(jones0[ant1, ichan]) * model[irow, ichan] * np.conj(model[irow, ichan]) * weights[irow, ichan]

    return numerator, denominator