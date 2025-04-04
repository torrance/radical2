from argparse import ArgumentParser
import math
import time

from astropy.coordinates import SkyCoord, AltAz
import cupy
import matplotlib.pyplot as plt
from numba import cuda
import numpy as np
import mwa_hyperbeam
import scipy
import yaml

import radical.antsol
from radical.mset import MS
from radical.phasesol import phasesol, makeJHWJ, makeJHWr
# from radical.phasesol import phasesol
from radical.predict import predict
from radical.transforms import radec_to_lmndash
from radical.util import matmul_NxNx2x2, inv_NxNx2x2

def run():
    parser = ArgumentParser()
    parser.add_argument("mset")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mwabeam", required=True)
    args = parser.parse_args()

    mset = MS(args.mset)

    print(mset.midlocation.lat, mset.midlocation.lon)

    with open(args.model) as f:
        sources = yaml.safe_load(f)["sources"]

    ras = np.array([np.deg2rad(source["ra"]) for source in sources], dtype=np.float32)
    decs = np.array([np.deg2rad(source["dec"]) for source in sources], dtype=np.float32)
    fluxes = np.array([source["flux"] for source in sources], dtype=np.float32)
    refreqs = np.array([source["freq"] for source in sources], dtype=np.float32)
    alphas = np.array([source["alpha"] for source in sources], dtype=np.float32)

    lmndash = radec_to_lmndash(ras, decs, mset.phasecenter.ra.rad, mset.phasecenter.dec.rad)

    # 1. Convert Stokes to instrumental
    fluxes = np.array([
        fluxes[:, 0] + fluxes[:, 1],       # XX = I + Q
        fluxes[:, 2] + 1j * fluxes[:, 3],  # XY = U + i V
        fluxes[:, 2] - 1j * fluxes[:, 3],  # YX = U - i V
        fluxes[:, 0] - fluxes[:, 1]        # YY = I - Q
    ], dtype=np.complex64).T.reshape(-1, 2, 2)

    # 2. Convert from instrumental to apparent instrumental
    coords = SkyCoord(ras, decs, unit=("rad", "rad"))
    altazs = coords.transform_to(AltAz(obstime=mset.midtime, location=mset.midlocation))
    beam = mwa_hyperbeam.FEEBeam(args.mwabeam)
    jones = beam.calc_jones_array(
        altazs.az.rad, np.pi / 2 - altazs.alt.rad,
        mset.midfreq, mset.mwadelays, [1] * 16,
        True, mset.midlocation.lat.rad, False
    ).reshape(-1, 2, 2)
    # fluxes = np.matmul(
    #     jones,
    #     np.matmul(
    #         fluxes, np.conj(np.transpose(jones, [0, 2, 1]))
    #     )
    # )

    # 3. Factor in spectral index [ sourceid, freq, pol, pol ]
    fluxes = fluxes[:, None, :, :] * ((mset.freq[None, :] / refreqs[:, None])**alphas[:, None])[:, :, None, None]
    print(fluxes.shape)

    phasesol(mset, lmndash, fluxes)
