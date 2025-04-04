import numpy as np

def radec_to_lmndash(ra, dec, ra0, dec0):
    l = np.cos(dec) * np.sin(ra - ra0)
    m = np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(ra - ra0)

    return np.array([l, m, ndash(l, m)], dtype=ra.dtype).T

def lm_to_radec(l, m, ra0, dec0):
    n = np.sqrt(1 - l**2 - m**2)
    delta_ra = np.arctan2(l, n * np.cos(dec0) - m * np.sin(dec0))
    ra = ra0 + delta_ra
    dec = np.arcsin(m * np.cos(dec0) + n * np.sin(dec0))
    return np.array([ra, dec]).T

def ndash(l, m):
    r2 = l**2 + m**2
    r2[r2 > 1] = 1
    return  -r2 / (1 + np.sqrt(1 - r2))