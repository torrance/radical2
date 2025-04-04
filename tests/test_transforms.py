import numpy as np
import unittest

from radical.transforms import radec_to_lmndash, lm_to_radec

class Util(unittest.TestCase):
    def test_lmradec_idempotence(self):
        ras = np.random.rand(1000) - 0.5
        decs = np.random.rand(1000) - 0.5

        lmndash = radec_to_lmndash(ras, decs, 0.1, -0.05)

        radecs = lm_to_radec(lmndash[:, 0], lmndash[:, 1], 0.1, -0.05)

        np.testing.assert_allclose(ras, radecs[:, 0])
        np.testing.assert_allclose(decs, radecs[:, 1])