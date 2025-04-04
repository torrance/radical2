import numpy as np
import unittest

from radical.util import *

class Util(unittest.TestCase):
    def test_matmul_AxB(self):
        A = (np.random.rand(100, 2, 2) + 1j * np.random.rand(100, 2, 2)).astype(np.complex64)
        B = (np.random.rand(100, 2, 2) + 1j * np.random.rand(100, 2, 2)).astype(np.complex64)

        C = np.empty((100, 2, 2), dtype=np.complex64)
        for i in range(100):
            matmul_AxB(A[i], B[i], C[i])

        D = np.linalg.matmul(A, B)
        np.testing.assert_allclose(C, D, rtol=1e-6)

    def test_matmul_AxBH(self):
        A = (np.random.rand(100, 2, 2) + 1j * np.random.rand(100, 2, 2)).astype(np.complex64)
        B = (np.random.rand(100, 2, 2) + 1j * np.random.rand(100, 2, 2)).astype(np.complex64)

        C = np.empty((100, 2, 2), dtype=np.complex64)
        for i in range(100):
            matmul_AxBH(A[i], B[i], C[i])

        D = np.linalg.matmul(A, np.conj(np.transpose(B, [0, 2, 1])))
        np.testing.assert_allclose(C, D, rtol=1e-6)

    def test_matmul_AHxB(self):
        A = (np.random.rand(100, 2, 2) + 1j * np.random.rand(100, 2, 2)).astype(np.complex64)
        B = (np.random.rand(100, 2, 2) + 1j * np.random.rand(100, 2, 2)).astype(np.complex64)

        C = np.empty((100, 2, 2), dtype=np.complex64)
        for i in range(100):
            matmul_AHxB(A[i], B[i], C[i])

        D = np.linalg.matmul(np.conj(np.transpose(A, [0, 2, 1])), B)
        np.testing.assert_allclose(C, D, rtol=1e-6)

    def test_matmul_AHxBH(self):
        A = (np.random.rand(100, 2, 2) + 1j * np.random.rand(100, 2, 2)).astype(np.complex64)
        B = (np.random.rand(100, 2, 2) + 1j * np.random.rand(100, 2, 2)).astype(np.complex64)

        C = np.empty((100, 2, 2), dtype=np.complex64)
        for i in range(100):
            matmul_AHxBH(A[i], B[i], C[i])

        D = np.linalg.matmul(
            np.conj(np.transpose(A, [0, 2, 1])), np.conj(np.transpose(B, [0, 2, 1]))
        )
        np.testing.assert_allclose(C, D, rtol=1e-6)

    def test_matmul_Nx2x2(self):
        A = (np.random.rand(100, 2, 2) + 1j * np.random.rand(100, 2, 2)).astype(np.complex64)
        B = (np.random.rand(100, 2, 2) + 1j * np.random.rand(100, 2, 2)).astype(np.complex64)

        C = A @ B
        D = matmul_Nx2x2(A, B)

        np.testing.assert_allclose(C, D, rtol=1e-6)

    def test_matmul_NxNx2x2(self):
        A = (np.random.rand(100, 100, 2, 2) + 1j * np.random.rand(100, 100, 2, 2)).astype(np.complex64)
        B = (np.random.rand(100, 100, 2, 2) + 1j * np.random.rand(100, 100, 2, 2)).astype(np.complex64)

        C = A @ B
        D = matmul_NxNx2x2(A, B)

        np.testing.assert_allclose(C, D, rtol=1e-6)

    def test_inv(self):
        A = (np.random.rand(100, 100, 2, 2) + 1j * np.random.rand(100, 100, 2, 2)).astype(np.complex64)

        B = np.linalg.inv(A)
        C = inv_NxNx2x2(A)

        np.testing.assert_allclose(B, C, rtol=1e-6)

if __name__ == '__main__':
    unittest.main()