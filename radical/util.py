import numba
from numba import njit, prange, cuda
import numpy as np

@cuda.jit(device=True)
def conj(a):
    return a.real - 1j * a.imag

@njit([
    numba.void(numba.complex64[:, :], numba.complex64[:, :], numba.complex64[:, :]),
    numba.void(numba.complex128[:, :], numba.complex128[:, :], numba.complex128[:, :])
])
def matmul_AxB(A, B, C):
    assert(A.shape[0] == B.shape[0] == C.shape[0] == 2)
    assert(A.shape[1] == B.shape[1] == C.shape[1] == 2)
    C[0, 0] = A[0, 0] * B[0, 0] + A[0, 1] * B[1, 0]
    C[0, 1] = A[0, 0] * B[0, 1] + A[0, 1] * B[1, 1]
    C[1, 0] = A[1, 0] * B[0, 0] + A[1, 1] * B[1, 0]
    C[1, 1] = A[1, 0] * B[0, 1] + A[1, 1] * B[1, 1]

@njit([
    numba.void(numba.complex64[:, :], numba.complex64[:, :], numba.complex64[:, :]),
    numba.void(numba.complex128[:, :], numba.complex128[:, :], numba.complex128[:, :])
])
def matmul_AxBH(A, B, C):
    assert(A.shape[0] == B.shape[0] == C.shape[0] == 2)
    assert(A.shape[1] == B.shape[1] == C.shape[1] == 2)
    C[0, 0] = A[0, 0] * np.conj(B[0, 0]) + A[0, 1] * np.conj(B[0, 1])
    C[0, 1] = A[0, 0] * np.conj(B[1, 0]) + A[0, 1] * np.conj(B[1, 1])
    C[1, 0] = A[1, 0] * np.conj(B[0, 0]) + A[1, 1] * np.conj(B[0, 1])
    C[1, 1] = A[1, 0] * np.conj(B[1, 0]) + A[1, 1] * np.conj(B[1, 1])

@njit([
    numba.void(numba.complex64[:, :], numba.complex64[:, :], numba.complex64[:, :]),
    numba.void(numba.complex128[:, :], numba.complex128[:, :], numba.complex128[:, :])
])
def matmul_AHxB(A, B, C):
    assert(A.shape[0] == B.shape[0] == C.shape[0] == 2)
    assert(A.shape[1] == B.shape[1] == C.shape[1] == 2)
    C[0, 0] = np.conj(A[0, 0]) * B[0, 0] + np.conj(A[1, 0]) * B[1, 0]
    C[0, 1] = np.conj(A[0, 0]) * B[0, 1] + np.conj(A[1, 0]) * B[1, 1]
    C[1, 0] = np.conj(A[0, 1]) * B[0, 0] + np.conj(A[1, 1]) * B[1, 0]
    C[1, 1] = np.conj(A[0, 1]) * B[0, 1] + np.conj(A[1, 1]) * B[1, 1]

@njit([
    numba.void(numba.complex64[:, :], numba.complex64[:, :], numba.complex64[:, :]),
    numba.void(numba.complex128[:, :], numba.complex128[:, :], numba.complex128[:, :])
])
def matmul_AHxBH(A, B, C):
    assert(A.shape[0] == B.shape[0] == C.shape[0] == 2)
    assert(A.shape[1] == B.shape[1] == C.shape[1] == 2)
    C[0, 0] = np.conj(A[0, 0]) * np.conj(B[0, 0]) + np.conj(A[1, 0]) * np.conj(B[0, 1])
    C[0, 1] = np.conj(A[0, 0]) * np.conj(B[1, 0]) + np.conj(A[1, 0]) * np.conj(B[1, 1])
    C[1, 0] = np.conj(A[0, 1]) * np.conj(B[0, 0]) + np.conj(A[1, 1]) * np.conj(B[0, 1])
    C[1, 1] = np.conj(A[0, 1]) * np.conj(B[1, 0]) + np.conj(A[1, 1]) * np.conj(B[1, 1])

@njit([
    numba.complex64(numba.complex64[:, :]), numba.complex128(numba.complex128[:, :])
])
def trace(A):
    assert(A.shape[0] == 2)
    assert(A.shape[1] == 2)
    return A[0, 0] + A[1, 1]

@njit(numba.complex64[:, :, :](numba.complex64[:, :, :], numba.complex64[:, :, :]))
def matmul_Nx2x2(A, B):
    assert(A.shape[0] == B.shape[0])
    assert(A.shape[1] == B.shape[1] == 2)
    assert(A.shape[2] == B.shape[2] == 2)
    C = np.empty_like(A)

    for i in range(A.shape[0]):
        C[i, 0, 0] = A[i, 0, 0] * B[i, 0, 0] + A[i, 0, 1] * B[i, 1, 0]
        C[i, 0, 1] = A[i, 0, 0] * B[i, 0, 1] + A[i, 0, 1] * B[i, 1, 1]
        C[i, 1, 0] = A[i, 1, 0] * B[i, 0, 0] + A[i, 1, 1] * B[i, 1, 0]
        C[i, 1, 1] = A[i, 1, 0] * B[i, 0, 1] + A[i, 1, 1] * B[i, 1, 1]

    return C

@njit([
    numba.complex64[:, :, :, :](numba.complex64[:, :, :, :], numba.complex64[:, :, :, :]),
    numba.complex128[:, :, :, :](numba.complex128[:, :, :, :], numba.complex128[:, :, :, :])
], parallel=True)
def matmul_NxNx2x2(A, B):
    assert(A.shape[0] == B.shape[0])
    assert(A.shape[1] == B.shape[1])
    assert(A.shape[2] == B.shape[2] == 2)
    assert(A.shape[3] == B.shape[3] == 2)
    C = np.empty_like(A)

    for i in prange(A.shape[0]):
        for j in range(A.shape[1]):
            C[i, j, 0, 0] = A[i, j, 0, 0] * B[i, j, 0, 0] + A[i, j, 0, 1] * B[i, j, 1, 0]
            C[i, j, 0, 1] = A[i, j, 0, 0] * B[i, j, 0, 1] + A[i, j, 0, 1] * B[i, j, 1, 1]
            C[i, j, 1, 0] = A[i, j, 1, 0] * B[i, j, 0, 0] + A[i, j, 1, 1] * B[i, j, 1, 0]
            C[i, j, 1, 1] = A[i, j, 1, 0] * B[i, j, 0, 1] + A[i, j, 1, 1] * B[i, j, 1, 1]

    return C

def inv_NxNx2x2(A):
    B = np.empty_like(A)
    B[:, :, 0, 0] = A[:, :, 1, 1]
    B[:, :, 0, 1] = -A[:, :, 0, 1]
    B[:, :, 1, 0] = -A[:, :, 1, 0]
    B[:, :, 1, 1] = A[:, :, 0, 0]
    B /= np.linalg.det(A)[:, :, None, None]
    return B