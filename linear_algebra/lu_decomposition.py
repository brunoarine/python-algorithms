def lu_decomposition(mat, precision=1e-10):
    """
    Decomposes a square matrix into lower triangular (L) and upper triangular (U) matrices.
    :param mat: a 2-D square matrix in the form of nested lists or NumPy array
    :return: tuple (L, U)
    """
    n = len(mat)
    L = [([0.0] * n) for i in range(n)]
    U = [([0.0] * n) for i in range(n)]

    for i in range(n):
        # U
        for k in range(i, n):
            s1 = 0.0  # summation of L(i, j)*U(j, k)
            for j in range(i):
                s1 += L[i][j] * U[j][k]
            U[i][k] = mat[i][k] - s1

        # L
        for k in range(i, n):
            if i == k:
                # diagonal terms of L
                L[i][i] = 1.0
            else:
                s2 = 0.0  # summation of L(k, j)*U(j, i)
                for j in range(i):
                    s2 += L[k][j] * U[j][i]
                if U[i][i] == 0:
                    U[i][i] = precision
                L[k][i] = (mat[k][i] - s2) / U[i][i]
    return L, U

def diag(mat):
    n = len(mat)
    prod = 1
    for i in range(n):
        prod *= mat[i][i]
    return prod


def ludet(mat):
    L, U = lu_decomposition(mat)
    return diag(U) * diag(L)


if __name__ == "__main__":
    import numpy as np
    matA = [[0.0, 0.0, 0.0, 34.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0], [0.0, 0.0, 76.0, 0.0, 0.0, 95.0, 0.0, 0.0, 0.0, 0.0], [0.0, -77.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30.0, 49.0, 0.0], [-89.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 71.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 92.0, 0.0, 0.0], [0.0, -26.0, 0.0, 0.0, 0.0, 0.0, 0.0, 62.0, 0.0, 31.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 69.0, 0.0, 0.0], [-81.0, 0.0, -30.0, 0.0, -70.0, -41.0, -87.0, 0.0, 0.0, 0.0], [0.0, 0.0, -54.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -50.0, 0.0, -75.0, 0.0, 0.0, 0.0, 0.0]]
    L, U = lu_decomposition(matA, precision=1e-20)
    print("LU determinant:", diag(L)*diag(U))
    print("NumPy determinant:", np.linalg.det(matA))
    print()
    matB = [[0.0, 0.0, 0.0, 0.0, 76.0, 0.0, 73.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 94.0, 82.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 78.0, 35.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-42.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 0.0, 0.0], [0.0, -71.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 49.0], [-41.0, -54.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 28.0], [0.0, 0.0, 0.0, 0.0, -32.0, 0.0, 0.0, 0.0, 0.0, 55.0], [0.0, 0.0, -64.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 57.0], [0.0, 0.0, -47.0, 0.0, 0.0, -73.0, -53.0, -86.0, -16.0, 0.0]]
    L, U = lu_decomposition(matB, precision=1e-20)
    print("LU determinant:", diag(L)*diag(U))
    print("NumPy determinant:", np.linalg.det(matB))
    print()