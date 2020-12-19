import numpy as np


def gramSchmidtV1(matrix):
    assert np.size(matrix, 0) == np.size(matrix, 1)
    size = np.size(matrix, 0)
    Q = np.zeros((size, size))
    R = np.zeros((size, size))

    R[0, 0] = np.linalg.norm(matrix[:, 0])
    assert R[0, 0] != 0
    Q[:, 0:1] = matrix[:, 0]/R[0, 0]
    for j in range(1, size):
        approx = np.zeros((size,1))
        for i in range(0, j):
            R[i, j] = np.dot(np.transpose(matrix[:, j:j+1]), Q[:, i:i+1])
            approx += R[i, j]*Q[:, i:i+1];
        Q[:, j:j+1] = matrix[:, j]-approx;
        R[j, j] = np.linalg.norm(Q[:, j])
        assert R[j, j]!=0;
        Q[:, j:j+1]/= R[j,j]
    return Q,R;
    


m = np.matrix(np.random.rand(4, 4))
Q, R = gramSchmidtV1(m)
print(Q)
print(R)
print(np.dot(np.transpose(Q), Q))
print(m-np.dot(Q,R))
