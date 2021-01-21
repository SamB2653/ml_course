import numpy as np

'''
For matrix multiplication, the number of columns in the first matrix must be equal to the number of rows in the second
matrix. The result matrix has the number of rows of the first and the number of columns of the second matrix.

A = [3, 5]
B = [4, 3]
A * B = C
C = [4, 5]

'''

a = np.array([[5, 1, 3],  # [3, 5]
              [3, 1, 1],
              [1, 2, 2],
              [3, 2, 4],
              [2, 2, 3]])

b = np.array([[1, 2, 3, 4],  # [4, 3]
             [1, 4, 2, 5],
             [3, 2, 2, 1]])

print("Matrix a:\n", a)
print("Matrix b:\n", b)
print("Using np.tensordot for dot product of a and b\n", np.tensordot(a, b, axes=1))  # new shape [4, 5]
print("Using np.inner for inner product of b and b\n", np.inner(b, b))  # keeps b shape
print("Using np.transpose for transpose of b\n", np.transpose(b))  # interchange of rows and columns.

'''
Non-commutativity:
An operation is commutative if, given two elements A and B such that the product AB is defined, then BA is also defined
abd AB = BA.

Example:
[0 1] [0 0]   [1 0]
[0 0] [1 0] = [0 0]
however;
[0 0] [0 1]   [0 0]
[1 0] [0 0] = [0 1]
'''


