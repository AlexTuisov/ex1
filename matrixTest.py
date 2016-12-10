from scipy.sparse import csr_matrix
import numpy as np

def matTest():
    mat1 = csr_matrix([[1,1,1],[2,2,2],[3,3,3]],dtype=int)
    vec = np.array([1,2,3])
    print(mat1.dot(vec))
    print(vec.sum())
    vec2  = csr_matrix([[1,1,1]],dtype=int)
    print(np.exp(vec))
    print(vec+vec2)
    print(csr_matrix(vec).multiply(2))