import numpy as np
from numpy.linalg import pinv

def linear_regression(data):
    shape = np.array(data).shape
    #XTX = np.dot(np.transpose(data[:, :3]), data[:, :3])
    pseudo_inverse = np.linalg.pinv(data[:, :shape[1]-1])
    w = np.dot(pseudo_inverse, data[:, shape[1]-1])
    error_in_vector = data[:, shape[1]-1] - np.dot(data[:, :shape[1]-1], w) 
    error_in = np.linalg.norm(error_in_vector) ** 2
    error_in = error_in / data.shape[0]
    return w, error_in
