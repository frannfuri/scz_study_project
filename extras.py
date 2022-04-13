import numpy as np

def MAD(sequence):
    abs_dev = np.abs(sequence - np.median(sequence))
    return np.median(abs_dev)

def robust_z_score(sequence):
    dev = sequence - np.median(sequence)
    mad = MAD(sequence)
    return dev/(1.4826*mad)
