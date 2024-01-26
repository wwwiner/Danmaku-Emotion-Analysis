#!/usr/bin/env python
# coding: utf-8
import numpy as np

def cross_product(A, B):
    A = A.reshape((A.shape[0], 1))
    B = B.reshape((B.shape[0], 1))

    AB = np.hstack((A, B))
    BA = np.hstack((B, A))

    cp = np.cross(AB, BA)

    return cp.reshape((cp.shape[0], 1))
