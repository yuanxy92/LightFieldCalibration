from skimage.io import imread
from skimage.io import imsave
from skimage import img_as_uint

import scipy.misc

import matplotlib.pyplot as plt
import numpy as np
import os
from BilateralGrid import *
from BilateralSolver import *

def apply_fast_bilateral_solver(reference, target, confidence):
    im_shape = reference.shape[:2]
    assert(im_shape[0] == target.shape[0])
    assert(im_shape[1] == target.shape[1])
    assert(im_shape[0] == confidence.shape[0])
    assert(im_shape[1] == confidence.shape[1])
    grid_params = {
        'sigma_luma' : 5, # Brightness bandwidth
        'sigma_chroma': 5, # Color bandwidth
        'sigma_spatial': 6 # Spatial bandwidth
    }
    bs_params = {
        'lam': 256, # The strength of the smoothness parameter
        'A_diag_min': 1e-5, # Clamp the diagonal of the A diagonal in the Jacobi preconditioner.
        'cg_tol': 1e-5, # The tolerance on the convergence in PCG
        'cg_maxiter': 25 # The number of PCG iterations
    }
    grid = BilateralGrid(reference, **grid_params)
    t = target.reshape(-1,1).astype(np.float64) / (pow(2,16)-1)
    c = confidence.reshape(-1, 1).astype(np.float64) / (pow(2,16)-1)
    output_solver = BilateralSolver(grid, bs_params).solve(t, c).reshape(im_shape)

    return output_solver

if __name__ == "__main__":
    data_folder = 'E:\\Project\\LightFieldGiga\\data\\result'
    #data_folder = 'E:\\Project\\LightFieldGiga\\data'

    for i in range(332):
        print('Refine depth map %04d ...' % i)

        # The RGB image that whose edges we will respect
        reference = imread(os.path.join(data_folder, '%04d_left.jpg' % i))
        # The 1D image whose values we would like to filter
        target = imread(os.path.join(data_folder, '%04d_depth.png' % i))
        # A confidence image, representing how much we trust the values in "target".
        # Pixels with zero confidence are ignored.
        # Confidence can be set to all (2^16-1)'s to effectively disable it.
        confidence = imread(os.path.join(data_folder, '%04d_confidence.png' % i))

        output = apply_fast_bilateral_solver(reference, target, confidence)  
        output = output * (pow(2, 16) - 1)
        output = output.astype(np.uint16)

        imsave(os.path.join(data_folder, '%04d_depth_refine.png' % i), output)
