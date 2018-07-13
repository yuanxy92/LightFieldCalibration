#conda install pillow matplotlib
import copy
import math
import os
import shutil
from skimage.io import imread, imsave
import py3d as py3d
import numpy as np
import matplotlib.pyplot as plt
import util.camera_param_loader as loader

startind = 0
imgnum = 330
fragnum = 30
fragstep = 5

# path
datapath = 'E:\\Project\\LightFieldGiga\\data\\data1\\result'
outpath = 'E:\\Project\\LightFieldGiga\\data\\data1\\result_fusion'
bindir = 'E:\\Project\\LightFieldGiga\\SparseToDenseStereo\\build\\bin\\Release\\'
calibrate_exename = '%s\\PoseEstimator.exe' % bindir
fragmentfusion_exename = '%s\\MakeFragments.exe' % bindir
registration_exename = '%s\\RegistrateFragments.exe' % bindir

def get_pcd_from_rgbd_image(color_name, depth_name):
    color_raw = py3d.read_image(color_name)
    depth_raw = py3d.read_image(depth_name)
    rgbd_image = py3d.create_rgbd_image_from_color_and_depth(
        color_raw, depth_raw, 100, 300, False);
    pinhole_camera_intrinsic = py3d.read_pinhole_camera_intrinsic(
        "E:\\Project\\LightFieldGiga\\SparseToDenseStereo\\camera.json")
    pcd = py3d.create_point_cloud_from_rgbd_image(rgbd_image, pinhole_camera_intrinsic)
    return pcd

def make_fragments():
    # generate fragment 
    startInd = startind
    endInd = startInd + fragnum - 1
    representInd = math.floor((startInd + endInd) / 2)

    while endInd <= imgnum - 1:
        print('Process fragment %d to %d, with representative %d ...' % (startInd, endInd, representInd))
        # calibration, only R, T is small, so we assume T = 0
        outpath_ = '%s\\fragment_%d_%d_%d' % (outpath, startInd, endInd, representInd)
        if not os.path.isdir(outpath_):
            os.mkdir(outpath_)
        command = '%s %s %d %d %s' % (calibrate_exename, datapath, startInd, endInd, outpath_)
        os.system(command)
        # fusion to make a fragment
        command = '%s %s %d %d %d %s' % (fragmentfusion_exename, datapath, startInd, endInd, representInd, outpath_)
        os.system(command)

        # process next fragment
        startInd = startInd + fragstep
        endInd = startInd + fragnum - 1
        representInd = math.floor((startInd + endInd) / 2)

def registrate_fragments():
    # generate fragment 
    startInd = startind
    endInd = startInd + fragnum - 1
    representInd = math.floor((startInd + endInd) / 2)

    # copy to new folder to generate new dataset
    index = 0
    finalpath = '%s\\final' % outpath
    if not os.path.isdir(finalpath):
        os.mkdir(finalpath) 
    while endInd <= imgnum - 1:
        print('registrate fragment %d_%d_%d ...' % (startInd, endInd, representInd)) 
        inputpath = '%s\\fragment_%d_%d_%d' % (outpath, startInd, endInd, representInd)
        color_name = '%s\\color.png' % inputpath
        depth_name = '%s\\depth.png' % inputpath
        color_name_out = '%s\\%04d_left.jpg' % (finalpath, index)
        depth_name_out = '%s\\%04d_depth.png' % (finalpath, index)
        # copy file
        img = imread(color_name)
        imsave(color_name_out, img)
        shutil.copyfile(depth_name, depth_name_out)
        index = index + 1
        # process next fragment
        startInd = startInd + fragstep
        endInd = startInd + fragnum - 1
        representInd = math.floor((startInd + endInd) / 2) 
    
    # apply calibration
    print('Calibrate final result ...')
    command = '%s %s %d %d %s' % (calibrate_exename, finalpath, 0, index - 1, finalpath)
    os.system(command)

    # apply registrate on sphere
    print('Registrate final result ...')
    command = '%s %s %d %d %s' % (registration_exename, finalpath, 0, index - 1, 'mesh_texture')
    os.system(command)

if __name__ == "__main__":
    # make_fragments()
    registrate_fragments()