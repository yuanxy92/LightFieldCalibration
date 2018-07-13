****************************************************************
****************************************************************
Partial Matlab code for "Bundled Camera Paths for Video Stabilization" (SIGGRAPH 2013)

by Shuaicheng Liu (shuaicheng@nus.edu.sg;liuyangmao@gmail.com)

Implementation of motion model estimation.
1. As-similar-as-possible warping.
2. Local homography estimation on mesh quads.

If you use/adapt our code in your work (either as a stand-alone tool or as a component of any algorithm), you need to appropriately cite our SIGGRAPH 2013 paper.

This code is for academic purpose only. Not for commercial/industrial activities.


The running time reported in the paper is from C++ implementation. This matlab code is a reference for those who would like to reimplement.


****************************************************************
****************************************************************
Usage: 

run demo.m

demo.m will read two images, and warp one image to another. Local homography calculated based on the corresponding warp quads

