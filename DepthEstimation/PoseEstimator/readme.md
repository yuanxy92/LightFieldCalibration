# Robust Panorama Stitcher
## Introduction
Panorama stitching project based on OpenCV and ceres-solver. OpenCV is used to extract and match SURF features. Ceres-solver is used to do bundle adjustment (Symbolic solution is used, much faster than OpenCV's numeric solution).

### Notice
No seamfinder is implemented in this project. If you need, please use the OpenCV seamfinder.

### Error function
Two error functions are provided: BundleAdjustmentRay and BundleAdjustmentReproj. Default is the second one (seems better than the first one).

## Precompiled C++ library
link: https://pan.baidu.com/s/1nxh4ghb passwd: 7byr

OpenCV with GPU support and Ceres-solver is need. The pre-complied ceres-solver is complied without blas, lapack support (maybe a little slower). I will fix it in the future.

## Reference 
[1] Openpano: http://ppwwyyxx.com/2016/How-to-Write-a-Panorama-Stitcher/