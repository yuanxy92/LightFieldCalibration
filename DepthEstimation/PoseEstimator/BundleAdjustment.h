/**
@brief BundleAdjustment.h
C++ head file for bundle adjustment using ceres-solver
@author Shane Yuan
@date Feb 13, 2018
*/

#ifndef _ROBUST_STITCHER_BUNDLE_ADJUSTMENT_H_
#define _ROBUST_STITCHER_BUNDLE_ADJUSTMENT_H_ 

#include <cmath>
#include <cstdio>
#include <iostream>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "FeatureMatch.h"

namespace calib {
    /**
    @brief bundle adjustment class
    */
    class BundleAdjustment {
    private:
        // input camera parameters
        cv::Mat connection;
        std::vector<Imagefeature> features;
        std::vector<CameraParams> cameras;
	    std::vector<Matchesinfo> matchesInfo;
        // camera parameter for bundle adjustment
        double* cameraParam = NULL;
    public:

    private:
        /**
        @brief init camera parameters
        @return int
        */
        int initCameraParamMultipleFocal();

		/**
		@brief function to solve bundle adjustment problem
		@return int
		*/
		int solveMultipleFocal();

		/**
		@brief init camera parameters
		@return int
		*/
		int initCameraParamSingleFocal();

		/**
		@brief function to solve bundle adjustment problem
		@return int
		*/
		int solveSingleFocal();

    public:
        BundleAdjustment();
        ~BundleAdjustment(); 

        /**
	    @brief set input for camera parameter estimation
		@param cv::Mat connection: input connection matrix
        @param std::vector<Imagefeature> features: input matching points features
        @param std::vector<CameraParams> cameras: input cameras
		@param std::vector<Matchesinfo> matchesInfo: input matching infos
		@return int
		*/
		int init(cv::Mat connection,
            std::vector<Imagefeature> features,
            std::vector<CameraParams> cameras,
			std::vector<Matchesinfo> matchesInfo);

        /**
        @brief function to solve bundle adjustment problem
		@param int focalSetting : input focal length
		0 : use multiple focal length (every camera has its own focal length) 
		1 : use single focal length (every camera use the same focal length) 
        @return int
        */
        int solve(int focalSetting = 0);

        /**
		@brief get estimated camera params
		@return std::vector<CameraParams>: returned camera params 
		*/
		std::vector<CameraParams> getCameraParams();
	};

};


#endif