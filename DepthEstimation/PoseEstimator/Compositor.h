/**
@brief Compositor.h
C++ head file for compositing images into panorams
@author Shane Yuan
@date Feb 9, 2018
*/

#ifndef _ROBUST_STITCHER_COMPOSITOR_H_
#define _ROBUST_STITCHER_COMPOSITOR_H_ 

// std
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>

// opencv
#include <opencv2/opencv.hpp>

#include "CameraParamEstimator.h"
#include "Warper.h"

namespace calib {
	class Compositor {
	private:
		
	public:
		std::vector<cv::Mat> imgs;
		std::vector<calib::CameraParams> cameras;

		std::vector<cv::Point> corners;
		std::vector<cv::Size> sizes;
		std::vector<cv::Rect> roiRect;
		std::vector<cv::Mat> imgWarped;
		std::vector<cv::Mat> imgMask;

		std::vector<cv::Mat> imgMaskGraphCut;
		std::vector<cv::Mat> imgMaskGraphCutBackward;

		cv::Mat result;
		cv::Mat result_mask;

	private:
		/**
		@brief apply graph cut to update mask
		@return 0
		*/
		int applyGraphCut();

	public:
		Compositor();
		~Compositor();

		/**
		@brief init function set image and cameras
		@param std::vector<cv::Mat> imgs: input images
		@param std::vector<calib::CameraParams> cameras: input camera params
		@return int
		*/
		int init(std::vector<cv::Mat> imgs, std::vector<calib::CameraParams> cameras);

		/**
		@brief composite panorama
		@return int
		*/
		int composite();

		/**
		@brief get stitching result
		@param cv::Mat & result: stitched color image
		@param cv::Mat & result_mask: stitched mask image
		@return int
		*/
		int getStitchResult(cv::Mat & result, cv::Mat & result_mask);
	};

};

#endif