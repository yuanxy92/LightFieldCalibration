/**
@brief warper class, warping function
@author Shane Yuan
@date Jun 1, 2017
*/

#ifndef _ROBUST_STITCHER_WARPER_H_
#define _ROBUST_STITCHER_WARPER_H_

#include <stdio.h>
#include <cstdlib>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/features2d.hpp>
#include <opencv2/stitching/warpers.hpp>
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <opencv2/stitching/detail/exposure_compensate.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <opencv2/stitching/detail/blenders.hpp>
#include <opencv2/stitching/detail/camera.hpp>

namespace calib {

	class SphericalWarper {
	private:
		// camera parameters
		float scale;
		float k[9];
		float rinv[9];
		float r_kinv[9];
		float k_rinv[9];
		float t[3];
		// output 
		cv::Rect outRect;
	public:

	private:
		// detect regions of interest on result
		int detectResultRoi(cv::Size srcSize, cv::Point &dstTL, cv::Point &dstBR);
	public:
		SphericalWarper();
		~SphericalWarper();

		// setscale
		int setScale(float scale);

		// set camera parameter
		int setCameraParams(cv::Mat K, cv::Mat R);
		int setCameraParams(cv::Mat K, cv::Mat R, cv::Mat T);

		// forward/backward mapping function
		int mapForward(float x, float y, float &u, float &v);
		int mapBackward(float u, float v, float &x, float &y);

		// build maps
		cv::Rect buildMapsForward(cv::Size srcSize, cv::Mat& xmap, cv::Mat& ymap);
		cv::Rect buildMapsBackward(cv::Size dstSize, cv::Mat& xmap, cv::Mat& ymap);
		// reload build maps function
		cv::Rect buildMapsForward(cv::Size srcSize, cv::Mat K, cv::Mat R, cv::Mat& xmap, cv::Mat& ymap);
		cv::Rect buildMapsBackward(cv::Size dstSize, cv::Mat K, cv::Mat R, cv::Mat& xmap, cv::Mat& ymap);

		// warp images
		cv::Point warpForward(cv::Mat src, cv::Mat& dst, int interMode);
		cv::Point warpBackward(cv::Mat src, cv::Size dstSize, cv::Mat& dst, int interMode);
		// reload warp function
		cv::Point warpForward(cv::Mat src, cv::Mat K, cv::Mat R, int interMode, int borderMode, cv::Mat& dst);
		cv::Point warpForward(cv::UMat src, cv::Mat K, cv::Mat R, int interMode, int borderMode, cv::UMat& dst);
	};
};

#endif