/**
@brief ReferenceStitcher.h
C++ head file for compositing images into panorams
@author Shane Yuan
@date Apr 17, 2018
*/

#ifndef _ROBUST_STITCHER_H_
#define _ROBUST_STITCHER_H_ 

#if defined(_MSC_VER)
#ifdef REFERENCE_STITCHER_EXPORT
#define REFERENCE_STITCHER_DLL_EXPORT __declspec(dllexport)
#else
#define REFERENCE_STITCHER_DLL_EXPORT __declspec(dllimport)
#endif
#else
#define REFERENCE_STITCHER_DLL_EXPORT
#endif

// stl
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <chrono>

// opencv
#include <opencv2/opencv.hpp>

namespace calib {
	struct  CalibrateParamRef {
		float focal;
		cv::Mat K;
		cv::Mat R;
		cv::Mat mask;
		// constructor
		REFERENCE_STITCHER_DLL_EXPORT CalibrateParamRef();
		REFERENCE_STITCHER_DLL_EXPORT ~CalibrateParamRef();
		REFERENCE_STITCHER_DLL_EXPORT CalibrateParamRef(const CalibrateParamRef & param);
		REFERENCE_STITCHER_DLL_EXPORT CalibrateParamRef& operator=(const CalibrateParamRef & param);
		REFERENCE_STITCHER_DLL_EXPORT cv::Mat getIntrinsicMatrix();

	};

	class ReferenceCalibrate {
	private:
		void* data_;
		bool isInit;
	public:

	private:

	public:
		REFERENCE_STITCHER_DLL_EXPORT ReferenceCalibrate();
		REFERENCE_STITCHER_DLL_EXPORT ~ReferenceCalibrate();

		/**
		@brief init google log
		@param char* programname: input program name
		@return 
		*/
		REFERENCE_STITCHER_DLL_EXPORT int initGoogleLog(
			char* programname);

		/**
		@brief calibrate function for the whole camera array
		@param std::vector<cv::Mat> imgs: input images
		@param cv::Mat
		@return int
		*/
		REFERENCE_STITCHER_DLL_EXPORT int calibrate(
			std::vector<cv::Mat> imgs, cv::Mat connection,
			std::vector<CalibrateParamRef> & params);

	};

}

#endif
