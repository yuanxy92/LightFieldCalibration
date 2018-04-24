/**
@brief StereoCalibration.h
Online stereo calibration class
@author Shane Yuan
@date Apr 24, 2018
*/

#ifndef __LIGHT_FIELD_CALIBRATION_STEREO__
#define __LIGHT_FIELD_CALIBRATION_STEREO__

#include <cstdio>
#include <iostream>
#include <fstream>
#include <cstdlib>

#include <opencv2/opencv.hpp>

class StereoCalibration {
private:
	cv::Mat K1;
	cv::Mat K2;
public:

private:

public:
	StereoCalibration();
	~StereoCalibration();

	/**
	@brief init stereo camera calibrator
	@param cv::Mat K1: intrinsic parameter matrix of first camera
	@param cv::Mat K2: intrinsic parameter matrix of second camera
	@return int
	*/
	int init(cv::Mat K1, cv::Mat K2);
};


#endif