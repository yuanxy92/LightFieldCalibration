/**
@brief StereoCalibration.cpp
Online stereo calibration class
@author Shane Yuan
@date Apr 24, 2018
*/

#include "StereoCalibration.h"

StereoCalibration::StereoCalibration() {}
StereoCalibration::~StereoCalibration() {}

/**
@brief init stereo camera calibrator
@param cv::Mat K1: intrinsic parameter matrix of first camera
@param cv::Mat K2: intrinsic parameter matrix of second camera
@return int
*/
int StereoCalibration::init(cv::Mat K1, cv::Mat K2) {
	this->K1 = K1.clone();
	this->K2 = K2.clone();
	return 0;
}

