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
	// init parameter
	cv::Mat K1;
	cv::Mat K2;
	cv::Size chessBoardSize;
	// input images
	cv::Mat img1;
	cv::Mat img2;
	// calculate corners
	cv::Mat corner1;
	cv::Mat corner2;
	// image for display
	cv::Mat display;
public:

private:

public:
	StereoCalibration();
	~StereoCalibration();

	/**
	@brief init stereo camera calibrator
	@param cv::Mat K1: input intrinsic parameter matrix of first camera
	@param cv::Mat K2: input intrinsic parameter matrix of second camera
	@param cv::Size chessBoardSize: input chessboard size
	@return int
	*/
	int init(cv::Mat K1, cv::Mat K2, cv::Size chessBoardSize);

	/**
	@brief estimate extrinsic matrix in real time
	@param cv::Mat img1: input image of the first camera
	@param cv::Mat img2: input image of the second camera
	@return int
	*/
	int estimate(cv::Mat img1, cv::Mat img2);
};


#endif