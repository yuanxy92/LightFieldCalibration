/**
@brief main.cpp
Online stereo calibration main file
@author Shane Yuan
@date Apr 24, 2018
*/

#include "StereoCalibration.h"

int main(int argc, char* argv[]) {
	cv::Mat K1 = cv::Mat::zeros(3, 3, CV_32F);
	K1.at<float>(0, 0) = 7192.4;
	K1.at<float>(1, 1) = 7189.9;
	K1.at<float>(2, 2) = 1;
	K1.at<float>(0, 2) = 1197.5;
	K1.at<float>(1, 2) = 1025.1;
	cv::Mat K2 = K1.clone();
	cv::Size chessBoardSize(11, 8);

	cv::Mat img1 = cv::imread("1.png");
	cv::Mat img2 = cv::imread("2.png");

	StereoCalibration stereoCalibrator;
	stereoCalibrator.init(K1, K2, chessBoardSize);
	stereoCalibrator.estimate(img1, img2);

	return 0;
}