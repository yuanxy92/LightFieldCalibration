/**
@brief StereoCalibration.cpp
Online stereo calibration class
@author Shane Yuan
@date Apr 24, 2018
*/

// npp
#include <npp.h>
#include <nppi_compression_functions.h>

#include "SysUtil.hpp"
#include <opencv2/core/utility.hpp>
#include "StereoCalibration.h"

StereoCalibration::StereoCalibration() : chessBoardSize(11, 8) {}
StereoCalibration::~StereoCalibration() {}

/**
@brief init stereo camera calibrator
@param cv::Mat K1: intrinsic parameter matrix of first camera
@param cv::Mat K2: intrinsic parameter matrix of second camera
@param cv::Size chessBoardSize: input chessboard size
@return int
*/
int StereoCalibration::init(cv::Mat K1, cv::Mat K2, cv::Size chessBoardSize) {
	this->K1 = K1.clone();
	this->K2 = K2.clone();
	this->chessBoardSize = chessBoardSize;
	return 0;
}

// Calculates rotation matrix to euler angles
// The result is the same as MATLAB except the order
// of the euler angles ( x and z are swapped ).
cv::Vec3f rotationMatrixToEulerAngles(cv::Mat &R) {
	//assert(isRotationMatrix(R));
	float sy = sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + R.at<double>(1, 0) * R.at<double>(1, 0));
	bool singular = sy < 1e-6; // If
	float x, y, z;
	if (!singular) {
		x = atan2(R.at<double>(2, 1), R.at<double>(2, 2));
		y = atan2(-R.at<double>(2, 0), sy);
		z = atan2(R.at<double>(1, 0), R.at<double>(0, 0));
	}
	else {
		x = atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
		y = atan2(-R.at<double>(2, 0), sy);
		z = 0;
	}
	return cv::Vec3f(x, y, z);
}

/**
@brief estimate extrinsic matrix in real time
@param cv::Mat img1: input image of the first camera
@param cv::Mat img2: input image of the second camera
@return int
*/
int StereoCalibration::estimate(cv::Mat img1, cv::Mat img2) {
	// start record time
	cv::TickMeter tm;
	tm.start();
	// resize image
	cv::Size smallSize(img1.cols / 4, img1.rows / 4);
	cv::Mat smallImg1, smallImg2;
	cv::resize(img1, smallImg1, smallSize);
	cv::resize(img2, smallImg2, smallSize);
	display.create(smallImg1.rows, smallImg1.cols * 2, CV_8UC3);
	// change to gray image
	cv::cvtColor(img1, this->img1, cv::COLOR_BGR2GRAY);
	cv::cvtColor(img2, this->img2, cv::COLOR_BGR2GRAY);
	// find corners in chess board
	bool isFound1 = cv::findChessboardCorners(img1, chessBoardSize, 
		corner1, cv::CALIB_CB_ADAPTIVE_THRESH 
		+ cv::CALIB_CB_FAST_CHECK);
	if (isFound1) {
		cv::cornerSubPix(this->img1, corner1, cv::Size(11, 11), cv::Size(-1, -1),
			cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
	}
	if (!isFound1) {
		cv::Rect rect(0, 0, smallImg1.cols, smallImg1.rows);
		smallImg1.copyTo(display(rect));
		rect.x += smallImg1.cols;
		smallImg2.copyTo(display(rect));
		cv::imshow("corners", display);
		cv::waitKey(5);
		return -1;
	}
	bool isFound2 = cv::findChessboardCorners(img2, chessBoardSize,
		corner2, cv::CALIB_CB_ADAPTIVE_THRESH 
		+ cv::CALIB_CB_FAST_CHECK);
	if (isFound2) {
		cv::cornerSubPix(this->img2, corner2, cv::Size(11, 11), cv::Size(-1, -1),
			cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
	}
	if (!isFound2) {
		cv::Rect rect(0, 0, smallImg1.cols, smallImg1.rows);
		smallImg1.copyTo(display(rect));
		rect.x += smallImg1.cols;
		smallImg2.copyTo(display(rect));
		cv::imshow("corners", display);
		cv::waitKey(5);
		return -1;
	}
	// check
	if (corner1.rows != corner2.rows) {
		cv::Rect rect(0, 0, smallImg1.cols, smallImg1.rows);
		smallImg1.copyTo(display(rect));
		rect.x += smallImg1.cols;
		smallImg2.copyTo(display(rect));
		cv::imshow("corners", display);
		cv::waitKey(5);
		return -1;
	}
	// draw corner points
	cv::Mat corner1_small, corner2_small;
	corner1.convertTo(corner1_small, CV_32F, 0.25);
	corner2.convertTo(corner2_small, CV_32F, 0.25);
	drawChessboardCorners(smallImg1, chessBoardSize, cv::Mat(corner1_small), isFound1);
	drawChessboardCorners(smallImg2, chessBoardSize, cv::Mat(corner2_small), isFound2);
	cv::Rect rect(0, 0, smallImg1.cols, smallImg1.rows);
	smallImg1.copyTo(display(rect));
	rect.x += smallImg1.cols;
	smallImg2.copyTo(display(rect));
	cv::imshow("corners", display);
	cv::waitKey(5);
	// calculate rotation matrix
	cv::Mat R, T;
	cv::Vec3f angles;
	cv::Mat E = cv::findEssentialMat(corner1, corner2, this->K1, cv::RANSAC);
	cv::recoverPose(E, corner1, corner2, this->K1, R, T);
	angles = rotationMatrixToEulerAngles(R);
	system("cls");
	SysUtil::infoOutput("Rotation:");
	for (int row = 0; row < 3; row++) {
		for (int col = 0; col < 3; col++) {
			printf("%06.5lf ", R.at<double>(row, col));
		}
		printf("\n");
	}
	printf("%06.5lf ", angles.val[0]);
	printf("%06.5lf ", angles.val[1]);
	printf("%06.5lf ", angles.val[2]);
	printf("\n");
	SysUtil::infoOutput("Translation:");
	printf("%06.5lf ", T.at<double>(0, 0));
	printf("%06.5lf ", T.at<double>(0, 1));
	printf("%06.5lf ", T.at<double>(0, 2));
	printf("\n");

	float val = acos((R.at<double>(0, 0) + R.at<double>(1, 1) + R.at<double>(2, 2) - 1) / 2);
	SysUtil::infoOutput("Angle:");
	printf("%06.5lf ", val);
	printf("\n");
	// end recording time
	tm.stop();
	std::cout << cv::format("Find corner points, cost %f miliseconds ...", tm.getTimeMilli())
		<< std::endl;
	return 0;
}


cv::cuda::GpuMat StereoCalibration::applyWhiteBalance(cv::cuda::GpuMat img, 
	float red, float green, float blue) {
	Npp32f wbTwist[3][4] = {
		{ 1.0, 0.0, 0.0, 0.0 },
		{ 0.0, 1.0, 0.0, 0.0 },
		{ 0.0, 0.0, 1.0, 0.0 }
	};
	// update wbTwist
	wbTwist[0][0] = blue;
	wbTwist[1][1] = green;
	wbTwist[2][2] = red;
	NppiSize osize;
	osize.width = img.cols;
	osize.height = img.rows;
	nppiColorTwist32f_8u_C3IR(img.data,
		img.step, osize, wbTwist);
	return img;
}