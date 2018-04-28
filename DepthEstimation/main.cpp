/**
@brief main file 
*/

#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <memory>

#include <opencv2/opencv.hpp>
#include <opencv2/cudastereo.hpp>

#include "StereoRectify.h"

int main(int argc, char* argv[]) {
	cv::Mat left = cv::imread("00_00000.jpg");
	cv::Mat right = cv::imread("01_00000.jpg");
	cv::Size imgsize = left.size();

	StereoRectify sr;
	sr.init("intrinsics.yml", "extrinsics.yml", imgsize);
	sr.rectify(left, right);

	cv::Size size(left.cols, left.rows);

	cv::resize(left, left, size);
	cv::resize(right, right, size);

	cv::Mat disp;

	int mindisparity = 0;
	int ndisparities = 64;
	int SADWindowSize = 11;

	//SGBM  
	cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(mindisparity, ndisparities, SADWindowSize);
	int P1 = 8 * left.channels() * SADWindowSize* SADWindowSize;
	int P2 = 32 * left.channels() * SADWindowSize* SADWindowSize;
	sgbm->setP1(P1);
	sgbm->setP2(P2);

	sgbm->setPreFilterCap(15);
	sgbm->setUniquenessRatio(10);
	sgbm->setSpeckleRange(2);
	sgbm->setSpeckleWindowSize(100);
	sgbm->setDisp12MaxDiff(1);
	//sgbm->setMode(cv::StereoSGBM::MODE_HH);  

	sgbm->compute(left, right, disp);

	disp.convertTo(disp, CV_32F, 1.0 / 16);
	cv::Mat disp8U = cv::Mat(disp.rows, disp.cols, CV_8UC1);
	normalize(disp, disp8U, 0, 255, cv::NORM_MINMAX, CV_8UC1);

	cv::imwrite("SGBM.png", disp8U);
	cv::imwrite("left.png", left);
	cv::imwrite("right.png", right);
	return 0;
}
