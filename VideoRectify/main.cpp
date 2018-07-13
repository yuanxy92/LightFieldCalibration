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

int main2(int argc, char* argv[])
{
	std::string dir = std::string(argv[1]);
	std::string leftname = cv::format("%s/%s", dir.c_str(), argv[2]);
	std::string rightname = cv::format("%s/%s", dir.c_str(), argv[3]);
	std::string leftoutname = cv::format("%s/%s", dir.c_str(), argv[4]);
	std::string rightoutname = cv::format("%s/%s", dir.c_str(), argv[5]);

	int width, height;
	cv::Mat left = cv::imread(leftname);
	cv::Mat right = cv::imread(rightname);
	width = left.cols;
	height = left.rows;
	cv::Size imgsize(width, height);
	StereoRectify sr;
	std::string intname = cv::format("%s/intrinsics.yml", dir.c_str());
	std::string extname = cv::format("%s/extrinsics.yml", dir.c_str());
	sr.init(intname, extname, imgsize);
	sr.rectify(left, right);

	cv::imwrite(leftoutname, left);
	cv::imwrite(rightoutname, right);
}


int main(int argc, char* argv[]) {
	main2(argc, argv);
	return 0;
	std::string dir = std::string(argv[1]);
	std::string leftname = cv::format("%s/%s", dir.c_str(), argv[2]);
	std::string rightname = cv::format("%s/%s", dir.c_str(), argv[3]);
	std::string leftoutname = cv::format("%s/%s", dir.c_str(), argv[4]);
	std::string rightoutname = cv::format("%s/%s", dir.c_str(), argv[5]);


	int width, height, frame_count;
	cv::Mat left, right;
	right = cv::imread(leftname);
	left = cv::imread(rightname);
	cv::Size imgsize = left.size();
	
	StereoRectify sr;
	std::string intname = cv::format("%s/intrinsics.yml", dir.c_str());
	std::string extname = cv::format("%s/extrinsics.yml", dir.c_str());
	sr.init(intname, extname, imgsize);

	sr.rectify(left, right);
	cv::imwrite(leftoutname, left);
	cv::imwrite(rightoutname, right);

	return 0;
}
