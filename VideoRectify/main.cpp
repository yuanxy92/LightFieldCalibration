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

	cv::VideoCapture leftVideo(leftname);
	cv::VideoCapture rightVideo(rightname);

	int width, height, frame_count;
	width = leftVideo.get(cv::CAP_PROP_FRAME_WIDTH);
	height = leftVideo.get(cv::CAP_PROP_FRAME_HEIGHT);
	frame_count = leftVideo.get(cv::CAP_PROP_FRAME_COUNT);
	cv::Size imgsize(width, height);

	cv::VideoWriter leftOutVideo;
	cv::VideoWriter rightOutVideo;
	
	StereoRectify sr;
	std::string intname = cv::format("%s/intrinsics.yml", dir.c_str());
	std::string extname = cv::format("%s/extrinsics.yml", dir.c_str());
	sr.init(intname, extname, imgsize);

	cv::Mat left, right;
	int ind = 0;
	for (size_t i = 0; i < frame_count; i++) {
		leftVideo >> left;
		rightVideo >> right;
		sr.rectify(left, right);
		if (i == 0) {
			cv::Size imageSize = left.size();
			leftOutVideo.open(leftoutname, cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), 20, imageSize);
			rightOutVideo.open(rightoutname, cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), 20, imageSize);
		}
		leftOutVideo << left;
		rightOutVideo << right;
		if (i % 10 == 0) {
			cv::Size size(left.cols / 2, left.rows / 2);
			cv::resize(left, left, size);
			cv::resize(right, right, size);
			cv::imwrite(cv::format("data/0/%04d.jpg", ind), left);
			cv::imwrite(cv::format("data/1/%04d.jpg", ind), right);
			ind++;
		}
	}

	leftVideo.release();
	rightVideo.release();
	leftOutVideo.release();
	rightOutVideo.release();
	
	return 0;
}
