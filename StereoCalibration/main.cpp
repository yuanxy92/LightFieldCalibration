/**
@brief main.cpp
Online stereo calibration main file
@author Shane Yuan
@date Apr 24, 2018
*/

#include "StereoCalibration.h"

#include "NPPJpegCoder.h"

cv::Mat colorBGR2BayerRG(cv::Mat img) {
	cv::Mat out(img.rows, img.cols, CV_8U);
	size_t rows = img.rows / 2;
	size_t cols = img.cols / 2;
	for (size_t row = 0; row < rows; row++) {
		for (size_t col = 0; col < cols; col++) {
			out.at<uchar>(row * 2, col * 2) = img.at<cv::Vec3b>(row * 2, col * 2).val[1];
			out.at<uchar>(row * 2, col * 2 + 1) = img.at<cv::Vec3b>(row * 2, col * 2 + 1).val[2];
			out.at<uchar>(row * 2 + 1, col * 2) = img.at<cv::Vec3b>(row * 2 + 1, col * 2).val[0];
			out.at<uchar>(row * 2 + 1, col * 2 + 1) = img.at<cv::Vec3b>(row * 2 + 1, col * 2 + 1).val[1];
		}
	}
	return out;
}

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

	//int width = 4112;
	//int height = 3008;
	////int width = 4112;
	////int height = 3008;
	//cv::Mat img = cv::imread("raw.ref.bmp");
	//img(cv::Rect(0, 0, width, height)).copyTo(img);
	//cv::Mat bayerImg = colorBGR2BayerRG(img);
	//cv::cuda::GpuMat img_d(3000, 4000, CV_8U);
	//img_d.upload(bayerImg);
	//npp::NPPJpegCoder coder;
	//coder.init(width, height, 99);
	//coder.setWBRawType(false);
	//coder.setCfaBayerType(3);
	//coder.setWhiteBalanceGain(1.0f, 1.0f, 1.0f);
	//uchar* data = new uchar[width * height];
	//size_t maxlength = width * height;
	//size_t *length = new size_t;
	//cv::cuda::Stream stream;
	//coder.encode(img_d, data, length, maxlength, stream);
	//stream.waitForCompletion();

	//std::ofstream outputFile("test.jpg", std::ios::out | std::ios::binary);
	//outputFile.write((char*)data, *length);

	return 0;
}