/**
@brief main.cpp
Online stereo calibration main file
@author Shane Yuan
@date Apr 24, 2018
*/

#include "StereoCalibration.h"

#include "NPPJpegCoder.h"
#include "GenCameraDriver.h"

#include "SysUtil.hpp"

//cv::Mat colorBGR2BayerRG(cv::Mat img) {
//	cv::Mat out(img.rows, img.cols, CV_8U);
//	size_t rows = img.rows / 2;
//	size_t cols = img.cols / 2;
//	for (size_t row = 0; row < rows; row++) {
//		for (size_t col = 0; col < cols; col++) {
//			out.at<uchar>(row * 2, col * 2) = img.at<cv::Vec3b>(row * 2, col * 2).val[1];
//			out.at<uchar>(row * 2, col * 2 + 1) = img.at<cv::Vec3b>(row * 2, col * 2 + 1).val[2];
//			out.at<uchar>(row * 2 + 1, col * 2) = img.at<cv::Vec3b>(row * 2 + 1, col * 2).val[0];
//			out.at<uchar>(row * 2 + 1, col * 2 + 1) = img.at<cv::Vec3b>(row * 2 + 1, col * 2 + 1).val[1];
//		}
//	}
//	return out;
//}

int main(int argc, char* argv[]) {
	std::vector<cam::GenCamInfo> camInfos;
	std::shared_ptr<cam::GenCamera> cameraPtr = cam::createCamera(cam::CameraModel::XIMEA_xiC);
	cameraPtr->init();
	// set camera setting
	cameraPtr->startCapture();
	cameraPtr->setFPS(-1, 10);
	cameraPtr->setAutoExposure(-1, cam::Status::on);
	cameraPtr->setAutoExposureLevel(-1, 20);
	cameraPtr->makeSetEffective();
	// set capturing setting
	cameraPtr->setCaptureMode(cam::GenCamCaptureMode::Continous, 200);
	// get camera info
	cameraPtr->getCamInfos(camInfos);
	cameraPtr->startCaptureThreads();

	cv::Mat K1 = cv::Mat::zeros(3, 3, CV_32F);
	K1.at<float>(0, 0) = 7138;
	K1.at<float>(1, 1) = 7138;
	K1.at<float>(2, 2) = 1;
	K1.at<float>(0, 2) = 1232;
	K1.at<float>(1, 2) = 1028;
	cv::Mat K2 = K1.clone();
	cv::Size chessBoardSize(11, 8);

	cv::Mat img1 = cv::imread("1.png");
	cv::Mat img2 = cv::imread("2.png");

	StereoCalibration stereoCalibrator;
	stereoCalibrator.init(K1, K2, chessBoardSize);

	std::vector<cam::Imagedata> imgdatas(2);
	std::vector<cv::Mat> imgs(2);
	std::vector<cv::Mat> imgs_c(2);
	std::vector<cv::cuda::GpuMat> imgs_bayer_d(2);
	std::vector<cv::cuda::GpuMat> imgs_d(2);

	cv::Mat wbMat(camInfos[0].height, camInfos[0].width, CV_8UC3);
	wbMat.setTo(cv::Scalar(2, 1, 2));
	cv::cuda::GpuMat wbMat_d;
	wbMat_d.upload(wbMat);

	SysUtil::sleep(1000);
	int nframe = 0;
	cv::Rect rect;
	for (;;) {
		std::cout << nframe++ << std::endl;
		cameraPtr->captureFrame(imgdatas);

		imgs[0] = cv::Mat(camInfos[0].height, camInfos[0].width,
			CV_8U, reinterpret_cast<void*>(imgdatas[0].data));
		imgs[1] = cv::Mat(camInfos[1].height, camInfos[1].width,
			CV_8U, reinterpret_cast<void*>(imgdatas[1].data));

		imgs_bayer_d[0].upload(imgs[0]);
		imgs_bayer_d[1].upload(imgs[1]);

		cv::cuda::demosaicing(imgs_bayer_d[0], imgs_d[0], cv::COLOR_BayerBG2BGR, -1);
		cv::cuda::demosaicing(imgs_bayer_d[1], imgs_d[1], cv::COLOR_BayerBG2BGR, -1);

		imgs_d[0] = StereoCalibration::applyWhiteBalance(imgs_d[0], 2, 1, 2);
		imgs_d[1] = StereoCalibration::applyWhiteBalance(imgs_d[1], 2, 1, 2);

		imgs_d[0].download(imgs_c[0]);
		imgs_d[1].download(imgs_c[1]);
#if 0
		if (nframe == 1) {
			cv::Mat img;
			cv::Size sizeSmall(imgs_c[0].cols / 4, imgs_c[0].rows / 4);
			cv::resize(imgs_c[0], img, sizeSmall);
			rect = cv::selectROI(img, true);
			rect.x *= 4;
			rect.y *= 4;
			rect.width *= 4;
			rect.height *= 4;
		}
		else {
			cv::Mat img1, img2;
			imgs_c[0](rect).copyTo(img1);
			imgs_c[1](rect).copyTo(img2);
			cv::Mat showImg(img1.rows, img1.cols * 2, CV_8UC3);
			cv::Rect rect2(0, 0, img1.cols, img1.rows);
			img1.copyTo(showImg(rect2));
			rect2.x += img1.cols;
			img2.copyTo(showImg(rect2));
			cv::imshow("calib", showImg);
			cv::waitKey(5);
			cv::Mat result;
			cv::matchTemplate(img1, img2, result, cv::TM_CCOEFF_NORMED);
			system("cls");
			printf("Current zncc: %f\n", result.at<float>(0, 0));
		}
#else
		stereoCalibrator.estimate(imgs_c[0], imgs_c[1]);
#endif
		SysUtil::sleep(5);
	}

	cameraPtr->stopCaptureThreads();
	cameraPtr->release();

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