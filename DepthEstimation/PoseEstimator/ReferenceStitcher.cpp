/**
@brief ReferenceStitcher.h
C++ source file for compositing images into panorams
@author Shane Yuan
@date Apr 17, 2018
*/
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <thread>
#include <chrono>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "FeatureMatch.h"
#include "CameraParamEstimator.h"
#include "Compositor.h"
#include "ReferenceStitcher.h"

/****************************************************************************************/
/*                                   calibration data                                   */
/****************************************************************************************/
calib::CalibrateParamRef::CalibrateParamRef() {}
calib::CalibrateParamRef::~CalibrateParamRef() {}
calib::CalibrateParamRef::CalibrateParamRef(const CalibrateParamRef & param) {
	this->focal = param.focal;
	this->R = param.R;
	this->mask = param.mask;
}
calib::CalibrateParamRef& calib::CalibrateParamRef::operator=(const calib::CalibrateParamRef & param) {
	this->focal = param.focal;
	this->R = param.R;
	this->mask = param.mask;
	return *this;
}
cv::Mat calib::CalibrateParamRef::getIntrinsicMatrix() {
	cv::Mat K(3, 3, CV_32F);
	K.setTo(cv::Scalar(0));
	K.at<float>(0, 0) = focal;
	K.at<float>(1, 1) = focal;
	K.at<float>(0, 2) = static_cast<float>(mask.cols) / 2;
	K.at<float>(1, 2) = static_cast<float>(mask.rows) / 2;
	K.at<float>(2, 2) = 1.0f;
	return K;
}

/****************************************************************************************/
/*                             Reference Calibration Class                              */
/****************************************************************************************/
namespace calib {
	struct InsideData {
		// calibration data
		std::vector<cv::Mat> imgs;
		// feature match smart pointer
		std::shared_ptr<calib::FeatureMatch> matchPtr;
		// camera parameters estimation smart pointer
		std::shared_ptr<calib::CameraParamEstimator> estimatorPtr;
		// image compositor estimation smart pointer
		std::shared_ptr<calib::Compositor> compositorPtr;
	};
};

calib::ReferenceCalibrate::ReferenceCalibrate() {}
calib::ReferenceCalibrate::~ReferenceCalibrate() {}

/**
@brief init google log
@param char* programname: input program name
@return
*/
int calib::ReferenceCalibrate::initGoogleLog(
	char* programname) {
	google::InitGoogleLogging(programname);
	data_ = reinterpret_cast<InsideData*>(new InsideData());
	isInit = true;
	// get pointer
	InsideData* data = reinterpret_cast<InsideData*>(data_);
	data->matchPtr = std::make_shared<calib::FeatureMatch>();
	data->estimatorPtr = std::make_shared<calib::CameraParamEstimator>();
	data->compositorPtr = std::make_shared<calib::Compositor>();
	return 0;
}

/**
@brief calibrate function for the whole camera array
@param float calibScale: scale used for calibration
@return int
*/
int calib::ReferenceCalibrate::calibrate(
	std::vector<cv::Mat> imgs, cv::Mat connection, 
	std::vector<CalibrateParamRef> & params) {
	// get pointer
	InsideData* data = reinterpret_cast<InsideData*>(data_);
	if (imgs.size() > 1)
	{
		data->matchPtr->init(imgs, connection);
		data->matchPtr->match();
		data->estimatorPtr->init(imgs, connection, data->matchPtr->getImageFeatures(),
			data->matchPtr->getMatchesInfo());
		data->estimatorPtr->estimate();
		data->compositorPtr->init(imgs, data->estimatorPtr->getCameraParams());
		data->compositorPtr->composite();
		// get parameters
		params.resize(imgs.size());
		std::vector<calib::CameraParams> camParams = data->estimatorPtr->getCameraParams();
		for (size_t i = 0; i < imgs.size(); i++) {
			params[i].K = camParams[i].K();
			params[i].focal = camParams[i].focal;
			params[i].R = camParams[i].R;
			params[i].mask = data->compositorPtr->imgMaskGraphCutBackward[i];
		}
	}
	else
	{
		params.resize(imgs.size());
		for (size_t i = 0; i < imgs.size(); i++) {
			params[i].focal = 2000.0f;
			params[i].R = cv::Mat::eye(3, 3, CV_64F);
			params[i].mask.create(imgs[0].size(), CV_8U);
			params[i].mask.setTo(cv::Scalar(255));
		}
	}
	return 0;
}