/**
@brief Compositor.cpp
C++ source file for compositing images into panorams
@author Shane Yuan
@date Feb 9, 2018
*/

#include <memory>

#include <opencv2/stitching/warpers.hpp>

#include "Compositor.h"

//#define _DEBUG_COMPOSITOR

calib::Compositor::Compositor() {}
calib::Compositor::~Compositor() {}

/**
@brief init function set image and cameras
@param std::vector<cv::Mat> imgs: input images
@param std::vector<calib::CameraParams> cameras: input camera params
@return int
*/
int calib::Compositor::init(std::vector<cv::Mat> imgs, 
	std::vector<calib::CameraParams> cameras) {
	this->imgs = imgs;
	this->cameras = cameras;
	return 0;
}

/**
@brief apply graph cut to update mask
@return int
*/
int calib::Compositor::applyGraphCut() {
	// copy data to GPU
	float subscale = 0.25;
	int camNum = imgs.size();
	imgMaskGraphCut.resize(camNum);
	imgMaskGraphCutBackward.resize(camNum);
	std::vector<cv::UMat> imgsd(camNum);
	std::vector<cv::UMat> masksd(camNum);
	std::vector<cv::Point> corners(camNum);
	for (int i = 0; i < camNum; i++) {
		cv::Mat img;
		cv::Size smallsize = cv::Size(roiRect[i].width * subscale, roiRect[i].height * subscale);
		cv::resize(imgs[i], img, smallsize, cv::INTER_LINEAR);
		img.convertTo(imgsd[i], CV_32F);
		cv::Mat mask;
		cv::resize(imgMask[i], mask, smallsize, cv::INTER_NEAREST);
		mask.convertTo(masksd[i], CV_8U);
		corners[i] = roiRect[i].tl() * subscale;
	}
	// seamfinder
	cv::Ptr<cv::detail::SeamFinder> seamFinder = cv::makePtr<cv::detail::GraphCutSeamFinder>
		(cv::detail::GraphCutSeamFinderBase::COST_COLOR);
	seamFinder->find(imgsd, corners, masksd);
	// update masks
	for (int i = 0; i < camNum; i++) {
		cv::Mat updateMask, finalMask;
		masksd[i].convertTo(updateMask, CV_8U);
		cv::resize(updateMask, updateMask, imgMask[i].size(), cv::INTER_NEAREST);
		// dilate to seam mask
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
		cv::dilate(updateMask, updateMask, kernel);
		cv::threshold(updateMask, updateMask, 1, 255, cv::THRESH_BINARY);
		cv::bitwise_and(updateMask, imgMask[i], imgMaskGraphCut[i]);
	}
	return 0;
}

/**
@brief composite panorama
@return int
*/
int calib::Compositor::composite() {
	// warp images
	cv::Ptr<cv::detail::SphericalWarper> w = cv::makePtr<cv::detail::SphericalWarper>(false);
	std::shared_ptr<cv::detail::Blender> blender_ = std::make_shared<cv::detail::MultiBandBlender>(false);
	corners.resize(imgs.size());
	sizes.resize(imgs.size());
	roiRect.resize(imgs.size());
	imgWarped.resize(imgs.size());
	imgMask.resize(imgs.size());
	w->setScale(cameras[0].focal);
	for (int i = 0; i < imgs.size(); i++) {
		// calculate warping filed
		cv::Mat K, R;
		cameras[i].K().convertTo(K, CV_32F);
		cameras[i].R.convertTo(R, CV_32F);
		cv::Mat initmask(imgs[i].rows, imgs[i].cols, CV_8U);
		initmask.setTo(cv::Scalar::all(255));
		corners[i] = w->warp(imgs[i], K, R, cv::INTER_LINEAR, cv::BORDER_CONSTANT, imgWarped[i]);
		w->warp(initmask, K, R, cv::INTER_NEAREST, cv::BORDER_CONSTANT, imgMask[i]);
		sizes[i] = imgMask[i].size();
#ifdef _DEBUG_COMPOSITOR
		std::cout << corners[i] << std::endl;
		cv::imwrite(cv::format("out/%d.jpg", i), imgWarped[i]);
		cv::imwrite(cv::format("out/%d_mask.jpg", i), imgMask[i]);
#endif
		roiRect[i].x = corners[i].x;
		roiRect[i].y = corners[i].y;
		roiRect[i].width = sizes[i].width;
		roiRect[i].height = sizes[i].height;
	}

	// apply graph cut
	this->applyGraphCut();
	for (int i = 0; i < imgs.size(); i++) {
		cv::Mat K, R;
		cameras[i].K().convertTo(K, CV_32F);
		cameras[i].R.convertTo(R, CV_32F);
		w->warpBackward(imgMaskGraphCut[i], K, R, cv::INTER_NEAREST, 
			cv::BORDER_CONSTANT, imgs[i].size(), imgMaskGraphCutBackward[i]);
		//cv::imwrite(cv::format("mask_gc_%d.png", i), imgMaskGraphCutBackward[i]);
		cv::resize(imgMaskGraphCutBackward[i], imgMaskGraphCutBackward[i], imgs[i].size());
		cv::GaussianBlur(imgMaskGraphCutBackward[i], imgMaskGraphCutBackward[i], cv::Size(25, 25),
			11, 0, cv::BORDER_REPLICATE);
	}

	blender_->prepare(corners, sizes);
	for (int i = 0; i < imgs.size(); i++) {
		// feed to blender
		blender_->feed(imgWarped[i], imgMask[i], corners[i]);
	}
	blender_->blend(result, result_mask);
	result.convertTo(result, CV_8U);
#ifdef _DEBUG_COMPOSITOR
	cv::imwrite("out/result.jpg", result);
#endif

	return 0;
}

/**
@brief get stitching result
@param cv::Mat & result: stitched color image
@param cv::Mat & result_mask: stitched mask image
@return int
*/
int calib::Compositor::getStitchResult(cv::Mat & result, cv::Mat & result_mask) {
	result = this->result;
	result_mask = this->result_mask;
	return 0;
}
