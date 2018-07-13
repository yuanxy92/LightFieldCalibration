/**
@breif C++ source file of GraphCutMask class
calculate stitching mask using graph cut
@author: Shane Yuan
@date: Aug 13, 2017
*/

#include "GraphCutMask.h"
#include <time.h>

calib::GraphCutMask::GraphCutMask(){}
calib::GraphCutMask::~GraphCutMask(){}

/**
@brief calculate graphcut mask
@param float subscale: subscale for calculating graphcut mask
@return int
*/
int calib::GraphCutMask::calcGraphcutMask(float subscale) {
	// copy data to GPU
	std::vector<cv::UMat> imgsd(camNum);
	std::vector<cv::UMat> masksd(camNum);
	std::vector<cv::Point> corners(camNum);
	for (int i = 0; i < camNum; i++) {
		cv::Mat img;
		cv::Size smallsize = cv::Size(rects[i].width * subscale, rects[i].height * subscale);
		cv::resize(imgs[i], img, smallsize, cv::INTER_LINEAR);
		img.convertTo(imgsd[i], CV_32F);
		cv::Mat mask;
		cv::resize(masks[i], mask, smallsize, cv::INTER_NEAREST);
		mask.convertTo(masksd[i], CV_8U);
		corners[i] = rects[i].tl() * subscale;
	}
	// seamfinder
	cv::Ptr<cv::detail::SeamFinder> seamFinder = cv::makePtr<cv::detail::GraphCutSeamFinder>
		(cv::detail::GraphCutSeamFinderBase::COST_COLOR);
	seamFinder->find(imgsd, corners, masksd);
	// update masks
	for (int i = 0; i < camNum; i++) {
		cv::Mat updateMask, finalMask;
		masksd[i].convertTo(updateMask, CV_8U);
		cv::resize(updateMask, updateMask, masks[i].size(), cv::INTER_NEAREST);
		// dilate to seam mask
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(25, 25));
		cv::dilate(updateMask, updateMask, kernel);
		cv::threshold(updateMask, updateMask, 1, 255, cv::THRESH_BINARY);
		cv::bitwise_and(updateMask, masks[i], gcmasks[i]);
	}
	return 0;
}


/**
@brief update masks using graphcut
@param float subscale = 1.0f: subscale for calculating graphcut mask
@return int
*/
int calib::GraphCutMask::graphcut(std::vector<cv::Mat> & imgs,
	std::vector<cv::Mat> & masks,
	std::vector<cv::Rect> & rects,
	std::vector<cv::Mat> & gcmasks) {
	this->imgs = imgs;
	this->masks = masks;
	this->rects = rects;
	this->camNum = imgs.size();
	this->gcmasks = gcmasks;
	this->calcGraphcutMask(0.4f);
	gcmasks = this->gcmasks;
	return 0;
}