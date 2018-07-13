/**
@brief FeatureMatch.h
C++ source file for feature matching
@author Shane Yuan
@date Jan 30, 2018
*/

#include <numeric>
#include "SysUtil.hpp"
#include "FeatureMatch.h"

#define DEBUG_FEATURE_MATCH

/**
@brief funcions of Connection class
*/
int calib::Connection::construct(cv::Mat & connection, size_t imgnum) {
	connection = cv::Mat::zeros(imgnum, imgnum, CV_8U);
	return 0;
}
int calib::Connection::addConnection(cv::Mat & connection, size_t ind1, size_t ind2) {
	connection.at<uchar>(ind1, ind2) = 1;
	connection.at<uchar>(ind2, ind1) = 1;
	return 0;
}
int calib::Connection::removeConnection(cv::Mat & connection, size_t ind1, size_t ind2) {
	connection.at<uchar>(ind1, ind2) = 0;
	connection.at<uchar>(ind2, ind1) = 0;
	return 0;
}
int calib::Connection::countConnections(cv::Mat & connection) {
	return cv::countNonZero(connection);
}
std::vector<std::pair<size_t, size_t>> calib::Connection::getConnections(cv::Mat & connection) {
	std::vector<std::pair<size_t, size_t>> connections;
	for (size_t i = 0; i < connection.rows - 1; i++) {
		for (size_t j = i + 1; j < connection.cols; j++) {
			if (connection.at<uchar>(i, j) != 0) {
				connections.push_back(std::pair<size_t, size_t>(i, j));
			}
		}
	}
	return connections;
}

/**
@brief funcions of Matchesinfo class
*/
calib::Matchesinfo::Matchesinfo() : src_img_idx(-1), dst_img_idx(-1), num_inliers(0), confidence(0) {}


/**
@brief funcions of FeatureMatch class
*/
calib::FeatureMatch::FeatureMatch() : maxThNum(4), matchConf(0.3f),
	featureNumThresh(6) {}
calib::FeatureMatch::~FeatureMatch() {}

/**
@brief init feature matcher
@param std::vector<cv::Mat> imgs : input vector containing images
@param cv::Mat connection: input matrix denote the neighboring information
of the input images
@return int
*/
int calib::FeatureMatch::init(std::vector<cv::Mat> imgs, cv::Mat connection) {
	this->imgs = imgs;
	this->connection = connection;
	this->imgnum = imgs.size();
	thStatus.resize(imgnum);
	ths.resize(imgnum);
	features.resize(imgnum);
	for (size_t i = 0; i < imgnum; i++) {
		thStatus[i] = 0;
		features[i].imgsize = imgs[i].size();
		features[i].ind = i;
	}
	return 0;
}

/**
@breif compute features for one image
@param int ind: image index
@return int
*/
void calib::FeatureMatch::build_feature_thread_(int index) {
	cv::cuda::SURF_CUDA surf_d;
	cv::cuda::GpuMat des_d;
	cv::cuda::GpuMat keypt_d;
	cv::cuda::GpuMat img_d;
	// set surf parameter
	surf_d.keypointsRatio = 0.1f;
	surf_d.hessianThreshold = featureExtractorParam.hess_thresh;
	surf_d.extended = false;
	surf_d.nOctaves = featureExtractorParam.num_octaves;
	surf_d.nOctaveLayers = featureExtractorParam.num_layers;
	surf_d.upright = false;
	surf_d.hessianThreshold = featureExtractorParam.hess_thresh;
	// upload image
	img_d.upload(imgs[index]);
	cv::cuda::cvtColor(img_d, img_d, CV_BGR2GRAY);
	// extract keypoints
	cv::cuda::GpuMat keypoints_;
	cv::cuda::GpuMat descriptors_;
	surf_d(img_d, cv::cuda::GpuMat(), keypoints_);
	// calculate descriptors
	surf_d.nOctaves = featureExtractorParam.num_octaves_descr;
	surf_d.nOctaveLayers = featureExtractorParam.num_layers_descr;
	surf_d.upright = true;
	surf_d(img_d, cv::cuda::GpuMat(), keypoints_, descriptors_, true);
	surf_d.downloadKeypoints(keypoints_, features[index].keypt);
	descriptors_.download(features[index].des);
	SysUtil::infoOutput(SysUtil::sprintf("Finish feature extractor on image %d .", index));
	thStatus[index] = 0;
}

/**
@brief compute features
@return int
*/
int calib::FeatureMatch::buildFeatures() {
	// start threads for feature extraction
	thStatus.resize(imgnum);
	ths.resize(imgnum);
	for (size_t i = 0; i < imgnum; i++) {
		thStatus[i] = 0;
	}
	int activeThNum = std::accumulate(thStatus.begin(), thStatus.end(), 0);
	int index = 0;
	for (;;) {
		if (activeThNum < maxThNum) {
			thStatus[index] = 1;
			ths[index] = std::thread(&calib::FeatureMatch::build_feature_thread_, this, index);
#ifdef DEBUG_FEATURE_MATCH
			SysUtil::infoOutput(SysUtil::sprintf("Apply feature extractor on image %d ...", index));
#endif
			index++;
			if (index >= imgnum) {
				break;
			}
		}
		else {
			SysUtil::sleep(200);
		}
		activeThNum = std::accumulate(thStatus.begin(), thStatus.end(), 0);
	}
	// exit thread
	for (size_t i = 0; i < imgnum; i++) {
		ths[i].join();
	}
	return 0;
}

/**
@breif apply knn feature matching between two image descriptors
@param int ind1: index of the first image
@param int ind2: index of the second image
@return int
*/
void calib::FeatureMatch::match_feature_thread_(int index, int ind1, int ind2, 
	Matchesinfo & matchesinfo) {
	cv::cuda::GpuMat des1, des2;
	// upload descriptors
	des1.upload(features[ind1].des);
	des2.upload(features[ind2].des);
	// init l1 descriptor matcher
	cv::Ptr<cv::cuda::DescriptorMatcher> matcher = 
		cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_L1);
	// init variables for matching
	MatchesSet matches;
	std::vector< std::vector<cv::DMatch> > pair_matches;
	// find 1->2 matches
	pair_matches.clear();
	matcher->knnMatch(des1, des2, pair_matches, 2);
	for (size_t i = 0; i < pair_matches.size(); ++i) {
		if (pair_matches[i].size() < 2)
			continue;
		const cv::DMatch& m0 = pair_matches[i][0];
		const cv::DMatch& m1 = pair_matches[i][1];
		if (m0.distance < (1.f - matchConf) * m1.distance) {
			matchesinfo.matches.push_back(m0);
			matches.insert(std::make_pair(m0.queryIdx, m0.trainIdx));
		}
	}
	// find 2->1 matches
	pair_matches.clear();
	matcher->knnMatch(des2, des1, pair_matches, 2);
	for (size_t i = 0; i < pair_matches.size(); ++i) {
		if (pair_matches[i].size() < 2)
			continue;
		const cv::DMatch& m0 = pair_matches[i][0];
		const cv::DMatch& m1 = pair_matches[i][1];
		if (m0.distance < (1.f - matchConf) * m1.distance)
			if (matches.find(std::make_pair(m0.trainIdx, m0.queryIdx)) == matches.end())
				matchesinfo.matches.push_back(cv::DMatch(m0.trainIdx, m0.queryIdx, m0.distance));
	}
	// reset thread status
	matchesinfo.src_img_idx = ind1;
	matchesinfo.dst_img_idx = ind2;

	// compute essential matrix (homography)
	// Check if it makes sense to find homography
	if (matchesinfo.matches.size() < featureNumThresh) {
		SysUtil::infoOutput(SysUtil::sprintf("Image pair"\
			" <%u, %u> does not have enough number of feature points ...", ind1, ind2));
		thStatus[index] = 0;
		return;
	}
	// Construct point-point correspondences for homography estimation
	cv::Mat src_points(1, static_cast<int>(matchesinfo.matches.size()), CV_32FC2);
	cv::Mat dst_points(1, static_cast<int>(matchesinfo.matches.size()), CV_32FC2);
	for (size_t i = 0; i < matchesinfo.matches.size(); ++i) {
		const cv::DMatch& m = matchesinfo.matches[i];
		cv::Point2f p = features[ind1].keypt[m.queryIdx].pt;
		p.x -= features[ind1].imgsize.width * 0.5f;
		p.y -= features[ind1].imgsize.height * 0.5f;
		src_points.at<cv::Point2f>(0, static_cast<int>(i)) = p;
		p = features[ind2].keypt[m.trainIdx].pt;
		p.x -= features[ind2].imgsize.width * 0.5f;
		p.y -= features[ind2].imgsize.height * 0.5f;
		dst_points.at<cv::Point2f>(0, static_cast<int>(i)) = p;
	}
	// Find pair-wise motion
	matchesinfo.H = findHomography(src_points, dst_points, matchesinfo.inliers_mask, cv::RANSAC);
	if (matchesinfo.H.empty() || std::abs(cv::determinant(matchesinfo.H))
		< std::numeric_limits<double>::epsilon()) {
		SysUtil::infoOutput(SysUtil::sprintf("Image pair"\
			" <%u, %u> findHomography failed ...", ind1, ind2));
		thStatus[index] = 0;
		return;
	}
	// Find number of inliers
	matchesinfo.num_inliers = 0;
	for (size_t i = 0; i < matchesinfo.inliers_mask.size(); ++i)
		if (matchesinfo.inliers_mask[i])
			matchesinfo.num_inliers++;
	// These coeffs are from paper M. Brown and D. Lowe. "Automatic Panoramic Image Stitching
	// using Invariant Features"
	matchesinfo.confidence = matchesinfo.num_inliers / (8 + 0.3 * matchesinfo.matches.size());

	//// Set zero confidence to remove matches between too close images, as they don't provide
	//// additional information anyway. The threshold was set experimentally.
	//matchesinfo.confidence = matchesinfo.confidence > 3. ? 0. : matchesinfo.confidence;

	// Check if we should try to refine motion
	if (matchesinfo.num_inliers < featureNumThresh) {
		// Construct point-point correspondences for inliers only
		src_points.create(1, matchesinfo.num_inliers, CV_32FC2);
		dst_points.create(1, matchesinfo.num_inliers, CV_32FC2);
		int inlier_idx = 0;
		for (size_t i = 0; i < matchesinfo.matches.size(); ++i) {
			if (!matchesinfo.inliers_mask[i])
				continue;
			const cv::DMatch& m = matchesinfo.matches[i];
			cv::Point2f p = features[ind1].keypt[m.queryIdx].pt;
			p.x -= features[ind1].imgsize.width * 0.5f;
			p.y -= features[ind1].imgsize.height * 0.5f;
			src_points.at<cv::Point2f>(0, inlier_idx) = p;
			p = features[ind2].keypt[m.trainIdx].pt;
			p.x -= features[ind2].imgsize.width * 0.5f;
			p.y -= features[ind2].imgsize.height * 0.5f;
			dst_points.at<cv::Point2f>(0, inlier_idx) = p;
			inlier_idx++;
		}
		// Rerun motion estimation on inliers only
		matchesinfo.H = findHomography(src_points, dst_points, cv::RANSAC);
	}

	SysUtil::infoOutput(SysUtil::sprintf("Finish feature matcher on image pair"\
		" <%u, %u> ...", ind1, ind2));
	thStatus[index] = 0;
}

/**
@brief match feature descriptors between images with connection
@return int
*/
int calib::FeatureMatch::buildMatchers() {
	// get connection pairs
	std::vector<std::pair<size_t, size_t>> pairs = Connection::getConnections(connection);
	size_t connectionNum = pairs.size();
	matchesInfo.resize(connectionNum);
	// init threads
	thStatus.resize(connectionNum);
	ths.resize(connectionNum);
	for (size_t i = 0; i < connectionNum; i++) {
		thStatus[i] = 0;
	}
	int activeThNum = std::accumulate(thStatus.begin(), thStatus.end(), 0);
	int index = 0;
	for (;;) {
		if (activeThNum < maxThNum) {
			thStatus[index] = 1;
			ths[index] = std::thread(&calib::FeatureMatch::match_feature_thread_, this, 
				index, pairs[index].first, pairs[index].second, std::ref(matchesInfo[index]));
#ifdef DEBUG_FEATURE_MATCH
			SysUtil::infoOutput(SysUtil::sprintf("Apply feature matcher on image pair"\
				" <%u, %u> ...", pairs[index].first, pairs[index].second));
#endif
			index++;
			if (index >= connectionNum) {
				break;
			}
		}
		else {
			SysUtil::sleep(200);
		}
		activeThNum = std::accumulate(thStatus.begin(), thStatus.end(), 0);
	}
	// exit thread
	for (size_t i = 0; i < connectionNum; i++) {
		ths[i].join();
	}
	return 0;
}

/**
@brief function for debugging
*/
int calib::FeatureMatch::match() {
	this->buildFeatures();
	this->buildMatchers();
	return 0;
}

/**
@brief get functions
*/
std::vector<calib::Matchesinfo> calib::FeatureMatch::getMatchesInfo() {
	return matchesInfo;
}
std::vector<calib::Imagefeature> calib::FeatureMatch::getImageFeatures() {
	return features;
}

/**
@brief functions for visual debugging
@param size_t ind1: input index of the first image
@param size_t ind2: input index of the second image
@return cv::Mat: output image with drawn matching points
*/
cv::Mat calib::FeatureMatch::visualizeMatchingPts(size_t ind1, size_t ind2) {
	cv::Mat result;
	// make result image
	int width = imgs[ind1].cols * 2;
	int height = imgs[ind1].rows;
	result.create(height, width, CV_8UC3);
	cv::Rect rect(0, 0, imgs[ind1].cols, height);
	imgs[ind1].copyTo(result(rect));
	rect.x += imgs[ind1].cols;  
	imgs[ind2].copyTo(result(rect));
	// find matches info index
	int infoInd = -1;
	std::vector<std::pair<size_t, size_t>> pairs = Connection::getConnections(connection);
	for (size_t i = 0; i < pairs.size(); i ++) {
		if ((pairs[i].first == ind1 && pairs[i].second == ind2) ||
			(pairs[i].first == ind2 && pairs[i].second == ind1)) {
			infoInd = i;
			break;
		}
	}
	// draw matching points
	cv::RNG rng(12345);
    int r = 3;
	if (infoInd != -1) {
		for (size_t kind = 0; kind < matchesInfo[infoInd].matches.size(); kind++) {
			if (matchesInfo[infoInd].inliers_mask[kind]) {
				cv::Scalar color = cv::Scalar(rng.uniform(0, 255), 
					rng.uniform(0, 255), rng.uniform(0, 255));
				const cv::DMatch& m = matchesInfo[infoInd].matches[kind];
				cv::Point2f p1 = features[ind1].keypt[m.queryIdx].pt;
				cv::Point2f p2 = features[ind2].keypt[m.trainIdx].pt;
				p2.x += imgs[ind1].cols;
				cv::circle(result, p1, r, color, -1, cv::LINE_8, 0);
				cv::circle(result, p2, r, color, -1, cv::LINE_8, 0);
				//cv::line(result, p1, p2, color, 5, cv::LINE_8, 0);
			}
		}
	}
	return result;
}

/**
@brief debug function
*/
int calib::FeatureMatch::debug() {
	// get connection pairs
	std::vector<std::pair<size_t, size_t>> pairs = Connection::getConnections(connection);
	size_t connectionNum = pairs.size();
	for (size_t i = 0; i < connectionNum; i ++) {
		cv::Mat result = calib::FeatureMatch::visualizeMatchingPts(pairs[i].first, pairs[i].second);
		cv::imwrite(cv::format("matching_points_%02d_%02d.jpg", pairs[i].first, pairs[i].second),
			result);
	}	
	return 0;
}