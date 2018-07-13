/**
@brief class for feature matching
@author Shane Yuan
@date May 22, 2018
*/

#include "FeatureMatch.h"

//#define _DEBUG_FEATURE_MATCH

namespace fm {

	FeatureMatch::FeatureMatch() {}
	FeatureMatch::~FeatureMatch() {}

	/**
	@brief visualize matching points
	@param cv::Mat leftImg: left image
	@param cv::Mat rightImg: right image
	@return cv::Mat visualization result
	*/
	cv::Mat FeatureMatch::visualizeMatchingPoints(cv::Mat leftImg, cv::Mat rightImg,
		std::vector<MatchingPoints> pts) {
		cv::Mat visual(leftImg.rows, leftImg.cols + rightImg.cols, CV_8UC3);
		cv::Rect rect(0, 0, leftImg.cols, leftImg.rows);
		leftImg.copyTo(visual(rect));
		rect.x += leftImg.cols;
		rect.width = rightImg.cols;
		rightImg.copyTo(visual(rect));
		cv::RNG rng(12345);
		int r = 3;
		for (size_t i = 0; i < pts.size(); i++) {
			cv::Point2f leftp = pts[i].leftPt;
			cv::Point2f rightp = pts[i].rightPt;
			rightp.x += leftImg.cols;
			cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			cv::circle(visual, leftp, r, color, -1, 8, 0);
			cv::circle(visual, rightp, r, color, -1, 8, 0);
			//cv::line(visual, leftp, rightp, color, 1, 8, 0);
		}
		return visual;
	}

	/**
	@brief compute feature points
	@param cv::Mat image: input images
	@param cv::Mat & keypt: output key points
	@param cv::Mat & descriptor: output descriptor
	@return int
	*/
	int FeatureMatch::computeFeaturePoints(cv::Mat image, std::vector<cv::KeyPoint> & keypt,
		cv::Mat & descriptor) {
		cv::cuda::SURF_CUDA surf_d;
		cv::cuda::GpuMat des_d;
		cv::cuda::GpuMat keypt_d;
		cv::cuda::GpuMat img_d;
		FeatureExtractorParam featureExtractorParam;
		// set surf parameter
		surf_d.keypointsRatio = 0.1f;
		surf_d.hessianThreshold = featureExtractorParam.hess_thresh;
		surf_d.extended = false;
		surf_d.nOctaves = featureExtractorParam.num_octaves;
		surf_d.nOctaveLayers = featureExtractorParam.num_layers;
		surf_d.upright = false;
		surf_d.hessianThreshold = featureExtractorParam.hess_thresh;
		// upload image
		img_d.upload(image);
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
		surf_d.downloadKeypoints(keypoints_, keypt);
		descriptors_.download(descriptor);
		return 0;
	}

	/**
	@brief function to apply surf feature matching
	@param cv::Mat leftBlock: image block of left camera
	@param cv::Mat rightBlock: image block of right camera
	@return std::vector<MatchingPoints>: return matching points
	*/
	std::vector<MatchingPoints> FeatureMatch::applySurfMatching(cv::Mat leftBlock,
		cv::Mat rightBlock) {
		// compute keypoints and descriptors
		std::vector<cv::KeyPoint> leftKeypt;
		std::vector<cv::KeyPoint> rightKeypt;
		cv::Mat leftDes, rightDes;
		FeatureMatch::computeFeaturePoints(leftBlock, leftKeypt, leftDes);
		FeatureMatch::computeFeaturePoints(rightBlock, rightKeypt, rightDes);
		// match feature points
		cv::cuda::GpuMat leftDes_d, rightDes_d;
		// upload descriptors
		float matchConf = 0.3f;
		leftDes_d.upload(leftDes);
		rightDes_d.upload(rightDes);
		// init l1 descriptor matcher
		cv::Ptr<cv::cuda::DescriptorMatcher> matcher =
			cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_L1);
		// init variables for matching
		std::vector< std::vector<cv::DMatch> > pair_matches;
		std::vector<cv::DMatch> matches;
		MatchesSet matches_test;
		// find 1->2 matches
		pair_matches.clear();
		matcher->knnMatch(leftDes_d, rightDes_d, pair_matches, 2);
		for (size_t i = 0; i < pair_matches.size(); ++i) {
			if (pair_matches[i].size() < 2)
				continue;
			const cv::DMatch& m0 = pair_matches[i][0];
			const cv::DMatch& m1 = pair_matches[i][1];
			if (m0.distance < (1.f - matchConf) * m1.distance) {
				matches_test.insert(std::make_pair(m0.queryIdx, m0.trainIdx));
				matches.push_back(m0);
			}
		}
		// find 2->1 matches
		pair_matches.clear();
		matcher->knnMatch(rightDes_d, leftDes_d, pair_matches, 2);
		for (size_t i = 0; i < pair_matches.size(); ++i) {
			if (pair_matches[i].size() < 2)
				continue;
			const cv::DMatch& m0 = pair_matches[i][0];
			const cv::DMatch& m1 = pair_matches[i][1];
			if (m0.distance < (1.f - matchConf) * m1.distance)
				if (matches_test.find(std::make_pair(m0.trainIdx, m0.queryIdx)) == matches_test.end())
					matches.push_back(cv::DMatch(m0.trainIdx, m0.queryIdx, m0.distance));
		}
		// ransac
		// Construct point-point correspondences for homography estimation
		std::vector<cv::Point2f> leftPoints;
		std::vector<cv::Point2f> rightPoints;
		std::vector<int> indices;
		for (size_t i = 0; i < matches.size(); ++i) {
			const cv::DMatch& m = matches[i];
			cv::Point2f leftp = leftKeypt[m.queryIdx].pt;
			cv::Point2f rightp = rightKeypt[m.trainIdx].pt;
			if (abs(leftp.y - rightp.y) > 4) {
				continue;
			}
			leftPoints.push_back(leftp);
			rightPoints.push_back(rightp);
			indices.push_back(i);
		}
		// Find pair-wise motion
		cv::Mat mask;
		std::vector<MatchingPoints> pts;
		if (leftPoints.size() > 4) {
			findHomography(leftPoints, rightPoints, mask, cv::RANSAC, 2);
			// return result
			for (size_t i = 0; i < leftPoints.size(); i++) {
				MatchingPoints pt;
				if (mask.at<uchar>(i, 0) > 0) {
					pt.leftPt = leftPoints[i];
					pt.rightPt = rightPoints[i];
					pt.confidence = 1 / matches[indices[i]].distance;
					pts.push_back(pt);
				}
			}
		}
		else {
			pts.clear();
		}
		return pts;
	}
	
	/**
	@brief function to construct matching points
	@param cv::Mat leftImg: image of left camera
	@param cv::Mat rightImg: image of right camera
	@param int minDisparity: minimum disparity
	@param int maxDisparity: maximum disparity
	@return std::vector<MatchingPoints>
	*/
	std::vector<MatchingPoints> FeatureMatch::constructMatchingPoints(cv::Mat leftImg,
		cv::Mat rightImg, int minDisparity, int maxDisparity) {
		// get image size
		int width, height;
		width = leftImg.cols;
		height = leftImg.rows;
		int blockRows = 6;
		int blockCols = 6;
		int blockHeight = height / blockRows;
		int blockWidth = width / blockCols;
		// return matching points
		std::vector<MatchingPoints> matchingPts;
		// divide image into blocks
		cv::Rect leftRect, rightRect;
		for (int row = 0; row < blockRows; row++) {
			for (int col = 0; col < blockRows; col++) {
				// calculate left roi rect
				leftRect.x = col * blockWidth;
				leftRect.y = row * blockHeight;
				leftRect.width = blockWidth;
				leftRect.height = blockHeight;
				// calculate right roi rect
				rightRect.x = std::max<int>(leftRect.x - maxDisparity, 0);
				rightRect.y = leftRect.y;
				rightRect.width = std::min<int>(leftRect.width + maxDisparity,
					width - rightRect.x + 1);
				rightRect.height = leftRect.height;
				// apply surf feature matching
				cv::Mat leftBlock, rightBlock;
				leftBlock = leftImg(leftRect);
				rightBlock = rightImg(rightRect);
				std::vector<MatchingPoints> pts;
				pts = FeatureMatch::applySurfMatching(leftBlock, rightBlock);
#ifdef _DEBUG_FEATURE_MATCH
				cv::Mat visual = visualizeMatchingPoints(leftBlock, rightBlock, pts);
#endif
				// insert to final 
				for (size_t k = 0; k < pts.size(); k++) {
					pts[k].leftPt.x += leftRect.x;
					pts[k].leftPt.y += leftRect.y;
					pts[k].rightPt.x += rightRect.x;
					pts[k].rightPt.y += rightRect.y;
					if (pts[k].leftPt.x - pts[k].rightPt.x > maxDisparity)
						pts[k].confidence = 0;
				}
				matchingPts.insert(matchingPts.end(), pts.begin(), pts.end());
			}
		}
#ifdef _DEBUG_FEATURE_MATCH
		cv::Mat visual = visualizeMatchingPoints(leftImg, rightImg, matchingPts);
#endif
		return matchingPts;
	}

	/**
	@brief local ransac
	@param std::vector<MatchingPoints> & matchingPts: 
	@param int row: rows of mesh
	@param int col: cols of mesh
	@param int imgWidth: width of image
	@param int imgHeight: height of image
	@param float thresh: ransac thresh
	*/
	int FeatureMatch::localRansac(std::vector<MatchingPoints> & matchingPts,
		int rows, int cols, int imgWidth, int imgHeight, float thresh) {
		std::vector<std::vector<cv::Point2f>> localKp1;
		std::vector<std::vector<cv::Point2f>> localKp2;
		std::vector<std::vector<float>> localConfidence;
		localKp1.resize(rows * cols);
		localKp2.resize(rows * cols);
		localConfidence.resize(rows * cols);
		// divide
		int lens = matchingPts.size();
		float quadWidth = imgWidth / cols;
		float quadHeight = imgHeight / rows;
		for (int ind = 0; ind < lens; ind++) {
			int i = static_cast<int>(matchingPts[ind].leftPt.y / quadHeight);
			int j = static_cast<int>(matchingPts[ind].leftPt.x / quadWidth);
			int dist = std::abs(matchingPts[ind].leftPt.y - matchingPts[ind].rightPt.y) + 
				std::abs(matchingPts[ind].leftPt.x - matchingPts[ind].rightPt.x);
			if (dist < 1000) {
				int kpInd = i * cols + j;
				localKp1[kpInd].push_back(matchingPts[ind].leftPt);
				localKp2[kpInd].push_back(matchingPts[ind].rightPt);
				localConfidence[kpInd].push_back(matchingPts[ind].confidence);
			}
		}
		// local ransac
		matchingPts.clear();
		for (int i = 0; i < rows * cols; i++) {
			cv::Mat mask;
			if (localKp1[i].size() >= 4) {
				cv::Mat H = findHomography(localKp1[i], localKp2[i], cv::RANSAC, thresh, mask);
				for (int j = 0; j < mask.rows; j++) {
					if (mask.at<uchar>(j, 0) == 1) {
						MatchingPoints mpt;
						mpt.leftPt = localKp1[i][j];
						mpt.rightPt = localKp2[i][j];
						mpt.confidence = localConfidence[i][j];
						matchingPts.push_back(mpt);
					}
				}
			}
		}
		return 0;
	}

	/**
	@brief function to construct matching points
	@param cv::Mat leftImg: image of left camera
	@param cv::Mat rightImg: image of right camera
	@param int minDisparity: minimum disparity
	@param int maxDisparity: maximum disparity
	@return std::vector<MatchingPoints>
	*/
	std::vector<MatchingPoints> FeatureMatch::constructMatchingPointsZNCC(cv::Mat leftImg,
		cv::Mat rightImg, int minDisparity, int maxDisparity) {
		// return matching points
		std::vector<MatchingPoints> matchingPts;
		// find keypoints
		cv::Mat grayImg, corners;
		cv::cvtColor(leftImg, grayImg, cv::COLOR_BGR2GRAY);
		cv::goodFeaturesToTrack(grayImg, corners, 30000, 0.005, 10);
		// use ZNCC to find matching points
		int patchSize = 64;
		int verticalShift = 15;
		for (size_t i = 0; i < corners.rows; i++) {
			cv::Point2f pt = corners.at<cv::Point2f>(i, 0);
			cv::Rect patchRect(pt.x - patchSize / 2, pt.y - patchSize / 2, patchSize, patchSize);
			cv::Rect searchRect(pt.x - patchSize / 2 - maxDisparity, pt.y - patchSize / 2 - verticalShift,
				patchSize + maxDisparity, patchSize + verticalShift);
			if ((patchRect & cv::Rect(0, 0, leftImg.size().width, leftImg.size().height)) == patchRect
				&& (searchRect & cv::Rect(0, 0, leftImg.size().width, leftImg.size().height)) == searchRect) {
				cv::Mat result;

				cv::Mat templ = leftImg(patchRect);
				cv::Mat search = rightImg(searchRect);

				cv::matchTemplate(search, templ, result, cv::TM_CCOEFF_NORMED);
				double maxVal, minVal;
				cv::Point maxLoc, minLoc;
				cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
				cv::Point ptRight = cv::Point(searchRect.x + maxLoc.x + patchSize / 2,
					searchRect.y + maxLoc.y + patchSize / 2);
				MatchingPoints mpt;
				mpt.leftPt = pt;
				mpt.rightPt = ptRight;
				mpt.confidence = maxVal;
				matchingPts.push_back(mpt);
			}
		}
		// local ransac
		FeatureMatch::localRansac(matchingPts, 6, 6, leftImg.size().width,
			leftImg.size().height, 6.0f);

#ifdef _DEBUG_FEATURE_MATCH
		cv::Mat visual = visualizeMatchingPoints(leftImg, rightImg, matchingPts);
		cv::imwrite("feature_match.png", visual);
#endif
		return matchingPts;
	}

};