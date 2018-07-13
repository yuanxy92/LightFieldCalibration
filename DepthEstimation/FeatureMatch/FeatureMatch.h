/**
@brief class for feature matching
@author Shane Yuan
@date May 22, 2018
*/

#ifndef __FEATURE_MATCH_H__

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <chrono>

#include <opencv2/opencv.hpp>

namespace fm {

	struct MatchingPoints {
		cv::Point2f leftPt;
		cv::Point2f rightPt;
		float confidence;
	};

	struct FeatureExtractorParam {
		// basic parameter of feature extractor
		double hess_thresh;
		int num_octaves;
		int num_layers;
		int num_octaves_descr;
		int num_layers_descr;
		FeatureExtractorParam() : hess_thresh(300.0), num_octaves(3),
			num_layers(4), num_octaves_descr(3), num_layers_descr(4) {}
	};

	typedef std::set<std::pair<int, int> > MatchesSet;

	class FeatureMatch {
	private:

	public:

	private:
		/**
		@brief visualize matching points
		@param cv::Mat leftImg: left image
		@param cv::Mat rightImg: right image
		@param 
		@return cv::Mat visualization result
		*/
		static cv::Mat visualizeMatchingPoints(cv::Mat leftImg, cv::Mat rightImg,
			std::vector<MatchingPoints> pts);

		/**
		@brief compute feature points
		@param cv::Mat image: input images
		@param cv::Mat keypt: output key points
		@param cv::Mat descriptor: output descriptor
		@return int
		*/
		static int computeFeaturePoints(cv::Mat image, std::vector<cv::KeyPoint> & keypt,
			cv::Mat & descriptor);

		/**
		@brief function to apply surf feature matching
		@param cv::Mat leftBlock: image block of left camera
		@param cv::Mat rightBlock: image block of right camera
		@return std::vector<MatchingPoints>: return matching points
		*/
		static std::vector<MatchingPoints> applySurfMatching(cv::Mat leftBlock,
			cv::Mat rightBlock);

		/**
		@brief local ransac
		@param std::vector<MatchingPoints> & matchingPts: 
		@param int row: rows of mesh
		@param int col: cols of mesh
		@param int imgWidth: width of image
		@param int imgHeight: height of image
		@param float thresh: ransac thresh
		*/
		static int localRansac(std::vector<MatchingPoints> & matchingPts,
			int rows, int cols, int imgWidth, int imgHeight, float thresh);

	public:
		FeatureMatch();
		~FeatureMatch();

		/**
		@brief function to construct matching points
		@param cv::Mat leftImg: image of left camera
		@param cv::Mat rightImg: image of right camera
		@param int minDisparity: minimum disparity
		@param int maxDisparity: maximum disparity
		@return std::vector<MatchingPoints>
		*/
		static std::vector<MatchingPoints> constructMatchingPoints(cv::Mat leftImg,
			cv::Mat rightImg, int minDisparity, int maxDisparity);

		/**
		@brief function to construct matching points
		@param cv::Mat leftImg: image of left camera
		@param cv::Mat rightImg: image of right camera
		@param int minDisparity: minimum disparity
		@param int maxDisparity: maximum disparity
		@return std::vector<MatchingPoints>
		*/
		static std::vector<MatchingPoints> constructMatchingPointsZNCC(cv::Mat leftImg,
			cv::Mat rightImg, int minDisparity, int maxDisparity);

	};

};


#endif // !__FEATURE_MATCH_H__
