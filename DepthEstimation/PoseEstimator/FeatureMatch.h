/**
@brief FeatureMatch.h
C++ head file for feature matching
@author Shane Yuan
@date Jan 30, 2018
*/

#ifndef _ROBUST_STITCHER_FEATURE_MATCH_H_
#define _ROBUST_STITCHER_FEATURE_MATCH_H_ 

// std
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>

// opencv
#include <opencv2/opencv.hpp>

namespace calib {
	/**
	@brief camera parameter struct
	*/
	struct CameraParams {
		CameraParams();
		CameraParams(const CameraParams& other);
		CameraParams& operator =(const CameraParams& other);
		cv::Mat K() const;

		double focal; // Focal length
		double aspect; // Aspect ratio
		double ppx; // Principal point X
		double ppy; // Principal point Y
		cv::Mat R; // Rotation
		cv::Mat t; // Translation
	};

	/**
	@brief struct to save features
	*/
	struct Imagefeature {
		int ind;
		cv::Size imgsize;
		std::vector<cv::KeyPoint> keypt;
		cv::Mat des;
	};

	/**
	@brief struct to save matches
	*/
	typedef std::set<std::pair<int, int> > MatchesSet;

	/**
	@brief struct to save matches information
	*/
	struct Matchesinfo {
		int src_img_idx, dst_img_idx;       // Images indices (optional)
		std::vector<cv::DMatch> matches;
		std::vector<uchar> inliers_mask;    // Geometrically consistent matches mask
		int num_inliers;                    // Number of geometrically consistent matches
		cv::Mat H;                          // Estimated transformation
		double confidence;                  // Confidence two images are from the same panorama
		Matchesinfo();
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

	// class for construct connect matrix all static functions
	class Connection {
	public:
		static int construct(cv::Mat & connection, size_t imgnum);
		static int addConnection(cv::Mat & connection, size_t ind1, size_t ind2);
		static int removeConnection(cv::Mat & connection, size_t ind1, size_t ind2);
		static int countConnections(cv::Mat & connection);
		static std::vector<std::pair<size_t, size_t>> getConnections(cv::Mat & connection);
	};

	// class for feature matching
    class FeatureMatch {
    private:
		// input images and connection matrix
		size_t imgnum;
		std::vector<cv::Mat> imgs;
		cv::Mat connection;
		// feature descriptors
		float matchConf;
		std::vector<Imagefeature> features;
		FeatureExtractorParam featureExtractorParam;
		// feature matchers
		size_t featureNumThresh;
		std::vector<Matchesinfo> matchesInfo;
		// thread used for parallel processing
		size_t maxThNum;
		std::vector<int> thStatus;
		std::vector<std::thread> ths;
		
    public:

    private:
		/**
		@breif compute features for one image
		@param int ind: image index (also thread index)
		@return int
		*/
		void build_feature_thread_(int index);

		/**
		@brief compute surf features points for every image
		@return int
		*/
		int buildFeatures();

		/**
		@breif apply knn feature matching between two image descriptors
		@param int index: thread index
		@param int ind1: input index of the first image
		@param int ind2: input index of the second image
		@param Matchesinfo & matchesinfo: output matches info
		@return int
		*/
		void match_feature_thread_(int index, int ind1, int ind2, Matchesinfo & matchesinfo);

		/**
		@brief match feature descriptors between images with connection
		@return int
		*/
		int buildMatchers();

    public:
        FeatureMatch();
        ~FeatureMatch();

        /**
        @brief init feature matcher
        @param std::vector<cv::Mat> imgs : input vector containing images
        @param cv::Mat connection: input matrix denote the neighboring information
            of the input images
        @return int
        */
        int init(std::vector<cv::Mat> imgs, cv::Mat connection);

		/**
		@brief match function
		*/
		int match();

		/**
		@brief get functions
		*/
		std::vector<Matchesinfo> getMatchesInfo();
		std::vector<Imagefeature> getImageFeatures();

		/**
		@brief functions for visual debugging
		@param size_t ind1: input index of the first image
		@param size_t ind2: input index of the second image
		@return cv::Mat: output image with drawn matching points
		*/
		cv::Mat visualizeMatchingPts(size_t ind1, size_t ind2);

		/**
		@brief debug function
		*/
		int debug();
	};

};

#endif