/**
* @brief main function of reference camera stitcher
* @author Shane Yuan
* @date May 24, 2017
*/

#include <stdio.h>
#include <cstdlib>
#include <opencv2/opencv.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "FeatureMatch.h"
#include "CameraParamEstimator.h"
#include "Compositor.h"

int main(int argc, char* argv[]) {

	google::InitGoogleLogging(argv[0]);

	std::string datapath = std::string(argv[1]);
    std::string outpath = std::string(argv[4]);
    std::vector<cv::Mat> imgs;
	std::vector<cv::String> img_files;
	cv::Mat result, result_mask;

	int startInd = atoi(argv[2]);
	int endInd = atoi(argv[3]);
	for (int i = startInd; i <= endInd; i = i++) {
		img_files.push_back(cv::format("%s/%04d_left.jpg", datapath.c_str(), i));
		imgs.push_back(cv::imread(img_files[img_files.size() - 1]));
	}
	int img_num = imgs.size();
	cv::Mat connection;
	connection = cv::Mat::zeros(img_num, img_num, CV_8U);
	for (size_t i = 0; i < imgs.size(); i++) {
		for (size_t j = 0; j < imgs.size(); j++) {
			if (i == j)
				continue;
			connection.at<uchar>(i, j) = 1;
			connection.at<uchar>(j, i) = 1;
		}
	}
	calib::FeatureMatch match;
	match.init(imgs, connection);
	match.match();
	calib::CameraParamEstimator estimator;
	estimator.init(imgs, connection, match.getImageFeatures(), match.getMatchesInfo());
	estimator.estimate();
	calib::Compositor compositor;
	compositor.init(imgs, estimator.getCameraParams());
	compositor.composite();
	compositor.getStitchResult(result, result_mask);

	//output
	estimator.saveCalibrationResult(cv::format("%s/CameraParams_%04d_%04d.txt", outpath.c_str(), startInd, endInd));
	cv::imwrite(cv::format("%s/result.jpg", outpath.c_str()), result);
	cv::imwrite(cv::format("%s/result_mask.jpg", outpath.c_str()), result_mask);
    return 0;    
}