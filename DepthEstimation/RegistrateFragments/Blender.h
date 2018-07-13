/**
@brief blender class fusion depth map to get 
    better 3d model
@author Shane Yuan
@date May 31, 2018
*/

#ifndef _REGISTRATION_BLENDER_H_
#define _REGISTRATION_BLENDER_H_

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>

#include <opencv2/opencv.hpp>

namespace calib {

	class Blender {
	private:
		int actual_num_bands_, num_bands_;
		std::vector<cv::UMat> dst_pyr_laplace_;
		std::vector<cv::UMat> dst_band_weights_;
		cv::Rect dst_roi_final_;
		bool can_use_gpu_;
		int weight_type_; //CV_32F or CV_16S
		cv::UMat dst_, dst_mask_;
		cv::Rect dst_roi_;
		float weight_eps;
	public:

	private:
		int numBands() const;
		void setNumBands(int val);
		void createLaplacePyr(cv::InputArray img, int num_levels, 
			std::vector<cv::UMat> &pyr);
		cv::Rect resultRoi(const std::vector<cv::Point> &corners,
			const std::vector<cv::Size> &sizes);
		void normalizeUsingWeightMap(cv::InputArray _weight, 
			cv::InputOutputArray _src);
		void restoreImageFromLaplacePyr(std::vector<cv::UMat> &pyr);
	public:
		Blender();
		~Blender();

		void prepare(cv::Rect dst_roi);
		void prepare(const std::vector<cv::Point> &corners, 
			const std::vector<cv::Size> &sizes);
		void feed(cv::InputArray img, cv::InputArray mask, cv::Point tl);
		void blend(cv::InputOutputArray dst, cv::InputOutputArray dst_mask);
	};

};

#endif