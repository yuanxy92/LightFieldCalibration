/**
@breif C++ head file of GraphCutMask class
calculate stitching mask using graph cut
@author: Shane Yuan
@date: Aug 13, 2017
*/

#ifndef __GRAPHCUT_MASK_H__
#define __GRAPHCUT_MASK_H__

#include <cstdlib>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <opencv2/stitching/detail/seam_finders.hpp>

namespace calib {

	class GraphCutMask {
	private:
		int camNum;
		std::vector<cv::Mat> imgs;
		std::vector<cv::Mat> masks;
		std::vector<cv::Rect> rects;
		std::vector<cv::Mat> gcmasks;
	public:

	private:
		/**
		@brief calculate graphcut mask
		@param float subscale = 1.0f: subscale for calculating 
			graphcut mask
		@return int
		*/
		int calcGraphcutMask(float subscale = 1.0f);

	public:
		GraphCutMask();
		~GraphCutMask();

		/**
		@brief update masks using graphcut
		@param float subscale = 1.0f: subscale for calculating 
			graphcut mask
		@return int
		*/
		int graphcut(std::vector<cv::Mat> & imgs,
			std::vector<cv::Mat> & masks,
			std::vector<cv::Rect> & rects,
			std::vector<cv::Mat> & gcmasks);
	};
};


#endif