/**
@brief sphere mesh class used for VR content rendering
@author Shane Yuan
@date Jun 30, 2017
*/

#ifndef __SPHERE_MESH_H__
#define __SPHERE_MESH_H__

#include "OBJModelIO.h"

namespace three {

	class SphereMesh : public TexTriangleMesh {
	private:

	public:
		cv::Mat colorImg;
		cv::Mat_<unsigned short> depthMap;
		cv::Mat_<unsigned short> depthMapSmooth;
		cv::Mat_<float> disparityMap;
		cv::Mat_<float> K;
		int meshrows;
		int meshcols;
		cv::Mat_<cv::Vec2f> vertices_2d_;
		cv::Mat_<float> vertexDisparityMap;
		float lambda;

	private:
		/**
		@brief function to generate regular triangle mesh
		@return 
		*/
		int genTriangleMesh();

		/**
		@brief refine depth map
		@return 
		*/
		int refineDepthMap();

	public:
		SphereMesh();
		~SphereMesh();

		/**
		@brief set input for spherical mesh
		@param cv::Mat img: input color image
		@param cv::Mat_<unsigned short> depth: input depth image
		@param cv::Mat K: input intrinsic camera matrix
		@return int
		*/
		int setInput(cv::Mat img, cv::Mat_<unsigned short> depth, 
			cv::Mat_<float> K);

		int debug();
	};
};

#endif