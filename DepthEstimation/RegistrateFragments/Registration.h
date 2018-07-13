/**
@brief warper class, warping function
@author Shane Yuan
@date May 30, 2018
*/

#ifndef _REGISTRATION_H_
#define _REGISTRATION_H_

// stl
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <fstream>

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>

// opencv
#include <opencv2/opencv.hpp>

// open3d
#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>

#include "OBJModelIO.h"

namespace calib {

	struct CameraParam {
		Eigen::Matrix4d intEigen;
		cv::Mat intCV;
		Eigen::Matrix4d extEigen;
		cv::Mat extCV;
	};

	class Registration {
	private:
		// input filenames
		std::string calibname;
		std::vector<std::string> colornames;
		std::vector<std::string> depthnames;
		// camera parameters
		int cameraNum;
		std::vector<CameraParam> cameras;
		std::vector<cv::Mat> colorImgs;
		std::vector<cv::Mat> depthImgs;

		// sphere parameter
		std::vector<cv::Rect> rects;
		std::vector<cv::Mat_<cv::Vec3f>> point3Ds;
		std::vector<cv::Mat_<cv::Vec3f>> spherePoint3Ds;
		std::vector<cv::Mat_<float>> sphereWeightDepthMasks;
		cv::Rect sphereRect;
		cv::Mat_<cv::Vec3f> spherePoint3D;
		cv::Mat_<unsigned short> sphereDepthMap;
		cv::Mat_<cv::Vec2f> sphereAngleMap;
		cv::Point2f tlAngle;
		cv::Point2f brAngle;

		cv::Mat_<float> spherePoint3DMask;
		cv::Mat spherePanorama;
		cv::Mat sphereMask;

		// mesh
		std::shared_ptr<three::TriangleMesh> mesh;
		std::shared_ptr<three::TexTriangleMesh> texmesh;

	public:

	private:
		/*******************************************************************/
		/*                         utility function                        */
		/*******************************************************************/
		/**
		@brief function to inpaint depth images
		@return int
		*/
		static cv::Mat_<cv::Vec3f> inpaint(cv::Mat_<cv::Vec3f> point3DImg);

		/**
		@brief function to generate weighted depth map mask
		@return cv::Mat weightMask
		*/
		static cv::Mat_<float> generateWeightedMask(cv::Size imgsize);

		/**
		@brief function to project point 3d map to 2d distance map
		@return int
		*/
		static cv::Mat_<cv::Vec3f> project3DPointsToDistanceMap(cv::Mat_<cv::Vec3f> points);

		/**
		@brief function to project 2d depth map to 3d point
		@return int
		*/
		int projectDepthMapTo3DPoints(cv::Mat depth, cv::Mat_<cv::Vec3f> & points,
			cv::Mat K, cv::Mat R);

		/*******************************************************************/
		/*                      function for processing                    */
		/*******************************************************************/
		/**
		@brief function to read camera params
		@return int
		*/
		int readCameraParams();

		/**
		@brief funtion to merge point3D maps
		@return int
		*/
		int mergePoint3DImages();

		/**
		@brief project depth map to sphere
		@return int
		*/
		int sphereFusion();

		/**
		@brief project to 3D mesh
		@return int
		*/
		int projectTo3DMesh();

	public:
		Registration();
		~Registration();

		/**
		@brief init registation class
		@param std::string calibname: filename of calibration result
		@param std::vector<std::string> colornames: names of color images
		@param std::vector<std::string> depthnames: names of depth images
		@return int
		*/
		int init(std::string calibname, std::vector<std::string> colornames,
			std::vector<std::string> depthnames);

		/**
		@brief fusion 
		*/
		int registrate();

		/**
		@brief get registrated mesh
		@return std::shared_ptr<three::TexTriangleMesh> texmesh
		*/
		std::shared_ptr<three::TexTriangleMesh> getTexMesh();

		/**
		@brief get panorama depth map (uint16)
		@return cv::Mat: return depth map
		*/
		cv::Mat getDepthMap();

	};
}


#endif