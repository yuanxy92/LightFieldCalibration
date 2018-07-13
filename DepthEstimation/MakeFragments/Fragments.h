/**
@brief class for feature matching
@author Shane Yuan
@date May 26, 2018
*/

#ifndef __GIGA3D_FRAGMENT_H__
#define __GIGA3D_FRAGMENT_H__

// std
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <thread>
#include <chrono>

// opencv
#include <opencv2/opencv.hpp>

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>

// open3d
#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>

#include "OffScreenRender.h"

namespace FragmentsKernel {
	/**
	*/
}

class Fragments {
private:
	// input image names
	std::vector<std::string> colornames;
	std::vector<std::string> depthnames;
	std::string cameraname;
	std::string intname;
	int representInd;

	// images
	std::vector<cv::Mat> colorImgs;
	std::vector<cv::Mat> depthImgs;

	// calculated mesh
	float thresh; // thresh to check if three points are in a triagnle 
	std::vector<std::shared_ptr<three::TriangleMesh>> meshes;
	std::vector<Eigen::Matrix4d> camInts;
	std::vector<Eigen::Matrix4d> camExts;

	std::shared_ptr<gl::OffScreenRender> render;
	cv::Mat refineDepthMap;

public:

private:
	/**
	@brief read camera parameters
	@return int
	*/
	int readCameraParams();
	
	/**
	@brief generate single mesh
	@return int
	*/
	int generateSingleMesh();

	/**
	@brief depth map fusion 
	@return int
	*/
	int fuseDepthMaps();

	/**
	@brief fuse depth map by voting
	@return int
	*/
	int fuseDepthMapByVoting();

public:
	Fragments();
	~Fragments();

	/**
	@brief project image to 3d model
	@return int
	*/
	static int projectImageToModel(std::shared_ptr<three::TriangleMesh> mesh,
		Eigen::Matrix4d intmat, Eigen::Matrix4d extmat,
		cv::Mat colorImg, cv::Mat depthImg, float thresh = 1.0f);

	/**
	@brief project 3d model to image
	@return int
	*/
	static int projectModelToImage(std::shared_ptr<three::TriangleMesh> mesh,
		Eigen::Matrix4d intmat, Eigen::Matrix4d extmat,
		cv::Mat & colorImg, cv::Mat & depthImg, cv::Size imgsize,
		std::shared_ptr<gl::OffScreenRender> render);

	/**
	@brief set input color/depth image and camera poses
	@param std::vector<std::string> colornames: input color image names
	@param std::vector<std::string> depthnames: input depth image names
	@param std::string cameraname: input camera parameter name 
		(include camera focal and extrinsic matrix)
	@param representInd: input index which other depth map will re-project
		to this depth map
	@return int
	*/
	int init(std::vector<std::string> colornames, 
		std::vector<std::string> depthnames,
		std::string cameraname,
		int representInd);

	/**
	@brief fuse all the depth maps into representative frame  
	@return int
	*/
	int fusion();

	/**
	@brief function to get results
	@return int
	*/
	int getResults(cv::Mat& refineDepthMap, cv::Mat& colorImg);


};

#endif // !__GIGA3D_FRAGMENT_H__




