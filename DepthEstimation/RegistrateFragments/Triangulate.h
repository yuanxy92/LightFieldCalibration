/**
@brief triangulate image into triangle mesh
@author Shane Yuan
@date May 30, 2018
*/

#ifndef _TRIANGULATION_H_
#define _TRIANGULATION_H_

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>

// opencv
#include <opencv2/opencv.hpp>

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

// triangulation
#include <ttl/halfedge/HeTriang.h>
#include <ttl/halfedge/HeDart.h>
#include <ttl/halfedge/HeTraits.h>

class Triangulate {
private:
	cv::Mat img;
	cv::Mat corners;
	cv::Mat_<int> pointNumMat;
	cv::Size quadSize;
	std::vector<cv::Point> vertices;
	std::list<hed::Edge*> edges;
	cv::Mat visual;
public:

private:
	/**
	@brief visualize corners in image
	@return int
	*/
	int visualize();

	/**
	@brief add points in uniform/smooth region
	@return int
	*/
	std::vector<cv::Point> addPoints(cv::Mat & corners);

	/**
	@brief check if 3 points are counter clockwise or clockwise
	@param cv::Point[3] input : input 3 points
	@return true/false : ccw/cw
	*/
	bool checkCCW(cv::Point input[3]);

	/**
	@brief check if 3 points are counter clockwise or clockwise
	@param Eigen::Vector3d pt1: first triangle point
	@param Eigen::Vector3d pt2: second triangle point
	@param Eigen::Vector3d pt3: third triangle point
	@return float: length of longest edge
	*/
	float computeLongestEdge(Eigen::Vector3d pt1,
		Eigen::Vector3d pt2, Eigen::Vector3d pt3);

public:
	Triangulate();
	~Triangulate();

	/**
	@brief triangulate image into tri-mesh
	@param cv::Mat img: input color image
	@param cv::Mat_<cv::Vec3f> spherePoint3D: input 3d point image
	@param std::vector<Eigen::Vector3d> vertices: output vertices
	@param std::vector<Eigen::Vector3d> vertex_colors_: output vertice colors
	@param std::vector<Eigen::Vector2d> vertex_uvs_: output vertex uvs
	@param std::vector<Eigen::Vector3i> triangles_: output triangles
	@return int
	*/
	int triangulate(cv::Mat img, 
		cv::Mat_<cv::Vec3f> spherePoint3D ,
		std::vector<Eigen::Vector3d> & vertex_,
		std::vector<Eigen::Vector3d> & vertex_colors_,
		std::vector<Eigen::Vector2d> & vertex_uvs_,
		std::vector<Eigen::Vector3i> & triangles_
		);
};

#endif