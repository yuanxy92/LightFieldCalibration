/**
@brief class for *.obj file IO 
@author Shane Yuan
@date May 26, 2018
*/

#ifndef __OBJ_MODEL_IO_H__
#define __OBJ_MODEL_IO_H__

#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <string>

// opencv
#include <opencv2/opencv.hpp>

// open3d
#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>

namespace three {

	class TexTriangleMesh : public TriangleMesh {
	private:

	public:
		std::vector<Eigen::Vector2d> vertex_uvs_;
		cv::Mat texture_;
	private:

	public:
		TexTriangleMesh();
		~TexTriangleMesh();

		TexTriangleMesh &operator+=(const TexTriangleMesh &mesh);
		TexTriangleMesh operator+(const TexTriangleMesh &mesh) const;

		int generateTexTriangleMesh(cv::Mat_<cv::Vec3f> spherePoint3D,
			cv::Mat panorama);
	};


	class OBJModelIO {
	private:

	public:

	private:

	public:
		OBJModelIO();
		~OBJModelIO();

		/**
		@brief load obj file
		@param std::string filename: input obj filename
		@param std::shared_ptr<TexTriangleMesh> & texTriMesh: output triangle mesh
			with texture
		@return int
		*/
		static int load(std::string filename, std::shared_ptr<TexTriangleMesh> & texTriMesh);

		/**
		@brief save obj model with texture
		@param std::string dir: output dir to save the obj file
		@param std::string filename: output obj filename
		@param std::shared_ptr<TexTriangleMesh> texTriMesh: triangle mesh with texture 
		@return int
		*/
		static int save(std::string dir, 
			std::string filename, 
			std::shared_ptr<TexTriangleMesh> texTriMesh);
	};
};


#endif