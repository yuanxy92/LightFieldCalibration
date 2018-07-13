/**
@brief file OpenGLImageWarper.h
warp image using 2D mesh grid
@author Shane Yuan
@date Mar 14, 2018
*/

#ifndef __OPENGL_IMAGE_WARPER__
#define __OPENGL_IMAGE_WARPER__

// include stl
#include <memory>
#include <cstdlib>
// include GLAD
#include <GL/glew.h>
// include GLFW
#include <GLFW/glfw3.h>
// include GLM
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// include shader and texture loader
#include "shader.hpp"

// include opencv
#include <opencv2/opencv.hpp>

// eigen
#include <Eigen/Core>
#include <Eigen/Dense>

// open3d
#include <Core/Core.h>
#include <IO/IO.h>
#include <Visualization/Visualization.h>


namespace gl {
	// opengl image warper class
	class OffScreenRender {
	private:
		// window
		GLFWwindow* window;
		// program ID
		GLuint programID;
		// frame buffer for color and depth
		GLuint frameBufferID;
		GLuint frameBufferDepthID;
		// texture ID and size
		GLuint outputTextureID;
		cv::Size outputSize;
		// vertex array ID
		GLuint vertexArrayID;
		// vertex ID
		GLuint vertexID;
		// color ID
		GLuint colorID;
		// near and far
		int znear;
		int zfar;

	public:

	private:
		/**
		@brief error callback function
		@param int error: error id
		@param const char* description: error description
		@return int
		*/
		static void error_callback(int error, const char* description);

		/**
		@brief generate OpenGL projection matrix from intrinsic matrix
		@param glm::mat4& proj: output opengl projection matrix, column-major
		@param float fx: input focal length in x direction
		@param float fy: input focal length in y direction
		@param float width: input width of rendered image
		@param float height: input height of rendered image
		@param float cx: input x coordinate of principal point
		@param float cy: input y coordinate of principal point
		@param float near_clip: input near clip
		@param float far_clip: input far clip
		@return int
		*/
		static int set_proj(glm::mat4& proj, float fx, float fy, float width,
			float height, float cx, float cy, float near_clip, float far_clip);

		/**
		@brief transfer eigen matrix to glm matrix for opengl
		@param Eigen::Matrix4d mat: input eigen 4x4 matrix (row-major)
		@return glm::mat: return glm4x4 matrix (column-major)
		*/
		static glm::mat4 eigen4d2glm4(Eigen::Matrix4d mat);

		/**
		@brief generate vertex and uv buffer from mesh grid
		@param std::shared_ptr<three::TriangleMesh> mesh: input 
			triangle mesh for rendering
		@param GLfloat* vertexBuffer: output vertex buffer data
		@param GLfloat* colorBuffer: output buffer data
		@return int
		*/
		int genVertexColorBufferData(std::shared_ptr<three::TriangleMesh> mesh,
			GLfloat* vertexBuffer,
			GLfloat* uvBuffer);

	public:
		OffScreenRender();
		~OffScreenRender();

		/**
		@brief init function
			init OpenGL for image warping
		@param std::string vertexShaderName: vertex shader name
		@param std::string fragShaderName: fragment shader name
		@return int
		*/
		int init(std::string vertexShaderName, std::string fragShaderName);

		/**
		@brief release function
			release OpenGL buffers
		@return int
		*/
		int release();

		/**
		@brief render mesh to image
		@param std::shared_ptr<three::TriangleMesh> mesh:
			input mesh for rendering
		@param std::shared_ptr<GLCamParam> camera:
			input camera for rendering
		@param cv::Size imgsize: input size of the image
		@param cv::Mat & depth: output rendered depth image
		@param cv::Mat & color: output rendered color image
		@return int
		*/
		int render(std::shared_ptr<three::TriangleMesh> mesh, 
			Eigen::Matrix4d intmat,
			Eigen::Matrix4d extmat,
			cv::Size imgsize,
			cv::Mat & depth,
			cv::Mat & color);

	
	};
};


#endif