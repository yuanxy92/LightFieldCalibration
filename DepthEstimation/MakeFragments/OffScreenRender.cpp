/**
@brief file OffScreenRender.cpp
warp image using 2D mesh grid
@author Shane Yuan
@date Mar 14, 2018
*/

#include "OffScreenRender.h"

#include <thread>
#include <chrono>

using namespace gl;

OffScreenRender::OffScreenRender() : znear(20), zfar(400) {}
OffScreenRender::~OffScreenRender() {}

/**
@brief error callback function
@param int error: error id
@param const char* description: error description
@return int
*/
void OffScreenRender::error_callback(int error, const char* description) {
	fprintf(stderr, "Error: %s\n", description);
}

/**
@brief init function
init OpenGL for image warping
@param std::string vertexShaderName: vertex shader name
@param std::string fragShaderName: fragment shader name
@return int
*/
int OffScreenRender::init(std::string vertexShaderName, std::string fragShaderName) {
	// Initialise GLFW
	glfwSetErrorCallback(error_callback);
	if (!glfwInit()) {
		fprintf(stderr, "Failed to initialize GLFW\n");
		return -1;
	}
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
	glfwWindowHint(GLFW_SAMPLES, 2);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	window = glfwCreateWindow(1000, 1000, "hide window", NULL, NULL);
	if (window == nullptr) {
		fprintf(stderr, "Failed to open GLFW window.\n");
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	// hide window
	// glfwHideWindow(window);

	// Initialize GLEW
	glewExperimental = true; // Needed for core profile
	if (glewInit() != GLEW_OK) {
		fprintf(stderr, "Failed to initialize GLEW\n");
		return -1;
	}

	 // Black background
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	// Enable depth test
	glEnable(GL_DEPTH_TEST);
	// Accept fragment if it closer to the camera than the former one
	glDepthFunc(GL_LESS);
	// RGBA texture blending
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	// load shader
	programID = glshader::LoadShaders(vertexShaderName, fragShaderName);
	// generate vertex arrays
	glGenVertexArrays(1, &vertexArrayID);
	glBindVertexArray(vertexArrayID);
	// generate vertex/uv buffer
	glGenBuffers(1, &vertexID);
	glGenBuffers(1, &colorID);
	// generate frame buffer
	glGenFramebuffers(1, &frameBufferID);
	glGenRenderbuffers(1, &frameBufferDepthID);
	// generate textures
	glGenTextures(1, &outputTextureID);
	return 0;
}

/**
@brief release function
release OpenGL buffers
@return int
*/
int OffScreenRender::release() {
	glDeleteVertexArrays(1, &vertexArrayID);
	glDeleteBuffers(1, &vertexID);
	glDeleteBuffers(1, &colorID);
	glDeleteTextures(1, &outputTextureID);
	glDeleteFramebuffers(1, &frameBufferID);
	glDeleteRenderbuffers(1, &frameBufferDepthID);
	glfwTerminate();
	return 0;
}

/**
@brief generate vertex and uv buffer from mesh grid
@param std::shared_ptr<three::TriangleMesh> mesh: input
triangle mesh for rendering
@param GLfloat* vertexBuffer: output vertex buffer data
@param GLfloat* colorBuffer: output buffer data
@return int
*/
int OffScreenRender::genVertexColorBufferData(std::shared_ptr<three::TriangleMesh> mesh,
	GLfloat* vertexBuffer,
	GLfloat* colorBuffer) {
	size_t triangleNum = mesh->triangles_.size();
	for (size_t i = 0; i < triangleNum; i++) {
		Eigen::Vector3i triangle = mesh->triangles_[i];
		for (int vertexInd = 0; vertexInd < 3; vertexInd++) {
			// position
			vertexBuffer[i * 9 + vertexInd * 3 + 0] =
				mesh->vertices_[triangle(vertexInd)](0);
			vertexBuffer[i * 9 + vertexInd * 3 + 1] =
				mesh->vertices_[triangle(vertexInd)](1);
			vertexBuffer[i * 9 + vertexInd * 3 + 2] =
				mesh->vertices_[triangle(vertexInd)](2);
			// color
			colorBuffer[i * 9 + vertexInd * 3 + 0] =
				mesh->vertex_colors_[triangle(vertexInd)](0);
			colorBuffer[i * 9 + vertexInd * 3 + 1] =
				mesh->vertex_colors_[triangle(vertexInd)](1);
			colorBuffer[i * 9 + vertexInd * 3 + 2] =
				mesh->vertex_colors_[triangle(vertexInd)](2);
		}
	}
	return 0;
}

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
int OffScreenRender::set_proj(glm::mat4& proj, float fx, float fy, float width,
	float height, float cx, float cy, float near_clip, float far_clip) {
	proj[0][0] = 2 * fx / width;
	proj[1][0] = 0.0f;
	proj[2][0] = (2 * cx - width) / width;
	proj[3][0] = 0.0f;
	proj[0][1] = 0.0f;
	proj[1][1] = -2 * fy / height;
	proj[2][1] = (height - 2 * cy) / height;
	proj[3][1] = 0.0f;
	proj[0][2] = 0.0f;
	proj[1][2] = 0.0f;
	proj[2][2] = -(far_clip + near_clip) / (near_clip - far_clip);
	proj[3][2] = 2 * near_clip * far_clip / (near_clip - far_clip);
	proj[0][3] = 0.0f;
	proj[1][3] = 0.0f;
	proj[2][3] = 1.0f;
	proj[3][3] = 0.0f;
	return 0;
}

/**
@brief transfer eigen matrix to glm matrix for opengl
@param Eigen::Matrix4d mat: input eigen 4x4 matrix (row-major)
@return glm::mat: return glm4x4 matrix (column-major)
*/
glm::mat4 OffScreenRender::eigen4d2glm4(Eigen::Matrix4d mat) {
	glm::mat4 glmmat;
	for (size_t i = 0; i < 4; i++) {
		for (size_t j = 0; j < 4; j++) {
			glmmat[j][i] = mat(i, j);
		}
	}
	return glmmat;
}

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
int OffScreenRender::render(std::shared_ptr<three::TriangleMesh> mesh,
	Eigen::Matrix4d intmat,
	Eigen::Matrix4d extmat,
	cv::Size imgsize,
	cv::Mat & depth,
	cv::Mat & color) {
	// compute opengl projection matrix
	glm::mat4 proj;
	OffScreenRender::set_proj(proj, intmat(0, 0), intmat(1, 1),
		imgsize.width, imgsize.height, 
		static_cast<float>(imgsize.width) / 2,
		static_cast<float>(imgsize.height) / 2,
		znear, zfar);
	
	// bind frame buffer color
	glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);
	// generate output texture
	glBindTexture(GL_TEXTURE_2D, outputTextureID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, imgsize.width, imgsize.height,
		0, GL_BGR, GL_FLOAT, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	// bind output texture to frame buffer
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, outputTextureID, 0);

	// bind frame buffer depth 
#if 1
	glBindRenderbuffer(GL_RENDERBUFFER, frameBufferDepthID);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, imgsize.width, imgsize.height);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, frameBufferDepthID);
#else
	// Alternative : Depth texture. Slower, but you can sample it later in your shader
	GLuint depthTexture;
	glGenTextures(1, &depthTexture);
	glBindTexture(GL_TEXTURE_2D, depthTexture);
	glTexImage2D(GL_TEXTURE_2D, 0,GL_DEPTH_COMPONENT24, imgsize.width, imgsize.height, 0,GL_DEPTH_COMPONENT, GL_FLOAT, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depthTexture, 0);
#endif

	// set draw buffers number
	GLenum DrawBuffers[2] = { GL_COLOR_ATTACHMENT0, GL_DEPTH_ATTACHMENT };
	glDrawBuffers(2, DrawBuffers); // "1" is the size of DrawBuffers

	// check frame buffer
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
	}

	// compute array buffer from mesh
	size_t vertexBufferSize, colorBufferSize;
	GLfloat* vertexBuffer;
	GLfloat* colorBuffer;
	// generate buffer data
	size_t triangleNum = mesh->triangles_.size();
	vertexBuffer = new GLfloat[triangleNum * 9];
	colorBuffer = new GLfloat[triangleNum * 9];
	vertexBufferSize = triangleNum * 9 * sizeof(GLfloat);
	colorBufferSize = triangleNum * 9 * sizeof(GLfloat);
	this->genVertexColorBufferData(mesh, vertexBuffer, colorBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexID);
	glBufferData(GL_ARRAY_BUFFER, vertexBufferSize, vertexBuffer, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, colorID);
	glBufferData(GL_ARRAY_BUFFER, colorBufferSize, colorBuffer, GL_STATIC_DRAW);

	// draw mesh
	glBindFramebuffer(GL_FRAMEBUFFER, frameBufferID);
	glViewport(0, 0, imgsize.width, imgsize.height);
	// Clear the screen
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	// Use our shader
	glUseProgram(programID);
	// Send our transformation to the currently bound shader, 
	// in the "MVP" uniform
	// Get a handle for our "MVP" uniform
	GLuint matrixID = glGetUniformLocation(programID, "MVP");
	GLuint extmatID = glGetUniformLocation(programID, "EXTMAT");
	// Use our shader
	glm::mat4 Model = glm::mat4(1.0);
	glm::mat4 MVP = proj;
	glm::mat4 EXTMAT = OffScreenRender::eigen4d2glm4(extmat);
	glUniformMatrix4fv(matrixID, 1, GL_FALSE, &MVP[0][0]);
	glUniformMatrix4fv(extmatID, 1, GL_FALSE, &EXTMAT[0][0]);
	// 1rst attribute buffer : vertices
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, vertexID);
	glVertexAttribPointer(
		0,                  // attribute. No particular reason for 0, but must match the layout in the shader.
		3,                  // size
		GL_FLOAT,           // type
		GL_FALSE,           // normalized?
		0,                  // stride
		(void*)0            // array buffer offset
	);
	// 2nd attribute buffer : colors
	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, colorID);
	glVertexAttribPointer(
		1,                                // attribute. No particular reason for 1, but must match the layout in the shader.
		3,                                // size
		GL_FLOAT,                         // type
		GL_FALSE,                         // normalized?
		0,                                // stride
		(void*)0                          // array buffer offset
	);

	// Draw the mesh
	glDrawArrays(GL_TRIANGLES, 0, triangleNum * 3); // 12*3 indices starting at 0 -> 12 triangles
	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);

	glfwSwapBuffers(window);
	glfwPollEvents();

	// get drawn color image
	float* pixels = new float[imgsize.width * imgsize.height * 4];
	glReadPixels(0, 0, imgsize.width, imgsize.height, GL_RGBA, GL_FLOAT, pixels);
	cv::Mat img(imgsize, CV_32FC4, pixels);
	cv::cvtColor(img, color, cv::COLOR_RGBA2BGRA);
	cv::flip(color, color, 0);

	// get drawn depth map
	float* pixels2 = new float[imgsize.width * imgsize.height];
	glReadPixels(0, 0, imgsize.width, imgsize.height, GL_DEPTH_COMPONENT, GL_FLOAT, pixels2);
	cv::Mat img2(imgsize, CV_32F, pixels2);
	for (size_t i = 0; i < img2.rows; i++) {
		for (size_t j = 0; j < img2.cols; j++) {
			float z_b = img2.at<float>(i, j);
			float z_n = 2.0 * z_b - 1.0;
			float z_e = 2.0 * znear * zfar / (zfar + znear - z_n * (zfar - znear));
			img2.at<float>(i, j) = z_e;
		}
	}
	cv::flip(img2, depth, 0);

	delete[] colorBuffer;
	delete[] vertexBuffer;
	delete[] pixels;
	delete[] pixels2;

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glBindRenderbuffer(GL_FRAMEBUFFER, 0);

	return 0;
}
