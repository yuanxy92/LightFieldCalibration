/**
@brief class for feature matching
@author Shane Yuan
@date May 26, 2018
*/

#include "Fragments.h"

#define _DEBUG_FRAGMENTS

Fragments::Fragments() : thresh(1)  {
	render = std::make_shared<gl::OffScreenRender>();
	render->init("E:\\Project\\LightFieldGiga\\SparseToDenseStereo\\MakeFragments\\mesh_vertexcolor_render.vertex.glsl",
		"E:\\Project\\LightFieldGiga\\SparseToDenseStereo\\MakeFragments\\mesh_vertexcolor_render.fragment.glsl");
}
Fragments::~Fragments() {}

/**
@brief set input color/depth image and camera poses
@param std::vector<std::string> colornames: input color image names
@param std::vector<std::string> depthnames: input depth image names
@param std::string cameraname: input camera parameter name 
	(include camera focal and extrinsic matrix)
@return int
*/
int Fragments::init(std::vector<std::string> colornames, 
	std::vector<std::string> depthnames,
	std::string cameraname,
	int representInd){
	this->colornames = colornames;
	this->depthnames = depthnames;
	this->cameraname = cameraname;
	this->intname = intname;
	this->representInd = representInd;
	return 0;
}

/**
@brief read camera parameters
@return int
*/
int Fragments::readCameraParams() {
	// read images
	colorImgs.resize(colornames.size());
	depthImgs.resize(colornames.size());
	for (size_t i = 0; i < colornames.size(); i++) {
		this->colorImgs[i] = cv::imread(colornames[i]);
		this->depthImgs[i] = cv::imread(depthnames[i], cv::IMREAD_ANYDEPTH);
	}
	// read camera parameters
	camInts.resize(colornames.size());
	camExts.resize(colornames.size());
	std::fstream fs(cameraname, std::ios::in);
	for (size_t i = 0; i < colornames.size(); i++) {
		int ind;
		double focal;
		double ext[9];
		fs >> ind;
		fs >> focal;
		for (size_t j = 0; j < 9; j++) {
			fs >> ext[j];
		}
		// intrinsic matrix
		camInts[i] = Eigen::Matrix4d::Identity();
		camInts[i](0, 0) = focal;
		camInts[i](1, 1) = focal;
		camInts[i](0, 2) = this->colorImgs[i].cols / 2;
		camInts[i](1, 2) = this->colorImgs[i].rows / 2;
		// extrinsic matrix
		camExts[i] = Eigen::Matrix4d::Identity();
		camExts[i](0, 0) = ext[0];
		camExts[i](0, 1) = ext[1];
		camExts[i](0, 2) = ext[2];
		camExts[i](1, 0) = ext[3];
		camExts[i](1, 1) = ext[4];
		camExts[i](1, 2) = ext[5];
		camExts[i](2, 0) = ext[6];
		camExts[i](2, 1) = ext[7];
		camExts[i](2, 2) = ext[8];
#ifdef _DEBUG_FRAGMENTS2
		std::cout << camInts[i] << std::endl;
		std::cout << camExts[i] << std::endl;
#endif // _DEBUG_FRAGMENTS
	}
	fs.close();
	return 0;
}

/**
@brief project image to 3d model
@return int
*/
int Fragments::projectImageToModel(std::shared_ptr<three::TriangleMesh> mesh,
	Eigen::Matrix4d intmat, Eigen::Matrix4d extmat,
	cv::Mat colorImg, cv::Mat depthImg, float thresh) {
	// generate vertices in model
	Eigen::Matrix4d Pinv = (intmat * extmat).inverse();
	for (size_t row = 0; row < colorImg.rows; row++) {
		for (size_t col = 0; col < colorImg.cols; col++) {
			//float depth = 1.0f / 100.0f * static_cast<float>(depthImg.at<unsigned short>(row, col));
			float depth = 1.0f / 100.0f * (float)(((unsigned short *)
				depthImg.data)[row * depthImg.cols + col]);
			Eigen::Vector4d pimg(depth * col,
				depth * row, depth, 1.0f);
			Eigen::Vector4d pworld_homo = Pinv * pimg;
			Eigen::Vector3d pworld(pworld_homo(0) / pworld_homo(3),
				pworld_homo(1) / pworld_homo(3),
				pworld_homo(2) / pworld_homo(3));
			//cv::Vec3b color = colorImg.at<cv::Vec3b>(row, col);
			cv::Vec3b color = ((cv::Vec3b*)colorImg.data)[row * colorImg.cols + col];
			Eigen::Vector3d pworld_color(static_cast<double>(color.val[2]) / 255.0f,
				static_cast<double>(color.val[1]) / 255.0f,
				static_cast<double>(color.val[0]) / 255.0f);
			mesh->vertices_.push_back(pworld);
			mesh->vertex_colors_.push_back(pworld_color);
		}
	}
	mesh->vertex_normals_.clear();
	mesh->vertex_normals_.resize(mesh->vertices_.size());
	// generate triangles
	mesh->triangles_.clear();
	for (size_t row = 0; row < colorImg.rows - 1; row++) {
		for (size_t col = 0; col < colorImg.cols - 1; col++) {
			int indtl = row * colorImg.cols + col;
			int indtr = row * colorImg.cols + col + 1;
			int indbl = (row + 1) * colorImg.cols + col;
			int indbr = (row + 1) * colorImg.cols + col + 1;
			Eigen::Vector3d vertextl = mesh->vertices_[indtl];
			Eigen::Vector3d vertextr = mesh->vertices_[indtr];
			Eigen::Vector3d vertexbl = mesh->vertices_[indbl];
			Eigen::Vector3d vertexbr = mesh->vertices_[indbr];
			float dist_tltr = (vertextl - vertextr).norm();
			float dist_blbr = (vertexbl - vertexbr).norm();
			float dist_tlbl = (vertextl - vertexbl).norm();
			float dist_trbr = (vertextr - vertexbr).norm();
			float dist_tlbr = (vertextl - vertexbr).norm();
			float dist_bltr = (vertexbl - vertextr).norm();
			if (dist_tltr < thresh && dist_tlbl < thresh && dist_bltr < thresh) {
				mesh->triangles_.push_back(Eigen::Vector3i(indtl, indbl, indtr));
			}
			if (dist_blbr < thresh && dist_trbr < thresh && dist_bltr < thresh) {
				mesh->triangles_.push_back(Eigen::Vector3i(indbl, indbr, indtr));
			}
		}
	}
	// estimate normal
	mesh->triangle_normals_.resize(mesh->triangles_.size());
	mesh->ComputeVertexNormals(true);
	mesh->ComputeTriangleNormals(true);
	return 0;
}

/**
@brief project 3d model to image
@return int
*/
int Fragments::projectModelToImage(std::shared_ptr<three::TriangleMesh> mesh,
	Eigen::Matrix4d intmat, Eigen::Matrix4d extmat,
	cv::Mat & colorImg, cv::Mat & depthImg, cv::Size imgsize,
	std::shared_ptr<gl::OffScreenRender> render) {
	render->render(mesh, intmat, extmat, imgsize, depthImg, colorImg);
	return 0;
}

/**
@brief depth map fusion
@return int
*/
int Fragments::fuseDepthMaps() {
	std::vector<cv::Mat> reprojectDepths(colornames.size());
	std::vector<cv::Mat> reprojectColors(colornames.size());
	for (size_t i = 0; i < colornames.size(); i++) {
		projectModelToImage(meshes[i], camInts[i], camExts[representInd], reprojectColors[i],
			reprojectDepths[i], colorImgs[i].size(), render);
	}
	// fusion
	cv::Mat depth(colorImgs[0].size(), CV_32F);
	cv::Mat weight(colorImgs[0].size(), CV_32F);
	depth.setTo(0);
	weight.setTo(0);
	for (size_t i = 0; i < colornames.size(); i++) {
		int step = reprojectDepths[0].cols;
		for (size_t row = 0; row < reprojectDepths[i].rows; row++) {
			for (size_t col = 0; col < reprojectDepths[i].cols; col++) {
				float depthVal = ((float*)reprojectDepths[i].data)[row * step + col];
				if (depthVal < 400.0) {
					((float*)depth.data)[row * step + col] += depthVal;
					((float*)weight.data)[row * step + col] += 1;
				}
			}
		}
	}
	cv::Mat refineDepth;
	cv::divide(depth, weight, refineDepth);
	refineDepth = refineDepth * 100;
	refineDepth.convertTo(refineDepthMap, CV_16U);
	return 0;
}

/**
@brief generate single mesh
@return int
*/
int Fragments::generateSingleMesh() {
	// project all the point back to 3d space
	meshes.resize(colornames.size());
	for (size_t i = 0; i < colornames.size(); i++) {
		std::cout << cv::format("Project depth image %04d ...", i) << std::endl;
		meshes[i] = std::make_shared<three::TriangleMesh>();
		projectImageToModel(meshes[i], camInts[i], camExts[i], colorImgs[i], depthImgs[i], thresh);
	}
	return 0;
}

int Fragments::fusion() {
	readCameraParams();

	//generateSingleMesh();
	//fuseDepthMaps();

	fuseDepthMapByVoting();

	//std::shared_ptr<three::TriangleMesh> mesh = std::make_shared<three::TriangleMesh>();
	//projectImageToModel(mesh, camInts[representInd], camExts[representInd], colorImgs[representInd], refineDepthMap);
	//// write mesh into files
	//three::WriteTriangleMeshToPLY(cv::format("refine_mesh.ply", 0), *(mesh.get()));
	//cv::imwrite("refine_color.png", colorImgs[representInd]);
	//cv::imwrite("refine_depth.png", refineDepthMap);
	return 0;
}

/**
@brief function to get results
@return int
*/
int Fragments::getResults(cv::Mat& refineDepthMap, cv::Mat& colorImg) {
	refineDepthMap = this->refineDepthMap;
	colorImg = this->colorImgs[this->representInd];
	return 0;
}

/**
@brief fuse depth map by voting
@return int
*/
int Fragments::fuseDepthMapByVoting() {
	cv::Size size = this->colorImgs[this->representInd].size();
	cv::Mat baseVoteMap(size, CV_8UC1);
	cv::Mat baseCoverMap(size, CV_8UC1);
	baseVoteMap.setTo(1);
	baseCoverMap.setTo(1);
	std::shared_ptr<three::TriangleMesh> mesh = 
		std::make_shared<three::TriangleMesh>();
	// project image to model
	cv::Mat representColorImg;
	cv::Mat representDepthMap;
	this->projectImageToModel(mesh, camInts[representInd], camExts[representInd],
		colorImgs[representInd], depthImgs[representInd], thresh);
	this->projectModelToImage(mesh, camInts[representInd], camExts[representInd],
		representColorImg, representDepthMap, size, render);
	cv::Mat baseColorImg = representColorImg.clone();
	cv::Mat baseDepthMap = representDepthMap.clone();
	// map other depth map to reprenetative view
	float baselineFocal = 0.35 * camInts[representInd](0, 0);
	for (size_t i = 0; i < depthImgs.size(); i++) {
		if (i == representInd)
			continue;
		cv::Mat depthImgOther;
		cv::Mat colorImgOther;
		std::shared_ptr<three::TriangleMesh> meshOther =
			std::make_shared<three::TriangleMesh>();
		this->projectImageToModel(meshOther, camInts[i], camExts[i],
			colorImgs[i], depthImgs[i], thresh);
		this->projectModelToImage(meshOther, camInts[representInd], camExts[representInd],
			colorImgOther, depthImgOther, size, render);
		// vote depth value
		std::cout << "Process image depth map " << i << "..." << std::endl;
		for (size_t row = 0; row < depthImgOther.rows; row++) {
			for (size_t col = 0; col < depthImgOther.cols; col++) {
	/*			cv::Vec4b colorVal = ((cv::Vec4b*)colorImgOther.data)[row
					* colorImgOther.cols + col];
				if (colorVal.val[3] == 0)
					continue;*/
				// convert depth to disparity
				float representDepth = ((float*)representDepthMap.data)[row
					* representDepthMap.cols + col];
				float otherDepth = ((float*)depthImgOther.data)[row
					* depthImgOther.cols + col];
				if (otherDepth < 400) {
					((uchar*)baseCoverMap.data)[row * baseCoverMap.cols + col] ++;
					float representDisparity = baselineFocal / representDepth;
					float otherDisparity = baselineFocal / otherDepth;
					if (abs(representDisparity - otherDisparity) < 1) {
						uchar voteNum = ((uchar*)baseVoteMap.data)[row * baseVoteMap.cols + col];
						float voteDepthVal = ((float*)baseDepthMap.data)[row * baseDepthMap.cols + col];
						float newVoteDepthVal = (voteDepthVal * (float)voteNum + otherDepth) / ((float)voteNum + 1);
						((float*)baseDepthMap.data)[row * baseDepthMap.cols + col] = newVoteDepthVal;
						((uchar*)baseVoteMap.data)[row * baseVoteMap.cols + col] ++;
					}
				}
			}
		}
	}

	cv::Mat depthMap(baseDepthMap.size(), CV_16U);
	int voteNumThresh = depthImgs.size() / 3;
	for (size_t row = 0; row < baseDepthMap.rows; row++) {
		for (size_t col = 0; col < baseDepthMap.cols; col++) {
			int voteNum = (int)(((unsigned char*)baseVoteMap.data)[row * depthMap.cols + col]);
			int coverNum = (int)(((unsigned char*)baseCoverMap.data)[row * baseCoverMap.cols + col]);
			float voteRate = (float)(voteNum) / coverNum;
			if (voteNum > voteNumThresh && voteRate > 0.8) {
				float voteDepthVal = ((float*)baseDepthMap.data)[row * baseDepthMap.cols + col];
				if (voteDepthVal > 399.0f) {
					((unsigned short*)depthMap.data)[row * depthMap.cols + col] = 0;
				}
				else {
					unsigned short fileDepth = (unsigned short)(voteDepthVal * 100);
					((unsigned short*)depthMap.data)[row * depthMap.cols + col] = fileDepth;
				}
			}
			else {
				((unsigned short*)depthMap.data)[row * depthMap.cols + col] = 0;
			}
		}
	}

	//cv::imwrite("E:\\Project\\LightFieldGiga\\data\\data1\\result_fusion\\fragment_0_29_14\\depth_vote.png", depthMap);
	//cv::imwrite("E:\\Project\\LightFieldGiga\\data\\data1\\result_fusion\\fragment_0_29_14\\map_vote.png", baseVoteMap);

	refineDepthMap = depthMap;

	return 0;
}

