/**
@brief warper class, warping function
@author Shane Yuan
@date May 30, 2018
*/

#include "Registration.h"
#include "Warper.h"
#include "Blender.h"
#include "GraphCutMask.h"
#include "Triangulate.h"

#define _DEBUG_REGISTRATION

calib::Registration::Registration() {}
calib::Registration::~Registration() {}

/*******************************************************************/
/*                         utility function                        */
/*******************************************************************/
/**
@brief function to inpaint depth images
@return int
*/
cv::Mat_<cv::Vec3f> calib::Registration::inpaint(cv::Mat_<cv::Vec3f> point3DImg) {
	cv::Mat outDepth;
	// get mask
	cv::Mat mask(point3DImg.size(), CV_8U);
	mask.setTo(cv::Scalar(0));
	for (size_t i = 0; i < point3DImg.rows; i++) {
		for (size_t j = 0; j < point3DImg.cols; j++) {
			cv::Vec3f val = point3DImg.at<cv::Vec3f>(i, j);
			if (val.val[0] == 0 && val.val[1] == 0 && val.val[2] == 0)
				mask.at<uchar>(i, j) = 255;
		}
	}
	// split point3D image
	std::vector<cv::Mat> xyz3DImgs;
	cv::split(point3DImg, xyz3DImgs);
	for (size_t i = 0; i < 3; i++) {
		cv::inpaint(xyz3DImgs[i], mask, xyz3DImgs[i], 5, cv::INPAINT_NS);
	}
	cv::merge(xyz3DImgs, outDepth);
	return outDepth;
}

/**
@brief function to generate weighted depth map mask
@return cv::Mat weightMask
*/
cv::Mat_<float> calib::Registration::generateWeightedMask(cv::Size imgsize) {
	cv::Mat_<float> weightMask(imgsize, CV_32F);
	weightMask.setTo(0);
	for (int i = 0; i < imgsize.height; i++) {
		for (int j = 0; j < imgsize.width; j++) {
			float ratioX = static_cast<float>(abs(j - imgsize.width / 2)) /
				static_cast<float>(imgsize.width / 2);
			float ratioY = static_cast<float>(abs(i - imgsize.height / 2)) /
				static_cast<float>(imgsize.height / 2);
			weightMask(i, j) = 1.0f - std::max<float>(ratioX, ratioY);
		}
	}
	return weightMask;
}

/**
@brief function to project point 3d map to 2d distance map
@return int
*/
cv::Mat_<cv::Vec3f> calib::Registration::project3DPointsToDistanceMap(cv::Mat_<cv::Vec3f> points) {
	cv::Mat_<cv::Vec3f> distmap;
	distmap.create(points.size());
	for (size_t i = 0; i < distmap.rows; i++) {
		for (size_t j = 0; j < distmap.cols; j++) {
			cv::Vec3f val = points(i, j);
			float dist = sqrt(val.val[0] * val.val[0] + val.val[1] * val.val[1] +
				val.val[2] * val.val[2]);
			distmap(i, j) = cv::Vec3f(dist, dist, dist);
		}
	}
	return distmap;
}


/**
@brief function to project 2d depth map to 3d point
@return int
*/
int calib::Registration::projectDepthMapTo3DPoints(cv::Mat depth, 
	cv::Mat_<cv::Vec3f> & points,
	cv::Mat K, cv::Mat R) {
	cv::Mat_<float> K4 = cv::Mat::eye(4, 4, CV_64F);
	cv::Mat_<float> R4 = cv::Mat::eye(4, 4, CV_64F);
	K.copyTo(K4(cv::Rect(0, 0, 3, 3)));
	R.copyTo(R4(cv::Rect(0, 0, 3, 3)));
	cv::Mat_<float> Pinv = (K4 * R4).inv();
	cv::Mat_<float> pixels(4, depth.rows * depth.cols);
	for (size_t row = 0; row < depth.rows; row++) {
		for (size_t col = 0; col < depth.cols; col++) {
			float depthVal = static_cast<float>(depth.at
				<unsigned short>(row, col)) / 100.0f;
			pixels(0, row * depth.cols + col) = depthVal * col;
			pixels(1, row * depth.cols + col) = depthVal * row;
			pixels(2, row * depth.cols + col) = depthVal;
			pixels(3, row * depth.cols + col) = 1.0f;
		}
	}
	pixels = Pinv * pixels;
	size_t ind = 0;
	for (size_t row = 0; row < depth.rows; row++) {
		for (size_t col = 0; col < depth.cols; col++) {
			points.at<cv::Vec3f>(row, col) = cv::Vec3f(
				pixels(0, ind), pixels(1, ind), pixels(2, ind)
			);
			ind++;
		}
	}
	return 0;
}


/*******************************************************************/
/*                      function for processing                    */
/*******************************************************************/
/**
@brief function to read camera params
@return int
*/
int calib::Registration::readCameraParams() {
	this->cameraNum = colornames.size();
	this->cameras.resize(this->cameraNum);
	this->colorImgs.resize(this->cameraNum);
	this->depthImgs.resize(this->cameraNum);
	std::fstream fs(this->calibname, std::ios::in);
	for (size_t i = 0; i < this->cameraNum; i++) {
		this->colorImgs[i] = cv::imread(colornames[i]);
		this->depthImgs[i] = cv::imread(depthnames[i], cv::IMREAD_ANYDEPTH);
		int ind;
		float focal;
		fs >> ind;
		fs >> focal;
		this->cameras[i].intEigen = Eigen::Matrix4d::Identity();
		this->cameras[i].extEigen = Eigen::Matrix4d::Identity();
		this->cameras[i].intCV = cv::Mat::eye(3, 3, CV_64F);
		this->cameras[i].extCV = cv::Mat::eye(3, 3, CV_64F);
		this->cameras[i].intEigen(0, 0) = focal;
		this->cameras[i].intEigen(1, 1) = focal;
		this->cameras[i].intEigen(0, 2) = this->colorImgs[i].cols / 2;
		this->cameras[i].intEigen(1, 2) = this->colorImgs[i].rows / 2;
		this->cameras[i].intCV.at<double>(0, 0) = focal;
		this->cameras[i].intCV.at<double>(1, 1) = focal;
		this->cameras[i].intCV.at<double>(0, 2) = this->colorImgs[i].cols / 2;
		this->cameras[i].intCV.at<double>(1, 2) = this->colorImgs[i].rows / 2;
		for (size_t row = 0; row < 3; row++) {
			for (size_t col = 0; col < 3; col++) {
				float val;
				fs >> val;
				this->cameras[i].extEigen(row, col) = val;
				this->cameras[i].extCV.at<double>(row, col) = val;
			}
		}
	}
	fs.close();
	return 0;
}

/**
@brief funtion to merge point3D maps
@return int
*/
int calib::Registration::mergePoint3DImages() {
	// get final rect
	sphereRect = cv::Rect(0, 0, 0, 0);
	for (size_t i = 0; i < rects.size(); i++) {
		sphereRect = sphereRect | rects[i];
	}
	cv::Point tl = sphereRect.tl();
	cv::Size size = sphereRect.size();
	spherePoint3D.create(size);
	spherePoint3DMask.create(size);
	spherePoint3D.setTo(cv::Vec3f(0, 0, 0));
	spherePoint3DMask.setTo(0);
	for (size_t i = 0; i < rects.size(); i++) {
		cv::Rect rect = rects[i];
		rect.x -= tl.x;
		rect.y -= tl.y;
		for (size_t row = 0; row < spherePoint3Ds[i].rows; row++) {
			for (size_t col = 0; col < spherePoint3Ds[i].cols; col++) {
				float weight = sphereWeightDepthMasks[i](row, col);
				spherePoint3D(rect.y + row, rect.x + col) += cv::Vec3f(
					spherePoint3Ds[i](row, col).val[0] * weight,
					spherePoint3Ds[i](row, col).val[1] * weight,
					spherePoint3Ds[i](row, col).val[2] * weight
				);
				spherePoint3DMask(rect.y + row, rect.x + col) += weight;
			}
		}
	}
	for (size_t i = 0; i < spherePoint3D.rows; i++) {
		for (size_t j = 0; j < spherePoint3D.cols; j++) {
			float weight = spherePoint3DMask(i, j);
			if (weight > 0) {
				cv::Vec3f val = spherePoint3D(i, j);
				spherePoint3D(i, j) = cv::Vec3f(val.val[0] / weight,
					val.val[1] / weight, val.val[2] / weight);
			}
		}
	}
	// project to depth map
	sphereDepthMap.create(spherePoint3D.rows, spherePoint3D.cols);
	sphereAngleMap.create(spherePoint3D.rows, spherePoint3D.cols);
	for (size_t i = 0; i < spherePoint3D.rows; i++) {
		for (size_t j = 0; j < spherePoint3D.cols; j++) {
			float weight = spherePoint3DMask(i, j);
			if (weight > 0) {
				cv::Vec3f val = spherePoint3D(i, j);
				float distance = sqrt(pow(val.val[0], 2) +
					pow(val.val[1], 2) + pow(val.val[2], 2));
				unsigned short distanceUint16 = distance * 100;
				sphereDepthMap(i, j) = distanceUint16;
				cv::Vec2f angle;
				angle.val[0] = acosf(val.val[2] / distance);
				angle.val[1] = atan2f(val.val[1], val.val[0]);
				//angle.val[1] = atanf(val.val[1] / val.val[0]);
				sphereAngleMap(i, j) = angle;
			}
			else {
				sphereDepthMap(i, j) = 600 * 100;
				sphereAngleMap(i, j) = cv::Vec2f(0, 0);
			}
		}
	}
	return 0;
}

/**
@brief project depth map to sphere
@return int
*/
int calib::Registration::sphereFusion() {
	// prepare variables
	rects.resize(this->cameraNum);
	point3Ds.resize(this->cameraNum);
	std::shared_ptr<calib::SphericalWarper> warper =
		std::make_shared<calib::SphericalWarper>();
	warper->setScale(this->cameras[0].intEigen(0, 0));
	std::shared_ptr<calib::Blender> blender = 
		std::make_shared<calib::Blender>();
	std::shared_ptr<cv::detail::Blender> colorBlender =
		std::make_shared<cv::detail::MultiBandBlender>();
	std::shared_ptr<calib::GraphCutMask> cuter =
		std::make_shared<calib::GraphCutMask>();
	// 
	std::vector<cv::Mat> xmaps(this->cameraNum);
	std::vector<cv::Mat> ymaps(this->cameraNum);
	// project 
	for (size_t i = 0; i < this->cameraNum; i++) {
		cv::Mat K = this->cameras[i].intCV;
		cv::Mat R = this->cameras[i].extCV;
		cv::Mat Kf, Rf;
		K.convertTo(Kf, CV_32F);
		R.convertTo(Rf, CV_32F);
		cv::Mat Rfinv = Rf.inv();
		// build forward map 
		rects[i] = warper->buildMapsForward(this->depthImgs[i].size(),
			Kf, Rfinv, xmaps[i], ymaps[i]);
		// project 2d depth map to 3d points map
		this->point3Ds[i].create(this->depthImgs[i].size());
		this->projectDepthMapTo3DPoints(this->depthImgs[i],
			this->point3Ds[i], Kf, Rf);
#ifdef _DEBUG_REGISTRATION
		cv::Mat pointview = this->point3Ds[i];
#endif // _DEBUG_REGISTRATION
		// inpainting
		//this->point3Ds[i] = Registration::inpaint(this->point3Ds[i]);
	}
	// prepare large sphere image
	sphereRect = cv::Rect(0, 0, 0, 0);
	std::vector<cv::Point> corners(this->cameraNum);
	std::vector<cv::Size> sizes(this->cameraNum);
	for (size_t i = 0; i < this->cameraNum; i++) {
		sphereRect = sphereRect | rects[i];
		corners[i] = rects[i].tl();
		sizes[i] = rects[i].size();
	}
	colorBlender->prepare(corners, sizes);
	// generate final sphere point3D
	spherePoint3Ds.resize(this->cameraNum);
	sphereWeightDepthMasks.resize(this->cameraNum);
	std::vector<cv::Mat> sphereColorImgs(this->cameraNum);
	std::vector<cv::Mat> masks(this->cameraNum);
	std::vector<cv::Mat> gcmasks(this->cameraNum);
	for (size_t i = 0; i < this->cameraNum; i++) {
		// warp 3d points
		cv::remap(this->point3Ds[i], spherePoint3Ds[i], xmaps[i], 
			ymaps[i], cv::INTER_NEAREST);
		// warp color
		cv::remap(this->colorImgs[i], sphereColorImgs[i], xmaps[i], 
			ymaps[i], cv::INTER_NEAREST);
		// warp mask
		cv::Mat mask(this->colorImgs[i].size(), CV_8U);
		mask.setTo(cv::Scalar(255));
		cv::remap(mask, masks[i], xmaps[i], ymaps[i], cv::INTER_NEAREST);
		cv::threshold(masks[i], masks[i], 254, 255, cv::THRESH_BINARY);
		int erosion_size = 5;
		cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT,
			cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
			cv::Point(erosion_size, erosion_size));
		cv::erode(masks[i], masks[i], element);
		// weight depth mask
		sphereWeightDepthMasks[i] = Registration::generateWeightedMask(this->colorImgs[i].size());
		cv::remap(this->sphereWeightDepthMasks[i], this->sphereWeightDepthMasks[i], xmaps[i],
			ymaps[i], cv::INTER_NEAREST);
	}
	// get color panorama
	// apply graph cut
	cuter->graphcut(sphereColorImgs, masks, rects, gcmasks);
 	for (size_t i = 0; i < this->cameraNum; i++) {
		// merge to sphere
		colorBlender->feed(sphereColorImgs[i], gcmasks[i], corners[i]);
	}
	// get merged result
	colorBlender->blend(spherePanorama, sphereMask);
	spherePanorama.convertTo(spherePanorama, CV_8U);
	// get 3d point panorama
	this->mergePoint3DImages();
	return 0;
}

/**
@brief project to 3D mesh
@return int
*/
int calib::Registration::projectTo3DMesh() {
	float thresh = 1.5f;
	mesh = std::make_shared<three::TriangleMesh>();
	cv::Mat_<int> indexMat(spherePanorama.size());
	indexMat.setTo(-1);
	int ind = 0;
	for (size_t row = 0; row < spherePoint3D.rows; row++) {
		for (size_t col = 0; col < spherePoint3D.cols; col++) {
			uchar maskVal = sphereMask.at<uchar>(row, col);
			if (maskVal > 0) {
				indexMat(row, col) = ind;
				cv::Vec3f val = spherePoint3D(row, col);
				Eigen::Vector3d point(val.val[0], val.val[1], val.val[2]);
				cv::Vec3b color = spherePanorama.at<cv::Vec3b>(row, col);
				Eigen::Vector3d point_color(static_cast<double>(color.val[2]) / 255.0f,
					static_cast<double>(color.val[1]) / 255.0f,
					static_cast<double>(color.val[0]) / 255.0f);
				mesh->vertices_.push_back(point);
				mesh->vertex_colors_.push_back(point_color);
				ind++;
			}
		}
	}
	mesh->vertex_normals_.clear();
	mesh->vertex_normals_.resize(mesh->vertices_.size());
	// generate triangles
	mesh->triangles_.clear();
	for (size_t row = 0; row < spherePanorama.rows - 1; row++) {
		for (size_t col = 0; col < spherePanorama.cols - 1; col++) {
			int indtl = indexMat(row, col);
			int indtr = indexMat(row, col + 1);
			int indbl = indexMat(row + 1, col);
			int indbr = indexMat(row + 1, col + 1);
			// direction: tl->br
			if (indtl >= 0 && indbr >= 0) {
				Eigen::Vector3d vertextl = mesh->vertices_[indtl];
				Eigen::Vector3d vertexbr = mesh->vertices_[indbr];
				if (indtr >= 0) {
					Eigen::Vector3d vertextr = mesh->vertices_[indtr];
					float dist_tltr = (vertextl - vertextr).norm();
					float dist_trbr = (vertextr - vertexbr).norm();
					float dist_tlbr = (vertextl - vertexbr).norm();
					if (dist_tltr < thresh && dist_trbr < thresh && dist_tlbr < thresh) {
						mesh->triangles_.push_back(Eigen::Vector3i(indtl, indbr, indtr));
					}
				}
				if (indbl >= 0) {
					Eigen::Vector3d vertexbl = mesh->vertices_[indbl];
					float dist_blbr = (vertexbl - vertexbr).norm();
					float dist_tlbl = (vertextl - vertexbl).norm();
					float dist_tlbr = (vertextl - vertexbr).norm();
					if (dist_blbr < thresh && dist_tlbl < thresh && dist_tlbr < thresh) {
						mesh->triangles_.push_back(Eigen::Vector3i(indtl, indbl, indbr));
					}
				}
			}
			else if (indbl >= 0 && indtr >= 0) {// direction: bl->tr
				Eigen::Vector3d vertexbl = mesh->vertices_[indbl];
				Eigen::Vector3d vertextr = mesh->vertices_[indtr];
				if (indtl >= 0) {
					Eigen::Vector3d vertextl = mesh->vertices_[indtl];
					float dist_tlbl = (vertextl - vertexbl).norm();
					float dist_bltr = (vertexbl - vertextr).norm();
					float dist_tltr = (vertextl - vertextr).norm();
					if (dist_tlbl < thresh && dist_bltr < thresh && dist_tltr < thresh) {
						mesh->triangles_.push_back(Eigen::Vector3i(indtl, indbl, indtr));
					}
				}
				if (indbr >= 0) {
					Eigen::Vector3d vertexbr = mesh->vertices_[indbr];
					float dist_blbr = (vertexbl - vertexbr).norm();
					float dist_trbr = (vertextr - vertexbr).norm();
					float dist_bltr = (vertexbl - vertextr).norm();
					if (dist_blbr < thresh && dist_trbr < thresh && dist_bltr < thresh) {
						mesh->triangles_.push_back(Eigen::Vector3i(indbl, indbr, indtr));
					}
				}
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
@brief init registation class
@param std::string calibname: filename of calibration result
@param std::vector<std::string> colornames: names of color images
@param std::vector<std::string> depthnames: names of depth images
@return int
*/
int calib::Registration::init(std::string calibname, 
	std::vector<std::string> colornames,
	std::vector<std::string> depthnames) {
	this->calibname = calibname;
	this->colornames = colornames;
	this->depthnames = depthnames;
	// read camera parameters
	this->readCameraParams();
	return 0;
}

int calib::Registration::registrate() {
	this->sphereFusion();
	this->projectTo3DMesh();

	Triangulate triangulate;
	std::vector<Eigen::Vector3d> vertex_;
	std::vector<Eigen::Vector3d> vertex_colors_;
	std::vector<Eigen::Vector2d> vertex_uvs_;
	std::vector<Eigen::Vector3i> triangles_;
	triangulate.triangulate(spherePanorama, spherePoint3D,
		vertex_, vertex_colors_, vertex_uvs_, triangles_);

	texmesh = std::make_shared<three::TexTriangleMesh>();
	texmesh->vertices_ = vertex_;
	texmesh->vertex_colors_ = vertex_colors_;
	texmesh->triangles_ = triangles_;
	texmesh->vertex_uvs_ = vertex_uvs_;
	texmesh->texture_ = spherePanorama;
	texmesh->ComputeVertexNormals(true);
	texmesh->ComputeTriangleNormals(true);

	//three::WriteTriangleMeshToPLY(cv::format("mesh_texture.ply", 0), *(texmesh.get()));
	//three::OBJModelIO::save(".", "mesh_texture", texmesh);
	return 0;
}

/**
@brief get registrated mesh
@return std::shared_ptr<three::TexTriangleMesh> texmesh
*/
std::shared_ptr<three::TexTriangleMesh> calib::Registration::getTexMesh() {
	return texmesh;
}

/**
@brief get panorama depth map (uint16)
@return cv::Mat: return depth map
*/
cv::Mat calib::Registration::getDepthMap() {
	return sphereDepthMap;
}
