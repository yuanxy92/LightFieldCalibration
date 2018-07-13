/**
@brief triangulate image into triangle mesh
@author Shane Yuan
@date May 30, 2018
*/

#include "Triangulate.h"

//#define _DEBUG_TRIANGULATE

Triangulate::Triangulate() : quadSize(30, 30) {}
Triangulate::~Triangulate() {}

/**
@brief visualize corners in image
@return int
*/
int Triangulate::visualize() {
	visual = img.clone();
	for (size_t i = 0; i < vertices.size(); i++) {
		cv::Point2f p = vertices[i];
		cv::circle(visual, p, 2, cv::Scalar(0, 0, 255), -1, 8, 0);
	}
	list<hed::Edge*>::const_iterator it;
	for (it = edges.begin(); it != edges.end(); ++it) {
		hed::Edge* edge = *it;
		for (int i = 0; i < 3; ++i) {
			hed::Edge* twinedge = edge->getTwinEdge();
			// Print only one edge (the highest value of the pointer)
			if (twinedge == NULL || (size_t)edge >(size_t)twinedge) {
				// Print source node and target node
				hed::Node* node1 = edge->getSourceNode();
				hed::Node* node2 = edge->getTargetNode();
				cv::Point2f p1(node1->x(), node1->y());
				cv::Point2f p2(node2->x(), node2->y());
				cv::line(visual, p1, p2, cv::Scalar(255, 0, 0), 1, 8, 0);
			}
			edge = edge->getNextEdgeInFace();
		}
	}
	return 0;
}

/**
@brief add points in uniform/smooth region
@return int
*/
std::vector<cv::Point> Triangulate::addPoints(cv::Mat & corners) {
	// calculate size
	cv::Size size;
	std::vector<cv::Point> points;
	size.width = img.cols / quadSize.width + 1;
	size.height = img.cols / quadSize.height + 1;
	pointNumMat.create(size);
	pointNumMat.setTo(0);
	for (size_t i = 0; i < corners.rows; i++) {
		cv::Point2f p = corners.at<cv::Point2f>(i, 0);
		int rowInd = p.y / quadSize.height;
		int colInd = p.x / quadSize.width;
		pointNumMat(rowInd, colInd) += 1;
		points.push_back(p);
	}
	// find quad with no points
	for (size_t row = 0; row < pointNumMat.rows; row++) {
		for (size_t col = 0; col < pointNumMat.cols; col++) {
			if (pointNumMat(row, col) == 0) {
				float x = (col + 0.5f) * quadSize.width;
				float y = (row + 0.5f) * quadSize.height;
				if (x > 0 && x < img.cols - 1 && y > 0 
					&& y < img.rows - 1) {
					points.push_back(cv::Point2f(x, y));
				}
			}
		}
	}
	return points;
}

/**
@brief check if 3 points are counter clockwise or clockwise
@param cv::Point[3] input : input 3 points
@return true/false : ccw/cw
*/
bool Triangulate::checkCCW(cv::Point input[3]) {
	cv::Mat_<float> mat(3, 3, CV_32F);
	for (size_t i = 0; i < 3; i++) {
		mat(i, 0) = input[i].x;
		mat(i, 1) = input[i].y;
		mat(i, 2) = 1.0f;
	}
	float val = cv::determinant(mat);
	bool returnVal;
	if (val > 0)
		returnVal = false;
	else returnVal = true;
	return returnVal;
}

/**
@brief check if 3 points are counter clockwise or clockwise
@param Eigen::Vector3d pt1: first triangle point
@param Eigen::Vector3d pt2: second triangle point
@param Eigen::Vector3d pt3: third triangle point
@return float: length of longest edge
*/
float Triangulate::computeLongestEdge(Eigen::Vector3d pt1,
	Eigen::Vector3d pt2, Eigen::Vector3d pt3) {
	double length = 0;
	double len1 = (pt1 - pt2).norm();
	double len2 = (pt1 - pt3).norm();
	double len3 = (pt2 - pt3).norm();
	return std::max<double>(len1, std::max<double>(len2, len3));
}

inline bool eqPoints(hed::Node*& p1, hed::Node*& p2) {
	double dx = p1->x() - p2->x();
	double dy = p1->y() - p2->y();
	double dist2 = dx*dx + dy*dy;
	const double eps = 1.0e-12;
	if (dist2 < eps)
		return true;

	return false;
}

inline bool ltLexPoint(const hed::Node* p1, const hed::Node* p2) {
	return (p1->x() < p2->x()) || (p1->x() == p2->x() && p1->y() < p2->y());
};

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
int Triangulate::triangulate(cv::Mat img,
	cv::Mat_<cv::Vec3f> spherePoint3D ,
	std::vector<Eigen::Vector3d> & vertex_,
	std::vector<Eigen::Vector3d> & vertex_colors_,
	std::vector<Eigen::Vector2d> & vertex_uvs_,
	std::vector<Eigen::Vector3i> & triangles_
	) {
	this->img = img;
	cv::Mat grayImg;
	cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
	cv::goodFeaturesToTrack(grayImg, corners, 20000, 0.005, 10);
	std::vector<cv::Point> vertices_ = addPoints(corners);

	hed::Triangulation triang;

	std::vector<hed::Node*> nodes;
	for (size_t i = 0; i < vertices_.size(); i++) {
		cv::Vec3b val = img.at<cv::Vec3b>(vertices_[i].y, vertices_[i].x);
		if (val.val[0] + val.val[1] + val.val[2] > 10) {
			hed::Node* node = new hed::Node(vertices_[i].x, vertices_[i].y);
			nodes.push_back(node);
			vertices.push_back(vertices_[i]);
			cv::Vec3f point = spherePoint3D(vertices_[i].y, vertices_[i].x);
			vertex_.push_back(Eigen::Vector3d(point.val[0], point.val[1], point.val[2]));
			vertex_colors_.push_back(Eigen::Vector3d(static_cast<double>(val.val[2]) / 255.0,
				static_cast<double>(val.val[1]) / 255.0,
				static_cast<double>(val.val[0]) / 255.0));
			float u = static_cast<float>(vertices_[i].x) / img.cols;
			float v = 1 - static_cast<float>(vertices_[i].y) / img.rows;
			vertex_uvs_.push_back(Eigen::Vector2d(u, v));
		}
	}

	std::cout << 6 << std::endl;
	triang.createDelaunay(nodes.begin(), nodes.end());
	edges = triang.getLeadingEdges();

	std::cout << 7 << std::endl;
	std::cout << edges.size() << std::endl;
	std::cout << vertices.size() << std::endl;

	Eigen::SparseMatrix<int> conn(vertices.size(), vertices.size());
	list<hed::Edge*>::const_iterator it;
	for (it = edges.begin(); it != edges.end(); ++it) {
		hed::Edge* edge = *it;
		for (int i = 0; i < 3; ++i) {
			hed::Edge* twinedge = edge->getTwinEdge();
			// Print only one edge (the highest value of the pointer)
			if (twinedge == NULL || (size_t)edge >(size_t)twinedge) {
				// Print source node and target node
				hed::Node* node1 = edge->getSourceNode();
				hed::Node* node2 = edge->getTargetNode();
				int ind1 = node1->id();
				int ind2 = node2->id();
				conn.insert(ind1, ind2) = 1;
				conn.insert(ind2, ind1) = 1;
			}
			edge = edge->getNextEdgeInFace();
		}
	}
	std::cout << 8 << std::endl;

#ifdef _DEBUG_TRIANGULATE
	this->visualize();
	cv::imwrite("visual.png", this->visual);
#endif

	std::cout << 9 << std::endl;

	// go through all the non-zero elements
	for (int firstInd = 0; firstInd < conn.outerSize(); firstInd++) {
		// get connected vertices
		cv::Point firstVertex = vertices[firstInd];
		std::vector<int> neighbors;
		std::vector<cv::Point> neighbotPoints;
		for (Eigen::SparseMatrix<int>::InnerIterator it(conn, firstInd); it; ++it) {
			int ind = it.row();
			if (ind > firstInd) {
				neighbors.push_back(ind);
			}
		}
		if (neighbors.size() < 2)
			continue;
		neighbotPoints.resize(neighbors.size());
		std::sort(neighbors.begin(), neighbors.end(), std::less<int>());
		for (size_t i = 0; i < neighbors.size(); i++) {
			neighbotPoints[i] = vertices[neighbors[i]];
		}
		// find connected 3 points
		cv::Point triangle[3];
		triangle[0] = firstVertex;
		for (int second = 0; second < neighbors.size() - 1; second++) {
			int secondInd = neighbors[second];
			triangle[1] = vertices[secondInd];
			for (int third = second + 1; third < neighbors.size(); third++) {
				int thirdInd = neighbors[third];
				triangle[2] = vertices[thirdInd];
				double longestEdge = computeLongestEdge(vertex_[firstInd],
					vertex_[secondInd], vertex_[thirdInd]);
				if (longestEdge < 15) {
					if (checkCCW(triangle)) {
						triangles_.push_back(Eigen::Vector3i(firstInd,
							secondInd, thirdInd));
					}
					else {
						triangles_.push_back(Eigen::Vector3i(firstInd,
							thirdInd, secondInd));
					}
				}
			}
		}
	}
	return 0;
}