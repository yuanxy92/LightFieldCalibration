/**
@brief sphere mesh class used for VR content rendering
@author Shane Yuan
@date Jun 30, 2017
*/

#include "SphereMesh.h"

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Cholesky>
#include <Eigen/SparseCholesky>
#include <Eigen/IterativeLinearSolvers>	


three::SphereMesh::SphereMesh() : meshrows(300), meshcols(400), lambda(4.5) {}
three::SphereMesh::~SphereMesh() {}

/**
@brief function to generate regular triangle mesh
@return
*/
int three::SphereMesh::genTriangleMesh() {
	float quadWidth = static_cast<float>(colorImg.cols - 1) / (meshcols - 1);
	float quadHeight = static_cast<float>(colorImg.rows - 1) / (meshrows - 1);
	vertices_2d_.create(meshrows, meshcols);
	for (size_t row = 0; row < meshrows; row++) {
		for (size_t col = 0; col < meshcols; col++) {
			vertices_2d_(row, col) = cv::Vec2f(quadWidth * col, quadHeight * row);
		}
	}
	return 0;
}

/**
@brief refine depth map
@return
*/
int three::SphereMesh::refineDepthMap() {
	// get disparity map
	disparityMap.create(depthMap.rows, depthMap.cols);
	for (size_t row = 0; row < depthMap.rows; row++) {
		for (size_t col = 0; col < depthMap.cols; col++) {
			disparityMap(row, col) = 1 / (static_cast<float>(depthMap(row, col)) / 100000);
		}
	}
	vertexDisparityMap.create(meshrows, meshcols);
	for (size_t row = 0; row < meshrows; row++) {
		for (size_t col = 0; col < meshcols; col++) {
			cv::Point2f pt = vertices_2d_(row, col);
			vertexDisparityMap(row, col) = disparityMap(pt.y, pt.x);
		}
	}
	// prepare matrix for refine depth map
	int nodeNum = meshrows * meshcols;
	int edgeNum = (meshrows - 1) * (meshcols - 1) * 3 + meshrows + meshcols - 2;
	Eigen::VectorXf b(nodeNum + edgeNum);
	Eigen::SparseMatrix<float> A(nodeNum + edgeNum, nodeNum);
	std::vector<Eigen::Triplet<float>> spVals(nodeNum + edgeNum * 2);
	Eigen::VectorXf x(nodeNum);
	// fill matrix data term
	for (size_t row = 0; row < meshrows; row++) {
		for (size_t col = 0; col < meshcols; col++) {
			int ind = row * meshcols + col;
			b(ind) = vertexDisparityMap(row, col);
			x(ind) = b(ind);
			spVals[ind] = Eigen::Triplet<float>(ind, ind, 1);
		}
	}
	// fill matrix smooth term
	int index = meshrows * meshcols;
	int ind = index;
	for (int row = 0; row < meshrows; row++) {
		for (int col = 0; col < meshcols; col++) {
			// center node
			int indCenter = row * meshcols + col;
			// six neighbors
			// bottom right node
			if (row < meshrows - 1 && col < meshcols - 1) {
				int indBR = (row + 1) * meshcols + (col + 1);
				spVals[ind] = Eigen::Triplet<float>(index, indCenter, lambda);
				ind++;
				spVals[ind] = Eigen::Triplet<float>(index, indBR, -lambda);
				ind++;
				b(index) = 0;
				index++;
			}
			// bottom node
			if (row < meshrows - 1) {
				int indB = (row + 1) * meshcols + col;
				spVals[ind] = Eigen::Triplet<float>(index, indCenter, lambda);
				ind++;
				spVals[ind] = Eigen::Triplet<float>(index, indB, -lambda);
				ind++;
				b(index) = 0;
				index++;
			}
			// right node
			if (col < meshcols - 1) {
				int indR = row * meshcols + (col + 1);
				spVals[ind] = Eigen::Triplet<float>(index, indCenter, lambda);
				ind++;
				spVals[ind] = Eigen::Triplet<float>(index, indR, -lambda);
				ind++;
				b(index) = 0;
				index++;
			}
		}
	}
	A.setFromTriplets(spVals.begin(), spVals.end());
	Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<float>> solver;
	solver.setMaxIterations(1000);
	solver.setTolerance(1e-6f);
	solver.compute(A);
	x = solver.solveWithGuess(b, x);
	std::cout << "#iterations:     " << solver.iterations() << std::endl;
	std::cout << "estimated error: " << solver.error() << std::endl;

	// write back to opencv mat
	cv::Mat_<float> vertexDisparityMap2(meshrows, meshcols);
	for (int row = 0; row < meshrows; row++) {
		for (int col = 0; col < meshcols; col++) {
			int ind = row * meshcols + col;
			vertexDisparityMap2(row, col) = x(ind);
		}
	}
	cv::Mat_<float> disparitySmooth;
	cv::resize(vertexDisparityMap2, disparitySmooth, depthMap.size());
	depthMapSmooth.create(depthMap.rows, depthMap.cols);
	for (size_t row = 0; row < depthMap.rows; row++) {
		for (size_t col = 0; col < depthMap.cols; col++) {
			depthMapSmooth(row, col) = static_cast<unsigned short>(1 / (disparitySmooth(row, col)) * 100000);
		}
	}

	return 0;
}

/**
@brief set input for spherical mesh
@param cv::Mat img: input color image
@param cv::Mat_<unsigned short> depth: input depth image
@param cv::Mat K: input intrinsic camera matrix
@return int
*/
int three::SphereMesh::setInput(cv::Mat img, cv::Mat_<unsigned short> depth,
	cv::Mat_<float> K) {
	this->colorImg = img;
	this->depthMap = depth;
	this->K = K;
	return 0;
}

int three::SphereMesh::debug() {
	this->genTriangleMesh();
	this->refineDepthMap();
	return 0;
}