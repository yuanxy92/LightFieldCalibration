/**
@brief CameraParamEstimator.h
C++ head file for camera parameter estimation
@author Shane Yuan
@date Feb 8, 2018
*/

#ifndef _ROBUST_STITCHER_CAMERA_PARAMETER_ESTIMATOR_H_
#define _ROBUST_STITCHER_CAMERA_PARAMETER_ESTIMATOR_H_ 

// std
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>

// opencv
#include <opencv2/opencv.hpp>

#include "FeatureMatch.h"
#include "BundleAdjustment.h"

#define DEBUG_CAMERA_PARAM_ESTIMATOR

namespace calib {
	
	/**
	@brief graph class 
	*/
	class DisjointSets {
	private:
		std::vector<int> rank_;
		std::vector<int> parent;
		std::vector<int> size;
	public:
		DisjointSets(int elem_count = 0) { createOneElemSets(elem_count); }
		void createOneElemSets(int elem_count);
		int findSetByElem(int elem);
		int mergeSets(int set1, int set2);
	};
	struct GraphEdge {
		int from, to;
		float weight;

		GraphEdge(int from, int to, float weight);
		bool operator <(const GraphEdge& other) const { return weight < other.weight; }
		bool operator >(const GraphEdge& other) const { return weight > other.weight; }
	};
	struct IncDistance {
		IncDistance(std::vector<int> &vdists) : dists(&vdists[0]) {}
		void operator ()(const GraphEdge &edge) { dists[edge.to] = dists[edge.from] + 1; }
		int* dists;
	};

	struct CalcRotation {
		int num_images;
		const Matchesinfo* pairwise_matches;
		CameraParams* cameras;
		cv::Mat connection;
		std::vector<std::pair<size_t, size_t>> pairs;

		CalcRotation(int _num_images, const std::vector<Matchesinfo> &_pairwise_matches,
			std::vector<CameraParams> &_cameras, cv::Mat con)
			: num_images(_num_images), pairwise_matches(&_pairwise_matches[0]),
			cameras(&_cameras[0]), connection(con) {
			pairs = Connection::getConnections(connection);
		}

		void operator ()(const GraphEdge &edge) {
			cv::Mat essMat;
			for (size_t i = 0; i < pairs.size(); i++) {
				if (pairs[i].first == edge.from && pairs[i].second == edge.to) {
					essMat = pairwise_matches[i].H.inv();
				}
				else if (pairs[i].second == edge.from && pairs[i].first == edge.to) {
					essMat = pairwise_matches[i].H;
				}
			}

			cv::Mat_<double> K_from = cv::Mat::eye(3, 3, CV_64F);
			K_from(0, 0) = cameras[edge.from].focal;
			K_from(1, 1) = cameras[edge.from].focal * cameras[edge.from].aspect;
			K_from(0, 2) = cameras[edge.from].ppx;
			K_from(1, 2) = cameras[edge.from].ppy;

			cv::Mat_<double> K_to = cv::Mat::eye(3, 3, CV_64F);
			K_to(0, 0) = cameras[edge.to].focal;
			K_to(1, 1) = cameras[edge.to].focal * cameras[edge.to].aspect;
			K_to(0, 2) = cameras[edge.to].ppx;
			K_to(1, 2) = cameras[edge.to].ppy;

			cv::Mat R = K_from.inv() * essMat * K_to;
			cameras[edge.to].R = cameras[edge.from].R * R;
		}
	};
	class Graph {
	private:
		std::vector< std::list<GraphEdge> > edges_;
	public:
		
	private:

	public:
		Graph(int num_vertices = 0) { create(num_vertices); }
		void create(int num_vertices) { edges_.assign(num_vertices, std::list<GraphEdge>()); }
		int numVertices() const { return static_cast<int>(edges_.size()); }
		void addEdge(int from, int to, float weight);
		template <typename B> B forEach(B body) const;
		template <typename B> B walkBreadthFirst(int from, B body) const;
	};

	/**
	@brief class for camera parameter estimation
	*/
	class CameraParamEstimator {
	private:
		// input variables
		std::vector<cv::Mat> imgs;
		cv::Mat connection;
		std::vector<Imagefeature> features;
		std::vector<Matchesinfo> matchesInfo;
		// camera params
		bool isFocalSet;
		std::vector<CameraParams> cameras;
		// graph used for camera params estimation
		Graph graph;
		std::vector<int> centers;
		// parameter for bundle adjustment refinement
		cv::Mat camParamBA;
		std::vector<std::pair<size_t, size_t>> edgesBA;
		std::vector<size_t> edgesIndBA;
		cv::Mat err1_, err2_;
		size_t totalMatchesNum;
	public:

	private:
		/**
		@brief construct max spanning tree
		@return int
		*/
		int constructMaxSpanningTree();

		/**
		@brief estimate focal length from homography matrix
		@param cv::Mat H: input homography matrix
		@param double &f0: output estimated first focal length
		@param double &f1: output estimated second focal length
		@param bool &f0_ok: output estimation status for first focal length
		@param bool &f1_ok: output estimation status for second focal length
		@return int
		*/
		int focalsFromHomography(const cv::Mat H, double &f0, double &f1,
			bool &f0_ok, bool &f1_ok);

		/**
		@brief estimate focal length
		@return int
		*/
		int estimateFocal();

		/**
		@brief estimate camera params from matches info
		@return int
		*/
		int estimateInitCameraParams();

		/**
		@brief refine camera parameters using bundle adjustment
		@return int
		*/
		int bundleAdjustRefine();

		/**
		@brief calculate error in bundle adjustment
		@param cv::Mat & err: output error matrix
		@return int
		*/
		int bundleAdjustCalcError(cv::Mat & err);

		/**
		@brief calculate jacobian in bundle adjustment	
		@param cv::Mat & jac: output calculate jacobian matrix
		@return int
		*/
		int bundleAdjustCalcJacobian(cv::Mat & jac);

	public:
		CameraParamEstimator();
		~CameraParamEstimator();

		/**
		@brief set input for camera parameter estimation
		@param std::vector<cv::Mat> imgs: input images
		@param cv::Mat connection: input connection matrix
		@param std::vector<Imagefeature> features: input computed features
		@param std::vector<Matchesinfo> matchesInfo: input matching infos
		@return int
		*/
		int init(std::vector<cv::Mat> imgs, cv::Mat connection,
			std::vector<Imagefeature> features, std::vector<Matchesinfo> matchesInfo);

		/**
		@brief get estimated camera params
		@return std::vector<CameraParams>: returned camera params 
		*/
		std::vector<CameraParams> getCameraParams();

		/**
		@brief set camera focal
		@param float focal: input focal length of cameras
		@return int
		*/
		int setFocal(float focal);

		/**
		@brief estimate camera parameters
		@return int
		*/
		int estimate();

		/**
		@brief save calibration result to file
		@param std::string filename: output filename
		@return int
		*/
		int saveCalibrationResult(std::string filename);

	};
}

#endif