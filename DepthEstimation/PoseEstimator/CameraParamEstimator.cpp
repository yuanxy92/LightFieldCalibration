/**
@brief CameraParamEstimator.h
C++ source file for camera parameter estimation
@author Shane Yuan
@date Feb 8, 2018
*/

#include "CameraParamEstimator.h"
#define HAVE_CERES_SOLVER

/**
@brief camera class functions
*/
calib::CameraParams::CameraParams() : focal(1), aspect(1), ppx(0), ppy(0),
R(cv::Mat::eye(3, 3, CV_64F)), t(cv::Mat::zeros(3, 1, CV_64F)) {}

calib::CameraParams::CameraParams(const CameraParams &other) { *this = other; }

calib::CameraParams& calib::CameraParams::operator =(const CameraParams &other) {
	focal = other.focal;
	ppx = other.ppx;
	ppy = other.ppy;
	aspect = other.aspect;
	R = other.R.clone();
	t = other.t.clone();
	return *this;
}

cv::Mat calib::CameraParams::K() const {
	cv::Mat_<double> k = cv::Mat::eye(3, 3, CV_64F);
	k(0, 0) = focal; k(0, 2) = ppx;
	k(1, 1) = focal * aspect; k(1, 2) = ppy;
	return k;
}

/**
@brief graph class
*/
void calib::DisjointSets::createOneElemSets(int n) {
	rank_.assign(n, 0);
	size.assign(n, 1);
	parent.resize(n);
	for (int i = 0; i < n; ++i)
		parent[i] = i;
}

int calib::DisjointSets::findSetByElem(int elem) {
	int set = elem;
	while (set != parent[set])
		set = parent[set];
	int next;
	while (elem != parent[elem]) {
		next = parent[elem];
		parent[elem] = set;
		elem = next;
	}
	return set;
}

int calib::DisjointSets::mergeSets(int set1, int set2) {
	if (rank_[set1] < rank_[set2]) {
		parent[set1] = set2;
		size[set2] += size[set1];
		return set2;
	}
	if (rank_[set2] < rank_[set1]) {
		parent[set2] = set1;
		size[set1] += size[set2];
		return set1;
	}
	parent[set1] = set2;
	rank_[set2]++;
	size[set2] += size[set1];
	return set2;
}

calib::GraphEdge::GraphEdge(int _from, int _to, float _weight) : 
	from(_from), to(_to), weight(_weight) {}

template <typename B>
B calib::Graph::forEach(B body) const {
	for (int i = 0; i < numVertices(); ++i) {
		std::list<GraphEdge>::const_iterator edge = edges_[i].begin();
		for (; edge != edges_[i].end(); ++edge)
			body(*edge);
	}
	return body;
}

template <typename B>
B calib::Graph::walkBreadthFirst(int from, B body) const {
	std::vector<bool> was(numVertices(), false);
	std::queue<int> vertices;
	was[from] = true;
	vertices.push(from);
	while (!vertices.empty()) {
		int vertex = vertices.front();
		vertices.pop();
		std::list<GraphEdge>::const_iterator edge = edges_[vertex].begin();
		for (; edge != edges_[vertex].end(); ++edge) {
			if (!was[edge->to]) {
				body(*edge);
				was[edge->to] = true;
				vertices.push(edge->to);
			}
		}
	}
	return body;
}

void calib::Graph::addEdge(int from, int to, float weight) {
	edges_[from].push_back(GraphEdge(from, to, weight));
}


/**
@brief class for camera parameter estimation
*/
calib::CameraParamEstimator::CameraParamEstimator() : isFocalSet(false) {}
calib::CameraParamEstimator::~CameraParamEstimator() {}

/**
@brief construct max spanning tree
@return int
*/
int calib::CameraParamEstimator::constructMaxSpanningTree() {
	std::vector<GraphEdge> edges;
	// Construct images graph and remember its edges
	// get connection pairs
	std::vector<std::pair<size_t, size_t>> pairs = Connection::getConnections(connection);
	size_t connectionNum = pairs.size();
	for (size_t ind = 0; ind < connectionNum; ind++) {
		if (matchesInfo[ind].H.empty())
			continue;
		float conf = static_cast<float>(matchesInfo[ind].num_inliers);
		edges.push_back(GraphEdge(pairs[ind].first, pairs[ind].second, conf));
	}
	// init
	size_t num_images = imgs.size();
	DisjointSets comps(num_images);
	graph.create(num_images);
	std::vector<int> span_tree_powers(num_images, 0);

	// Find maximum spanning tree
	sort(edges.begin(), edges.end(), std::greater<GraphEdge>());
	for (size_t i = 0; i < edges.size(); ++i) {
		int comp1 = comps.findSetByElem(edges[i].from);
		int comp2 = comps.findSetByElem(edges[i].to);
		if (comp1 != comp2) {
			comps.mergeSets(comp1, comp2);
			graph.addEdge(edges[i].from, edges[i].to, edges[i].weight);
			graph.addEdge(edges[i].to, edges[i].from, edges[i].weight);
			span_tree_powers[edges[i].from]++;
			span_tree_powers[edges[i].to]++;
		}
	}

	// Find spanning tree leafs
	std::vector<int> span_tree_leafs;
	for (int i = 0; i < num_images; i ++) {
		if (span_tree_powers[i] == 1)
			span_tree_leafs.push_back(i);
	}

	// Find maximum distance from each spanning tree vertex
	std::vector<int> max_dists(num_images, 0);
	std::vector<int> cur_dists;
	for (size_t i = 0; i < span_tree_leafs.size(); ++i) {
		cur_dists.assign(num_images, 0);
		graph.walkBreadthFirst(span_tree_leafs[i], IncDistance(cur_dists));
		for (int j = 0; j < num_images; ++j)
			max_dists[j] = std::max(max_dists[j], cur_dists[j]);
	}

	// Find min-max distance
	int min_max_dist = max_dists[0];
	for (int i = 1; i < num_images; ++i) {
		if (min_max_dist > max_dists[i])
			min_max_dist = max_dists[i];
	}

	// Find spanning tree centers
	centers.clear();
	for (int i = 0; i < num_images; ++i) {
		if (max_dists[i] == min_max_dist)
			centers.push_back(i);
	}

	if (centers.size() == 0 || centers.size() > 2) {
		centers.clear();
		std::cout << "Can not find center vertex automatically"\
			", choose the vertex with index " << num_images / 2 <<
			" as center !"<< std::endl;
		centers.push_back(num_images / 2);
	}

	return 0;
}

/**
@brief estimate focal length from homography matrix
@param cv::Mat H: input homography matrix
@param double &f0: output estimated first focal length
@param double &f1: output estimated second focal length
@param bool &f0_ok: output estimation status for first focal length
@param bool &f1_ok: output estimation status for second focal length
@return int
*/
int calib::CameraParamEstimator::focalsFromHomography(const cv::Mat H, double &f0, double &f1,
	bool &f0_ok, bool &f1_ok) {
	assert(H.type() == CV_64F && H.size() == cv::Size(3, 3));
	const double* h = H.ptr<double>();
	double d1, d2; // Denominators
	double v1, v2; // Focal squares value candidates

	f1_ok = true;
	d1 = h[6] * h[7];
	d2 = (h[7] - h[6]) * (h[7] + h[6]);
	v1 = -(h[0] * h[1] + h[3] * h[4]) / d1;
	v2 = (h[0] * h[0] + h[3] * h[3] - h[1] * h[1] - h[4] * h[4]) / d2;
	if (v1 < v2) std::swap(v1, v2);
	if (v1 > 0 && v2 > 0) f1 = std::sqrt(std::abs(d1) > std::abs(d2) ? v1 : v2);
	else if (v1 > 0) f1 = std::sqrt(v1);
	else f1_ok = false;

	f0_ok = true;
	d1 = h[0] * h[3] + h[1] * h[4];
	d2 = h[0] * h[0] + h[1] * h[1] - h[3] * h[3] - h[4] * h[4];
	v1 = -h[2] * h[5] / d1;
	v2 = (h[5] * h[5] - h[2] * h[2]) / d2;
	if (v1 < v2) std::swap(v1, v2);
	if (v1 > 0 && v2 > 0) f0 = std::sqrt(std::abs(d1) > std::abs(d2) ? v1 : v2);
	else if (v1 > 0) f0 = std::sqrt(v1);
	else f0_ok = false;
	return 0;
}

/**
@brief estimate focal length
@return int
*/
int calib::CameraParamEstimator::estimateFocal() {
	// get connection pairs
	std::vector<std::pair<size_t, size_t>> pairs = Connection::getConnections(connection);
	size_t connectionNum = pairs.size();
	// estimate focal
	std::vector<double> allFocals;
	for (size_t i = 0; i < connectionNum; i++) {
		const Matchesinfo &m = matchesInfo[i];
		if (m.H.empty())
			continue;
		double f0, f1;
		bool f0ok, f1ok;
		focalsFromHomography(m.H, f0, f1, f0ok, f1ok);
		if (f0ok && f1ok)
			allFocals.push_back(std::sqrt(f0 * f1));
	}
	std::sort(allFocals.begin(), allFocals.end());
	double median;
	if (allFocals.size() % 2 == 1)
		median = allFocals[allFocals.size() / 2];
	else
		median = (allFocals[allFocals.size() / 2 - 1] + allFocals[allFocals.size() / 2]) * 0.5;
	// set focal length for camera params
	for (size_t i = 0; i < imgs.size(); i++) {
		cameras[i].focal = median;
	}
	this->isFocalSet = true;
	return 0;
}

/**
@brief estimate camera params from matches info
@return int
*/
int calib::CameraParamEstimator::estimateInitCameraParams() {
	// set ppx ppy aspect
	for (size_t i = 0; i < imgs.size(); i ++) {
		cameras[i].ppx = static_cast<double>(features[i].imgsize.width) / 2;
		cameras[i].ppy = static_cast<double>(features[i].imgsize.height) / 2;
		cameras[i].aspect = 1.0f;
	}
	// set focal
	if (!isFocalSet) {
		this->estimateFocal();
	}
	// restore global motion
	constructMaxSpanningTree();
	graph.walkBreadthFirst(centers[0], CalcRotation(imgs.size(),
		matchesInfo, cameras, connection));
	return 0;
}

/**
@brief set input for camera parameter estimation
@param std::vector<cv::Mat> imgs: input images
@param cv::Mat connection: input connection matrix
@param std::vector<Imagefeature> features: input computed features
@param std::vector<Matchesinfo> matchesInfo: input matching infos
@return int
*/
int calib::CameraParamEstimator::init(std::vector<cv::Mat> imgs, cv::Mat connection,
	std::vector<calib::Imagefeature> features, std::vector<calib::Matchesinfo> matchesInfo) {
	this->imgs = imgs;
	this->connection = connection;
	this->features = features;
	this->matchesInfo = matchesInfo;
	this->cameras.resize(imgs.size());
	return 0;
}

/**
@brief get estimated camera params
@return std::vector<CameraParams>: returned camera params
*/
std::vector<calib::CameraParams> calib::CameraParamEstimator::getCameraParams() {
	return cameras;
}

/**
@brief estimate camera parameters
@return int
*/
int calib::CameraParamEstimator::setFocal(float focal) {
	// set ppx ppy aspect
	for (size_t i = 0; i < imgs.size(); i++) {
		cameras[i].focal = focal;
	}
	this->isFocalSet = true;
	return 0;
}

/**
@brief refine camera parameters using bundle adjustment
@return int
*/
int calib::CameraParamEstimator::bundleAdjustRefine() {
	// init camera parameter (we use focal, rotation, 4 dim for every camera)
	camParamBA.create(4 * cameras.size(), 1, CV_64F);
	cv::SVD svd;
    for (int i = 0; i < cameras.size(); ++i) {
        camParamBA.at<double>(i * 4, 0) = cameras[i].focal;
        svd(cameras[i].R, cv::SVD::FULL_UV);
        cv::Mat R = svd.u * svd.vt;
		R = R.inv();
        if (determinant(R) < 0)
            R *= -1;
        cv::Mat rvec;
        cv::Rodrigues(R, rvec);
        camParamBA.at<double>(i * 4 + 1, 0) = rvec.at<double>(0, 0);
        camParamBA.at<double>(i * 4 + 2, 0) = rvec.at<double>(1, 0);
        camParamBA.at<double>(i * 4 + 3, 0) = rvec.at<double>(2, 0);
    }
	// compute total matches num
	// Compute number of correspondences
    totalMatchesNum = 0;
	std::vector<std::pair<size_t, size_t>> pairs = Connection::getConnections(connection);
	edgesBA.clear();
	edgesIndBA.clear();
	size_t connectionNum = pairs.size();
    for (size_t i = 0; i < connectionNum; i ++) {
		if (matchesInfo[i].confidence > 1.5) {
			totalMatchesNum += matchesInfo[i].num_inliers; 
			edgesBA.push_back(pairs[i]);
			edgesIndBA.push_back(i);
		}
    }
	CvLevMarq solver(cameras.size() * 4, totalMatchesNum * 3, cv::TermCriteria(
		cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 1000, DBL_EPSILON));

    cv::Mat err, jac;
    CvMat matParams = camParamBA;
    cvCopy(&matParams, solver.param);

    int iter = 0;
    for(;;) {
        const CvMat* _param = 0;
        CvMat* _jac = 0;
        CvMat* _err = 0;
        bool proceed = solver.update(_param, _jac, _err);
        cvCopy(_param, &matParams);
        if (!proceed || !_err)
            break;
        if (_jac) {
            bundleAdjustCalcJacobian(jac);
            CvMat tmp = jac;
            cvCopy(&tmp, _jac);
        }
        if (_err) {
            bundleAdjustCalcError(err);
            iter++;
            CvMat tmp = err;
            cvCopy(&tmp, _err);
        }
		std::cout << "Bundle adjustment iteration " 
			<< iter << " ..." << std::endl;
    }
    // Check if all camera parameters are valid
    bool ok = true;
    for (int i = 0; i < camParamBA.rows; ++i) {
        if (cvIsNaN(camParamBA.at<double>(i,0))) {
            ok = false;
            break;
        }
    }
    if (!ok) {
        return false;
	}
	// update cameras
	for (size_t i = 0; i < cameras.size(); i ++) {
        cameras[i].focal = camParamBA.at<double>(i * 4, 0);
        cv::Mat rvec(3, 1, CV_64F);
        rvec.at<double>(0, 0) = camParamBA.at<double>(i * 4 + 1, 0);
        rvec.at<double>(1, 0) = camParamBA.at<double>(i * 4 + 2, 0);
        rvec.at<double>(2, 0) = camParamBA.at<double>(i * 4 + 3, 0);
        Rodrigues(rvec, cameras[i].R);
		cameras[i].R = cameras[i].R.inv();
    }
	return 0;
}

/**
@brief calculate error in bundle adjustment
@param cv::Mat & err: output error matrix
@return int
*/
int calib::CameraParamEstimator::bundleAdjustCalcError(cv::Mat & err) {
	err.create(totalMatchesNum * 3, 1, CV_64F);
    int match_idx = 0;
    for (size_t edge_idx = 0; edge_idx < edgesBA.size(); edge_idx ++) {
        int i = edgesBA[edge_idx].first;
        int j = edgesBA[edge_idx].second;
        double f1 = camParamBA.at<double>(i * 4, 0);
        double f2 = camParamBA.at<double>(j * 4, 0);

        double R1[9];
        cv::Mat R1_(3, 3, CV_64F, R1);
        cv::Mat rvec(3, 1, CV_64F);
        rvec.at<double>(0, 0) = camParamBA.at<double>(i * 4 + 1, 0);
        rvec.at<double>(1, 0) = camParamBA.at<double>(i * 4 + 2, 0);
        rvec.at<double>(2, 0) = camParamBA.at<double>(i * 4 + 3, 0);
        cv::Rodrigues(rvec, R1_);

        double R2[9];
        cv::Mat R2_(3, 3, CV_64F, R2);
        rvec.at<double>(0, 0) = camParamBA.at<double>(j * 4 + 1, 0);
        rvec.at<double>(1, 0) = camParamBA.at<double>(j * 4 + 2, 0);
        rvec.at<double>(2, 0) = camParamBA.at<double>(j * 4 + 3, 0);
        cv::Rodrigues(rvec, R2_);

        const calib::Imagefeature& features1 = features[i];
        const calib::Imagefeature& features2 = features[j];
        const calib::Matchesinfo& matches_info = matchesInfo[edgesIndBA[edge_idx]];

		cv::Mat_<double> K1 = cv::Mat::eye(3, 3, CV_64F);
        K1(0,0) = f1; K1(0,2) = features1.imgsize.width * 0.5;
        K1(1,1) = f1; K1(1,2) = features1.imgsize.height * 0.5;

        cv::Mat_<double> K2 = cv::Mat::eye(3, 3, CV_64F);
        K2(0,0) = f2; K2(0,2) = features2.imgsize.width * 0.5;
        K2(1,1) = f2; K2(1,2) = features2.imgsize.height * 0.5;

        cv::Mat_<double> H1 = R1_.inv() * K1.inv();
        cv::Mat_<double> H2 = R2_.inv() * K2.inv();

        for (size_t k = 0; k < matches_info.matches.size(); ++k) {
            if (!matches_info.inliers_mask[k])
                continue;
            const cv::DMatch& m = matches_info.matches[k];
            cv::Point2f p1 = features1.keypt[m.queryIdx].pt;
            double x1 = H1(0,0) * p1.x + H1(0,1) * p1.y + H1(0,2);
            double y1 = H1(1,0) * p1.x + H1(1,1) * p1.y + H1(1,2);
            double z1 = H1(2,0) * p1.x + H1(2,1) * p1.y + H1(2,2);
            double len = std::sqrt(x1 * x1 + y1 * y1 + z1 * z1);
            x1 /= len; y1 /= len; z1 /= len;

            cv::Point2f p2 = features2.keypt[m.trainIdx].pt;
            double x2 = H2(0,0) * p2.x + H2(0,1) * p2.y + H2(0,2);
            double y2 = H2(1,0) * p2.x + H2(1,1) * p2.y + H2(1,2);
            double z2 = H2(2,0) * p2.x + H2(2,1) * p2.y + H2(2,2);
            len = std::sqrt(x2*x2 + y2*y2 + z2*z2);
            x2 /= len; y2 /= len; z2 /= len;

            double mult = std::sqrt(f1 * f2);
            err.at<double>(3 * match_idx, 0) = mult * (x1 - x2);
            err.at<double>(3 * match_idx + 1, 0) = mult * (y1 - y2);
            err.at<double>(3 * match_idx + 2, 0) = mult * (z1 - z2);

            match_idx++;
        }
    }
	return 0;
}

/**
@brief calculate jacobian in bundle adjustment
@param cv::Mat & jac: output calculate jacobian matrix
@return int
*/
int calib::CameraParamEstimator::bundleAdjustCalcJacobian(cv::Mat & jac) {
	jac.create(totalMatchesNum * 3, cameras.size() * 4, CV_64F);
    double val;
    const double step = 1e-3;
    for (int i = 0; i < cameras.size(); i ++) {
        for (int j = 0; j < 4; j ++) {
            val = camParamBA.at<double>(i * 4 + j, 0);
            camParamBA.at<double>(i * 4 + j, 0) = val - step;
            bundleAdjustCalcError(err1_);
            camParamBA.at<double>(i * 4 + j, 0) = val + step;
            bundleAdjustCalcError(err2_);
			for (int k = 0; k < err1_.rows; k++)
        		jac.at<double>(k, i * 4 + j) = (err2_.at<double>(k, 0) 
				- err1_.at<double>(k, 0)) / (2 * step);
            camParamBA.at<double>(i * 4 + j, 0) = val;
        }
    }
	return 0;
}

/**
@brief estimate camera parameters
@return int
*/
int calib::CameraParamEstimator::estimate() {
	// estimate init camera parameters
	this->estimateInitCameraParams();
#ifdef HAVE_CERES_SOLVER
	calib::BundleAdjustment bundleAdjust;
	bundleAdjust.init(connection, features, cameras, matchesInfo);
	bundleAdjust.solve(1);
	this->cameras = bundleAdjust.getCameraParams();
#else
	this->bundleAdjustRefine();
#endif
	// wave correction
	std::vector<cv::Mat> rmats;
	for (size_t i = 0; i < cameras.size(); ++i) {
		rmats.push_back(cameras[i].R.clone());
		rmats[i].convertTo(rmats[i], CV_32F);
	}
	cv::detail::waveCorrect(rmats, cv::detail::WAVE_CORRECT_HORIZ);
	for (size_t i = 0; i < cameras.size(); ++i) {
		//cameras[i].R = rmats[i];
		rmats[i].convertTo(cameras[i].R, CV_64F);
	}
	return 0;
}

/**
@brief save calibration result to file
@param std::string filename: output filename
@return int
*/
int calib::CameraParamEstimator::saveCalibrationResult(std::string filename) {
	std::fstream fs(filename, std::ios::out);
	for (size_t i = 0; i < cameras.size(); i++) {
		fs << i << ' ';
		fs << cameras[i].focal << ' ';
		cv::Mat R = cameras[i].R.inv();
		for (size_t row = 0; row < R.rows; row++) {
			for (size_t col = 0; col < R.cols; col++) {
				fs << R.at<double>(row, col) << ' ';
			}
		}
		fs << std::endl;
	}
	fs.close();
	return 0;
}