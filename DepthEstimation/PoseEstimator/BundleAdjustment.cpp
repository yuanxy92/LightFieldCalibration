/**
@brief BundleAdjustment.h
C++ source file for bundle adjustment using ceres-solver
@author Shane Yuan
@date Feb 14, 2018
*/

#include "BundleAdjustment.h"

namespace calib {
	/**
	@define bundle adjustment reproj error
	*/
	struct BundleAdjustmentRaySingleFocalError {
		BundleAdjustmentRaySingleFocalError(double x1, double y1, double x2, double y2)
			: x1(x1), y1(y1), x2(x2), y2(y2) {}

		template <typename T>
		bool operator()(const T* const camera1, const T* const camera2, const T* focal,
			T* residuals) const {
			// points
			T point1[3];
			T point2[3];
			point1[0] = T(x1); point1[1] = T(y1); point1[2] = T(1.0f);
			point2[0] = T(x2); point2[1] = T(y2); point2[2] = T(1.0f);

			// change angle axis to rotation matrix
			T R1[9];
			T R2[9];
			ceres::AngleAxisToRotationMatrix(camera1, R1);
			ceres::AngleAxisToRotationMatrix(camera2, R2);

			// K1.inv() * (p1.x, p1.y, 1)'
			T tempx, tempy, tempz;
			point1[0] = point1[0] / focal[0];
			point1[1] = point1[1] / focal[0];
			point2[0] = point2[0] / focal[0];
			point2[1] = point2[1] / focal[0];

			// R1.inv() * K1.inv() * (p1.x, p1.y, 1)'  
			tempx = R1[0] * point1[0] + R1[3] * point1[1] + R1[6] * point1[2];
			tempy = R1[1] * point1[0] + R1[4] * point1[1] + R1[7] * point1[2];
			tempz = R1[2] * point1[0] + R1[5] * point1[1] + R1[8] * point1[2];
			point1[0] = tempx;
			point1[1] = tempy;
			point1[2] = tempz;
			tempx = R2[0] * point2[0] + R2[3] * point2[1] + R2[6] * point2[2];
			tempy = R2[1] * point2[0] + R2[4] * point2[1] + R2[7] * point2[2];
			tempz = R2[2] * point2[0] + R2[5] * point2[1] + R2[8] * point2[2];
			point2[0] = tempx;
			point2[1] = tempy;
			point2[2] = tempz;

			// normalize
			T l1 = sqrt(point1[0] * point1[0] + point1[1] * point1[1] + point1[2] * point1[2]);
			T l2 = sqrt(point2[0] * point2[0] + point2[1] * point2[1] + point2[2] * point2[2]);
			point1[0] /= l1;
			point1[1] /= l1;
			point1[2] /= l1;
			point2[0] /= l2;
			point2[1] /= l2;
			point2[2] /= l2;
			T noramlFocal = sqrt(focal[0] * focal[0]);

			// calculate residuals
			residuals[0] = (point1[0] - point2[0]) * focal[0];
			residuals[1] = (point1[1] - point2[1]) * focal[0];
			residuals[2] = (point1[2] - point2[2]) * focal[0];

			return true;
		}

		// Factory to hide the construction of the CostFunction object from
		// the client code.
		static ceres::CostFunction* Create(const double x1, const double y1,
			const double x2, const double y2) {
			return (new ceres::AutoDiffCostFunction<BundleAdjustmentRaySingleFocalError, 3, 3, 3, 1>(
				new BundleAdjustmentRaySingleFocalError(x1, y1, x2, y2)));
		}

		double x1;
		double y1;
		double x2;
		double y2;
	};

    /**
    @define bundle adjustment reproj error
    */
    struct BundleAdjustmentRayError {
        BundleAdjustmentRayError(double x1, double y1, double x2, double y2)
        : x1(x1), y1(y1), x2(x2), y2(y2) {}

        template <typename T>
        bool operator()(const T* const camera1, const T* const camera2,
            T* residuals) const {
            // points
            T point1[3];
            T point2[3];
            point1[0] = T(x1); point1[1] = T(y1); point1[2] = T(1.0f);
            point2[0] = T(x2); point2[1] = T(y2); point2[2] = T(1.0f);

            // change angle axis to rotation matrix
            T R1[9];
            T R2[9];
            ceres::AngleAxisToRotationMatrix(camera1, R1); 
            ceres::AngleAxisToRotationMatrix(camera2, R2);

			// calculate R1.inv() * K1.inv() 
			T tempx, tempy, tempz;
			tempx = (T(1.0f) / camera1[3]) * point1[0] + T(0.0f) * point1[1] + T(0.0f) * point1[2];
			tempy = T(0.0f) * point1[0] + (T(1.0f) / camera1[3]) * point1[1] + T(0.0f) * point1[2];
			tempz = T(0.0f) * point1[0] + T(0.0f) * point1[1] + T(1.0f) * point1[2];
			point1[0] = tempx;
			point1[1] = tempy;
			point1[2] = tempz;

			tempx = (T(1.0f) / camera2[3]) * point2[0] + T(0.0f) * point2[1] + T(0.0f) * point2[2];
			tempy = T(0.0f) * point2[0] + (T(1.0f) / camera2[3]) * point2[1] + T(0.0f) * point2[2];
			tempz = T(0.0f) * point2[0] + T(0.0f) * point2[1] + T(1.0f) * point2[2];
			point2[0] = tempx;
			point2[1] = tempy;
			point2[2] = tempz;

			tempx = R1[0] * point1[0] + R1[3] * point1[1] + R1[6] * point1[2];
			tempy = R1[1] * point1[0] + R1[4] * point1[1] + R1[7] * point1[2];
			tempz = R1[2] * point1[0] + R1[5] * point1[1] + R1[8] * point1[2];
			point1[0] = tempx;
			point1[1] = tempy;
			point1[2] = tempz;

			tempx = R2[0] * point2[0] + R2[3] * point2[1] + R2[6] * point2[2];
			tempy = R2[1] * point2[0] + R2[4] * point2[1] + R2[7] * point2[2];
			tempz = R2[2] * point2[0] + R2[5] * point2[1] + R2[8] * point2[2];
			point2[0] = tempx;
			point2[1] = tempy;
			point2[2] = tempz;

            // normalize
            T l1 = sqrt(point1[0] * point1[0] + point1[1] * point1[1] + point1[2] * point1[2]);
            T l2 = sqrt(point2[0] * point2[0] + point2[1] * point2[1] + point2[2] * point2[2]);
            point1[0] /= l1;
            point1[1] /= l1;
            point1[2] /= l1;
            point2[0] /= l2;
            point2[1] /= l2;
            point2[2] /= l2;
            T noramlFocal = sqrt(camera1[3] * camera2[3]);

            // calculate residuals
            residuals[0] = (point1[0] - point2[0]) * noramlFocal;
            residuals[1] = (point1[1] - point2[1]) * noramlFocal;
            residuals[2] = (point1[2] - point2[2]) * noramlFocal;

            return true;
        }

        // Factory to hide the construction of the CostFunction object from
        // the client code.
        static ceres::CostFunction* Create(const double x1, const double y1,
            const double x2, const double y2) {
            return (new ceres::AutoDiffCostFunction<BundleAdjustmentRayError, 3, 4, 4>(
                new BundleAdjustmentRayError(x1, y1, x2, y2)));
        }

        double x1;
        double y1;
        double x2;
        double y2;
    };

	/**
	@define bundle adjustment reproj error (single camera focal length setting)
	*/
	struct BundleAdjustmentReprojSingeFocalError {
		BundleAdjustmentReprojSingeFocalError(double x1, double y1, double x2, double y2)
			: x1(x1), y1(y1), x2(x2), y2(y2) {}

		template <typename T>
		bool operator()(const T* const camera1, const T* const camera2, const T* focal,
			T* residuals) const {
			// points
			T point1[3];
			point1[0] = T(x1); point1[1] = T(y1); point1[2] = T(1.0f);

			// change angle axis to rotation matrix
			T R1[9];
			T R2[9];
			ceres::AngleAxisToRotationMatrix(camera1, R1);
			ceres::AngleAxisToRotationMatrix(camera2, R2);
			
			T tempx, tempy, tempz;
			// K1.inv() * (p1.x, p1.y, 1)'
			point1[0] = point1[0] / focal[0];
			point1[1] = point1[1] / focal[0];

			// R1.inv() * K1.inv() * (p1.x, p1.y, 1)'  
			tempx = R1[0] * point1[0] + R1[3] * point1[1] + R1[6] * point1[2];
			tempy = R1[1] * point1[0] + R1[4] * point1[1] + R1[7] * point1[2];
			tempz = R1[2] * point1[0] + R1[5] * point1[1] + R1[8] * point1[2];
			point1[0] = tempx;
			point1[1] = tempy;
			point1[2] = tempz;

			// R2 * R1.inv() * K1.inv() * (p1.x, p1.y, 1)'  
			tempx = R2[0] * point1[0] + R2[1] * point1[1] + R2[2] * point1[2];
			tempy = R2[3] * point1[0] + R2[4] * point1[1] + R2[5] * point1[2];
			tempz = R2[6] * point1[0] + R2[7] * point1[1] + R2[8] * point1[2];
			point1[0] = tempx;
			point1[1] = tempy;
			point1[2] = tempz;

			// K2 * R1.inv() * K1.inv() * (p1.x, p1.y, 1)'   
			tempx = point1[0] * focal[0] / point1[2];
			tempy = point1[1] * focal[0] / point1[2];
			tempz = T(1.0f);
			point1[0] = tempx;
			point1[1] = tempy;
			point1[2] = tempz;

			// calculate residuals
			residuals[0] = point1[0] - T(x2);
			residuals[1] = point1[1] - T(y2);

			return true;
		}

		// Factory to hide the construction of the CostFunction object from
		// the client code.
		static ceres::CostFunction* Create(const double x1, const double y1,
			const double x2, const double y2) {
			return (new ceres::AutoDiffCostFunction<BundleAdjustmentReprojSingeFocalError, 2, 3, 3, 1>(
				new BundleAdjustmentReprojSingeFocalError(x1, y1, x2, y2)));
		}

		double x1;
		double y1;
		double x2;
		double y2;
	};

    /**
    @define bundle adjustment reproj error
    */
    struct BundleAdjustmentReprojError {
        BundleAdjustmentReprojError(double x1, double y1, double x2, double y2)
        : x1(x1), y1(y1), x2(x2), y2(y2) {}

        template <typename T>
        bool operator()(const T* const camera1, const T* const camera2,
            T* residuals) const {
            // points
            T point1[3];
            point1[0] = T(x1); point1[1] = T(y1); point1[2] = T(1.0f);

            // change angle axis to rotation matrix
            T R1[9];
            T R2[9];
            ceres::AngleAxisToRotationMatrix(camera1, R1); 
            ceres::AngleAxisToRotationMatrix(camera2, R2);

            // K1.inv() * (p1.x, p1.y, 1)'
			T tempx, tempy, tempz;
            point1[0] = point1[0] / camera1[3];
            point1[1] = point1[1] / camera1[3];

            // R1.inv() * K1.inv() * (p1.x, p1.y, 1)'  
			tempx = R1[0] * point1[0] + R1[3] * point1[1] + R1[6] * point1[2];
			tempy = R1[1] * point1[0] + R1[4] * point1[1] + R1[7] * point1[2];
			tempz = R1[2] * point1[0] + R1[5] * point1[1] + R1[8] * point1[2];
			point1[0] = tempx;
			point1[1] = tempy;
			point1[2] = tempz;

            // R2 * R1.inv() * K1.inv() * (p1.x, p1.y, 1)'  
			tempx = R2[0] * point1[0] + R2[1] * point1[1] + R2[2] * point1[2];
			tempy = R2[3] * point1[0] + R2[4] * point1[1] + R2[5] * point1[2];
			tempz = R2[6] * point1[0] + R2[7] * point1[1] + R2[8] * point1[2];
			point1[0] = tempx;
			point1[1] = tempy;
			point1[2] = tempz;

            // K2 * R1.inv() * K1.inv() * (p1.x, p1.y, 1)'   
            tempx = point1[0] * camera2[3] / point1[2]; 
            tempy = point1[1] * camera2[3] / point1[2]; 
            tempz = T(1.0f);
			point1[0] = tempx;
			point1[1] = tempy;
			point1[2] = tempz;

            // calculate residuals
            residuals[0] = point1[0] - T(x2);
            residuals[1] = point1[1] - T(y2); 

            return true;
        }

        // Factory to hide the construction of the CostFunction object from
        // the client code.
        static ceres::CostFunction* Create(const double x1, const double y1,
            const double x2, const double y2) {
            return (new ceres::AutoDiffCostFunction<BundleAdjustmentReprojError, 2, 4, 4>(
                new BundleAdjustmentReprojError(x1, y1, x2, y2)));
        }

        double x1;
        double y1;
        double x2;
        double y2;
    };
}

calib::BundleAdjustment::BundleAdjustment() {}
calib::BundleAdjustment::~BundleAdjustment() {
    if (cameraParam)
        delete[] cameraParam; 
}

/**
@brief set input for camera parameter estimation
@param cv::Mat connection: input connection matrix
@param std::vector<Imagefeature> features: input matching points features
@param std::vector<CameraParams> cameras: input cameras
@param std::vector<Matchesinfo> matchesInfo: input matching infos
@return int
*/
int calib::BundleAdjustment::init(cv::Mat connection,
    std::vector<Imagefeature> features,
    std::vector<CameraParams> cameras,
	std::vector<Matchesinfo> matchesInfo) {
    this->connection = connection;
    this->features = features;
    this->cameras = cameras;
    this->matchesInfo = matchesInfo;
    return 0;
}

/**
@brief init camera parameters
@return int
*/
int calib::BundleAdjustment::initCameraParamMultipleFocal() {
    int cameraNum = cameras.size();
    cameraParam = new double[cameraNum * 4];
	cv::SVD svd;
    for (size_t i = 0; i < cameraNum; i ++) {
        svd(cameras[i].R, cv::SVD::FULL_UV);
        cv::Mat R = svd.u * svd.vt;
		R = R.t();
        if (determinant(R) < 0)
            R *= -1;
        double R_[9];
        double Rvec_[3];
        R_[0] = R.at<double>(0, 0); R_[1] = R.at<double>(0, 1); R_[2] = R.at<double>(0, 2);
        R_[3] = R.at<double>(1, 0); R_[4] = R.at<double>(1, 1); R_[5] = R.at<double>(1, 2);
        R_[6] = R.at<double>(2, 0); R_[7] = R.at<double>(2, 1); R_[8] = R.at<double>(2, 2);
        ceres::RotationMatrixToAngleAxis(R_, Rvec_); 
        cameraParam[i * 4 + 0] = Rvec_[0];
        cameraParam[i * 4 + 1] = Rvec_[1];
        cameraParam[i * 4 + 2] = Rvec_[2];
        cameraParam[i * 4 + 3] = cameras[i].focal;;
    }
    return 0;
}

/**
@brief init camera parameters
@return int
*/
int calib::BundleAdjustment::initCameraParamSingleFocal() {
	int cameraNum = cameras.size();
	cameraParam = new double[cameraNum * 3 + 1];
	cv::SVD svd;
	for (size_t i = 0; i < cameraNum; i++) {
		svd(cameras[i].R, cv::SVD::FULL_UV);
		cv::Mat R = svd.u * svd.vt;
		R = R.t();
		if (determinant(R) < 0)
			R *= -1;
		double R_[9];
		double Rvec_[3];
		R_[0] = R.at<double>(0, 0); R_[1] = R.at<double>(0, 1); R_[2] = R.at<double>(0, 2);
		R_[3] = R.at<double>(1, 0); R_[4] = R.at<double>(1, 1); R_[5] = R.at<double>(1, 2);
		R_[6] = R.at<double>(2, 0); R_[7] = R.at<double>(2, 1); R_[8] = R.at<double>(2, 2);
		ceres::RotationMatrixToAngleAxis(R_, Rvec_);
		cameraParam[i * 3 + 0] = Rvec_[0];
		cameraParam[i * 3 + 1] = Rvec_[1];
		cameraParam[i * 3 + 2] = Rvec_[2];
	}
	cameraParam[cameraNum * 3] = cameras[0].focal;;
	return 0;
}

/**
@brief function to solve bundle adjustment problem
@return int
*/
int calib::BundleAdjustment::solveMultipleFocal() {
    // init camera param
    this->initCameraParamMultipleFocal();
    // get edges
    int totalMatchesNum = 0;
	std::vector<std::pair<size_t, size_t>> edgesBA;
	std::vector<size_t> edgesIndBA;
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
    // Create residuals for each observation in the bundle adjustment problem. The
    // parameters for cameras and points are added automatically.
    ceres::Problem problem;
    for (size_t edge_idx = 0; edge_idx < edgesBA.size(); edge_idx ++) {
        int i = edgesBA[edge_idx].first;
        int j = edgesBA[edge_idx].second;
        const calib::Imagefeature& features1 = features[i];
        const calib::Imagefeature& features2 = features[j];
        const calib::Matchesinfo& matches_info = matchesInfo[edgesIndBA[edge_idx]];
        for (size_t k = 0; k < matches_info.matches.size(); ++k) {
            if (!matches_info.inliers_mask[k])
                continue;
            const cv::DMatch& m = matches_info.matches[k];
            cv::Point2f p1 = features1.keypt[m.queryIdx].pt;
            cv::Point2f p2 = features2.keypt[m.trainIdx].pt;
            // Each Residual block takes a point and a camera as input and outputs a 2
            // dimensional residual. Internally, the cost function stores the observed
            // image location and compares the reprojection against the observation.
            ceres::CostFunction* cost_function =
                calib::BundleAdjustmentRayError::Create(p1.x - features1.imgsize.width / 2,
					p1.y - features1.imgsize.height / 2,
					p2.x - features2.imgsize.width / 2,
					p2.y - features2.imgsize.height / 2);
            problem.AddResidualBlock(cost_function,
                            NULL /* squared loss */,
                            &cameraParam[i * 4],
                            &cameraParam[j * 4]);
        }
    }
    // solve problem
    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
	options.jacobi_scaling = false;
	options.max_num_iterations = 2000;
	options.function_tolerance = 1e-8;
	options.parameter_tolerance = 1e-8;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";

    // update cameras
	for (size_t i = 0; i < cameras.size(); i ++) {
        cameras[i].focal = cameraParam[i * 4 + 3];
        double R_[9];
        double Rvec_[3];
        Rvec_[0] = cameraParam[i * 4 + 0];
        Rvec_[1] = cameraParam[i * 4 + 1];
        Rvec_[2] = cameraParam[i * 4 + 2];
        ceres::AngleAxisToRotationMatrix(Rvec_, R_);  
        cameras[i].R.at<double>(0, 0) = R_[0];  cameras[i].R.at<double>(0, 1) = R_[1];   cameras[i].R.at<double>(0, 2) = R_[2];     
        cameras[i].R.at<double>(1, 0) = R_[3];  cameras[i].R.at<double>(1, 1) = R_[4];   cameras[i].R.at<double>(1, 2) = R_[5];     
        cameras[i].R.at<double>(2, 0) = R_[6];  cameras[i].R.at<double>(2, 1) = R_[7];   cameras[i].R.at<double>(2, 2) = R_[8];     
		cameras[i].R = cameras[i].R.t();
    }
    return 0;
}

/**
@brief function to solve bundle adjustment problem
@return int
*/
int calib::BundleAdjustment::solveSingleFocal() {
	// init camera param
	this->initCameraParamSingleFocal();
	// get edges
	int totalMatchesNum = 0;
	std::vector<std::pair<size_t, size_t>> edgesBA;
	std::vector<size_t> edgesIndBA;
	std::vector<std::pair<size_t, size_t>> pairs = Connection::getConnections(connection);
	edgesBA.clear();
	edgesIndBA.clear();
	size_t connectionNum = pairs.size();
	for (size_t i = 0; i < connectionNum; i++) {
		if (matchesInfo[i].confidence > 1.5) {
			totalMatchesNum += matchesInfo[i].num_inliers;
			edgesBA.push_back(pairs[i]);
			edgesIndBA.push_back(i);
		}
	}
	// Create residuals for each observation in the bundle adjustment problem. The
	// parameters for cameras and points are added automatically.
	ceres::Problem problem;
	for (size_t edge_idx = 0; edge_idx < edgesBA.size(); edge_idx++) {
		int i = edgesBA[edge_idx].first;
		int j = edgesBA[edge_idx].second;
		const calib::Imagefeature& features1 = features[i];
		const calib::Imagefeature& features2 = features[j];
		const calib::Matchesinfo& matches_info = matchesInfo[edgesIndBA[edge_idx]];
		for (size_t k = 0; k < matches_info.matches.size(); ++k) {
			if (!matches_info.inliers_mask[k])
				continue;
			const cv::DMatch& m = matches_info.matches[k];
			cv::Point2f p1 = features1.keypt[m.queryIdx].pt;
			cv::Point2f p2 = features2.keypt[m.trainIdx].pt;
			// Each Residual block takes a point and a camera as input and outputs a 2
			// dimensional residual. Internally, the cost function stores the observed
			// image location and compares the reprojection against the observation.
			ceres::CostFunction* cost_function =
				calib::BundleAdjustmentRaySingleFocalError::Create(p1.x - features1.imgsize.width / 2,
					p1.y - features1.imgsize.height / 2,
					p2.x - features2.imgsize.width / 2,
					p2.y - features2.imgsize.height / 2);
			problem.AddResidualBlock(cost_function,
				NULL /* squared loss */,
				&cameraParam[i * 3],
				&cameraParam[j * 3],
				&cameraParam[cameras.size() * 3]);
		}
	}
	// solve problem
	// Make Ceres automatically detect the bundle structure. Note that the
	// standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
	// for standard bundle adjustment problems.
	ceres::Solver::Options options;
	options.linear_solver_type = ceres::SPARSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	options.jacobi_scaling = false;
	options.max_num_iterations = 2000;
	options.function_tolerance = 1e-8;
	options.parameter_tolerance = 1e-8;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n";

	// update cameras
	for (size_t i = 0; i < cameras.size(); i++) {
		cameras[i].focal = cameraParam[cameras.size() * 3];
		double R_[9];
		double Rvec_[3];
		Rvec_[0] = cameraParam[i * 3 + 0];
		Rvec_[1] = cameraParam[i * 3 + 1];
		Rvec_[2] = cameraParam[i * 3 + 2];
		ceres::AngleAxisToRotationMatrix(Rvec_, R_);
		cameras[i].R.at<double>(0, 0) = R_[0];  cameras[i].R.at<double>(0, 1) = R_[1];   cameras[i].R.at<double>(0, 2) = R_[2];
		cameras[i].R.at<double>(1, 0) = R_[3];  cameras[i].R.at<double>(1, 1) = R_[4];   cameras[i].R.at<double>(1, 2) = R_[5];
		cameras[i].R.at<double>(2, 0) = R_[6];  cameras[i].R.at<double>(2, 1) = R_[7];   cameras[i].R.at<double>(2, 2) = R_[8];
		cameras[i].R = cameras[i].R.inv();
	}
	return 0;
}

/**
@brief get estimated camera params
@return std::vector<CameraParams>: returned camera params 
*/
std::vector<calib::CameraParams> calib::BundleAdjustment::getCameraParams() {
    return cameras;
}

/**
@brief function to solve bundle adjustment problem
@param int focalSetting : input focal length
0 : use multiple focal length (every camera has its own focal length)
1 : use single focal length (every camera use the same focal length)
@return int
*/
int calib::BundleAdjustment::solve(int focalSetting) {
	if (focalSetting == 0) {
		this->solveMultipleFocal();
	}
	else {
		this->solveSingleFocal();
	}
	return 0;
}
