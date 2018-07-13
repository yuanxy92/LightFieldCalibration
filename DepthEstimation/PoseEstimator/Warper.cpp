/**
@brief warper class, warping function
@author Shane Yuan
@date Jun 1, 2017
*/

#include "Warper.h"

namespace calib {

	SphericalWarper::SphericalWarper() { this->scale = 3500.0f; }
	SphericalWarper::~SphericalWarper() {}

	/**
	@brief set scale
	@param float: camera scale
	@return int: ErrorCode
	*/
	int SphericalWarper::setScale(float scale) {
		this->scale = scale;
		return 0;
	}

	/**
	@brief set camera parameters
	@param cv::Mat K: intrinsic matrix
	@param cv::Mat R: rotation matrix
	@param cv::Mat T: translation matrix
	@return int: ErrorCode
	*/
	int SphericalWarper::setCameraParams(cv::Mat K, cv::Mat R, cv::Mat T) {

		CV_Assert(K.size() == cv::Size(3, 3) && K.type() == CV_32F);
		CV_Assert(R.size() == cv::Size(3, 3) && R.type() == CV_32F);
		CV_Assert((T.size() == cv::Size(1, 3) || T.size() == cv::Size(3, 1)) && T.type() == CV_32F);

		cv::Mat_<float> K_(K);
		k[0] = K_(0, 0); k[1] = K_(0, 1); k[2] = K_(0, 2);
		k[3] = K_(1, 0); k[4] = K_(1, 1); k[5] = K_(1, 2);
		k[6] = K_(2, 0); k[7] = K_(2, 1); k[8] = K_(2, 2);

		cv::Mat_<float> Rinv = R.t();
		rinv[0] = Rinv(0, 0); rinv[1] = Rinv(0, 1); rinv[2] = Rinv(0, 2);
		rinv[3] = Rinv(1, 0); rinv[4] = Rinv(1, 1); rinv[5] = Rinv(1, 2);
		rinv[6] = Rinv(2, 0); rinv[7] = Rinv(2, 1); rinv[8] = Rinv(2, 2);

		cv::Mat_<float> R_Kinv = R * K.inv();
		r_kinv[0] = R_Kinv(0, 0); r_kinv[1] = R_Kinv(0, 1); r_kinv[2] = R_Kinv(0, 2);
		r_kinv[3] = R_Kinv(1, 0); r_kinv[4] = R_Kinv(1, 1); r_kinv[5] = R_Kinv(1, 2);
		r_kinv[6] = R_Kinv(2, 0); r_kinv[7] = R_Kinv(2, 1); r_kinv[8] = R_Kinv(2, 2);

		cv::Mat_<float> K_Rinv = K * Rinv;
		k_rinv[0] = K_Rinv(0, 0); k_rinv[1] = K_Rinv(0, 1); k_rinv[2] = K_Rinv(0, 2);
		k_rinv[3] = K_Rinv(1, 0); k_rinv[4] = K_Rinv(1, 1); k_rinv[5] = K_Rinv(1, 2);
		k_rinv[6] = K_Rinv(2, 0); k_rinv[7] = K_Rinv(2, 1); k_rinv[8] = K_Rinv(2, 2);

		cv::Mat_<float> T_(T.reshape(0, 3));
		t[0] = T_(0, 0); t[1] = T_(1, 0); t[2] = T_(2, 0);
		return 0;
	}
	int SphericalWarper::setCameraParams(cv::Mat K, cv::Mat R) {
		cv::Mat T;
		T.create(3, 1, CV_32F);
		T.at<float>(0, 0) = 0; T.at<float>(1, 0) = 0; T.at<float>(2, 0) = 0;
		return this->setCameraParams(K, R, T);
	}

	/**
	@brief map points forward
	@param float x: input point coordinate x
	@param float y: input point coordinate y
	@param float u: output point coordinate x
	@param float v: output point coordinate y
	@return int: ErrorCode
	*/
	int SphericalWarper::mapForward(float x, float y, float& u, float& v) {
		float x_ = r_kinv[0] * x + r_kinv[1] * y + r_kinv[2];
		float y_ = r_kinv[3] * x + r_kinv[4] * y + r_kinv[5];
		float z_ = r_kinv[6] * x + r_kinv[7] * y + r_kinv[8];

		u = scale * atan2f(x_, z_);
		float w = y_ / sqrtf(x_ * x_ + y_ * y_ + z_ * z_);
		v = scale * (static_cast<float>(CV_PI) - acosf(w == w ? w : 0));
		return 0;
	}

	/**
	@brief map points backward
	@param float u: input point coordinate x
	@param float v: input point coordinate y
	@param float x: output point coordinate x
	@param float y: output point coordinate y
	@return int: ErrorCode
	*/
	int SphericalWarper::mapBackward(float u, float v, float& x, float& y) {
		u /= scale;
		v /= scale;

		float sinv = sinf(static_cast<float>(CV_PI) - v);
		float x_ = sinv * sinf(u);
		float y_ = cosf(static_cast<float>(CV_PI) - v);
		float z_ = sinv * cosf(u);

		float z;
		x = k_rinv[0] * x_ + k_rinv[1] * y_ + k_rinv[2] * z_;
		y = k_rinv[3] * x_ + k_rinv[4] * y_ + k_rinv[5] * z_;
		z = k_rinv[6] * x_ + k_rinv[7] * y_ + k_rinv[8] * z_;

		if (z > 0) { x /= z; y /= z; }
		else x = y = -1;
		return 0;
	}

	/**
	@brief detect regions of interest on result
	@param cv::Size srcSize: input image size
	@param cv::Point &dstTL: output top left corner coordinate
	@param cv::Point &dstBR: output bottom right corner coordinate
	@return int: ErrorCode
	*/
	int SphericalWarper::detectResultRoi(cv::Size srcSize, cv::Point &dstTL, cv::Point &dstBR) {
		float tl_uf = (std::numeric_limits<float>::max)();
		float tl_vf = (std::numeric_limits<float>::max)();
		float br_uf = -(std::numeric_limits<float>::max)();
		float br_vf = -(std::numeric_limits<float>::max)();
		float u, v;
		for (int y = 0; y < srcSize.height; ++y) {
			for (int x = 0; x < srcSize.width; ++x) {
				this->mapForward(static_cast<float>(x), static_cast<float>(y), u, v);
				tl_uf = (std::min)(tl_uf, u); tl_vf = (std::min)(tl_vf, v);
				br_uf = (std::max)(br_uf, u); br_vf = (std::max)(br_vf, v);
			}
		}
		dstTL.x = static_cast<int>(tl_uf);
		dstTL.y = static_cast<int>(tl_vf);
		dstBR.x = static_cast<int>(br_uf);
		dstBR.y = static_cast<int>(br_vf);

		this->outRect = cv::Rect(dstTL, dstBR);

		return 0;
	}

	/**
	@brief build correspondence maps forward
	@param cv::Size srcSize: input image size
	@param cv::Mat& xmap: output mapping field (x)
	@param cv::Mat& ymap: output mapping field (y)
	@return int: ErrorCode
	*/
	cv::Rect SphericalWarper::buildMapsForward(cv::Size srcSize, cv::Mat& xmap, cv::Mat& ymap) {
		// detect result ROI
		cv::Point dstTL, dstBR;
		this->detectResultRoi(srcSize, dstTL, dstBR);
		cv::Size outSize = cv::Size(static_cast<int>(dstBR.x - dstTL.x) + 1,
			static_cast<int>(dstBR.y - dstTL.y) + 1);
		xmap.create(outSize, CV_32F);
		ymap.create(outSize, CV_32F);
		// compute per-pixel correspondence
		float x, y;
		for (int v = dstTL.y; v <= dstBR.y; v++) {
			for (int u = dstTL.x; u <= dstBR.x; u++) {
				this->mapBackward(static_cast<float>(u), static_cast<float>(v), x, y);
				xmap.at<float>(v - dstTL.y, u - dstTL.x) = x;
				ymap.at<float>(v - dstTL.y, u - dstTL.x) = y;
			}
		}
		return cv::Rect(dstTL, dstBR);
	}
	cv::Rect SphericalWarper::buildMapsForward(cv::Size srcSize, cv::Mat K, cv::Mat R, cv::Mat& xmap, cv::Mat& ymap) {
		this->setCameraParams(K, R);
		return buildMapsForward(srcSize, xmap, ymap);
	}


	/**
	@brief build correspondence maps backward
	@param cv::Size dstSize: dstination image size
	@param cv::Mat& xmap: output mapping field (x)
	@param cv::Mat& ymap: output mapping field (y)
	@return int: ErrorCode
	*/
	cv::Rect SphericalWarper::buildMapsBackward(cv::Size dstSize, cv::Mat& xmap, cv::Mat& ymap) {
		// detect result ROI
		cv::Point srcTL, srcBR;
		this->detectResultRoi(dstSize, srcTL, srcBR);
		xmap.create(dstSize, CV_32F);
		ymap.create(dstSize, CV_32F);
		// compute per-pixel correspondence
		float u, v;
		for (int y = 0; y < dstSize.height; y++) {
			for (int x = 0; x < dstSize.width; x++) {
				this->mapForward(static_cast<float>(x), static_cast<float>(y), u, v);
				xmap.at<float>(y, x) = u - srcTL.x;
				ymap.at<float>(y, x) = v - srcTL.y;
			}
		}
		return cv::Rect(srcTL, srcBR);
	}
	cv::Rect SphericalWarper::buildMapsBackward(cv::Size dstSize, cv::Mat K, cv::Mat R, cv::Mat& xmap, cv::Mat& ymap) {
		this->setCameraParams(K, R);
		return buildMapsBackward(dstSize, xmap, ymap);
	}

	/**
	@brief build correspondence maps forward
	@param cv::Mat src: input source image
	@param cv::Mat& dst: output target image
	@param int interMode: interploation mode
	@return int: ErrorCode
	*/
	cv::Point SphericalWarper::warpForward(cv::Mat src, cv::Mat& dst, int interMode) {
		cv::Mat xmap, ymap;
		cv::Rect dstROI = buildMapsForward(src.size(), xmap, ymap);
		dst.create(dstROI.height + 1, dstROI.width + 1, src.type());
		cv::remap(src, dst, xmap, ymap, interMode, cv::BORDER_CONSTANT);
		return dstROI.tl();
	}
	cv::Point SphericalWarper::warpForward(cv::Mat src, cv::Mat K, cv::Mat R, int interMode, int borderMode, cv::Mat& dst) {
		this->setCameraParams(K, R);
		return this->warpForward(src, dst, interMode);
	}
	cv::Point SphericalWarper::warpForward(cv::UMat src, cv::Mat K, cv::Mat R, int interMode, int borderMode, cv::UMat& dst) {
		cv::Mat xmap, ymap;
		this->setCameraParams(K, R);
		cv::Rect dstROI = buildMapsForward(src.size(), xmap, ymap);
		dst.create(dstROI.height + 1, dstROI.width + 1, src.type());
		cv::remap(src, dst, xmap, ymap, interMode, cv::BORDER_CONSTANT);
		//return dstROI.tl();
		return cv::Point(dstROI.x, dstROI.y);
	}

	/**
	@brief build correspondence maps backford
	@param cv::Mat src: input source image
	@param cv::Size dstSize: destination image size
	@param cv::Mat& dst: output target image
	@param int interMode: interploation mode
	@return int: ErrorCode
	*/
	cv::Point SphericalWarper::warpBackward(cv::Mat src, cv::Size dstSize, cv::Mat& dst, int interMode) {
		cv::Mat xmap, ymap;
		cv::Rect dstROI;
		dstROI = buildMapsBackward(dstSize, xmap, ymap);
		dst.create(dstSize, src.type());
		cv::remap(src, dst, xmap, ymap, interMode, cv::BORDER_CONSTANT);
		return dstROI.tl();
	}

};