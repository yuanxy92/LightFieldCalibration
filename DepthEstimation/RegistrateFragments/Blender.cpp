/**
@brief blender class fusion depth map to get 
    better 3d model
@author Shane Yuan
@date May 31, 2018
*/

#include "Blender.h"

calib::Blender::Blender() : weight_type_(CV_32F), num_bands_(5),
	actual_num_bands_(5), weight_eps(1e-5f) {}
calib::Blender::~Blender() {}

cv::Rect calib::Blender::resultRoi(const std::vector<cv::Point> &corners,
	const std::vector<cv::Size> &sizes) {
	CV_Assert(sizes.size() == corners.size());
	cv::Point tl(std::numeric_limits<int>::max(), std::numeric_limits<int>::max());
	cv::Point br(std::numeric_limits<int>::min(), std::numeric_limits<int>::min());
	for (size_t i = 0; i < corners.size(); ++i)
	{
		tl.x = std::min(tl.x, corners[i].x);
		tl.y = std::min(tl.y, corners[i].y);
		br.x = std::max(br.x, corners[i].x + sizes[i].width);
		br.y = std::max(br.y, corners[i].y + sizes[i].height);
	}
	return cv::Rect(tl, br);
}

int calib::Blender::numBands() const {
	return actual_num_bands_;
}
void calib::Blender::setNumBands(int val) {
	actual_num_bands_ = val; 
}

void calib::Blender::prepare(cv::Rect dst_roi) {
	dst_roi_final_ = dst_roi;

	// Crop unnecessary bands
	double max_len = static_cast<double>(std::max(dst_roi.width, dst_roi.height));
	num_bands_ = std::min(actual_num_bands_, static_cast<int>(ceil(std::log(max_len) / std::log(2.0))));

	// Add border to the final image, to ensure sizes are divided by (1 << num_bands_)
	dst_roi.width += ((1 << num_bands_) - dst_roi.width % (1 << num_bands_)) % (1 << num_bands_);
	dst_roi.height += ((1 << num_bands_) - dst_roi.height % (1 << num_bands_)) % (1 << num_bands_);

	dst_.create(dst_roi.size(), CV_32FC3);
	dst_.setTo(cv::Scalar::all(0));
	dst_mask_.create(dst_roi.size(), CV_8U);
	dst_mask_.setTo(cv::Scalar::all(0));
	dst_roi_ = dst_roi;

	dst_pyr_laplace_.resize(num_bands_ + 1);
	dst_pyr_laplace_[0] = dst_;

	dst_band_weights_.resize(num_bands_ + 1);
	dst_band_weights_[0].create(dst_roi.size(), weight_type_);
	dst_band_weights_[0].setTo(0);

	for (int i = 1; i <= num_bands_; i++) {
		dst_pyr_laplace_[i].create((dst_pyr_laplace_[i - 1].rows + 1) / 2,
			(dst_pyr_laplace_[i - 1].cols + 1) / 2, CV_32FC3);
		dst_band_weights_[i].create((dst_band_weights_[i - 1].rows + 1) / 2,
			(dst_band_weights_[i - 1].cols + 1) / 2, weight_type_);
		dst_pyr_laplace_[i].setTo(cv::Scalar::all(0));
		dst_band_weights_[i].setTo(0);
	}
}

void calib::Blender::prepare(const std::vector<cv::Point> &corners,
	const std::vector<cv::Size> &sizes) {
	this->prepare(resultRoi(corners, sizes));
}

void calib::Blender::createLaplacePyr(cv::InputArray img, int num_levels, std::vector<cv::UMat> &pyr) {
	pyr.resize(num_levels + 1);
	pyr[0] = img.getUMat();
	for (int i = 0; i < num_levels; ++i)
		cv::pyrDown(pyr[i], pyr[i + 1]);
	cv::UMat tmp;
	for (int i = 0; i < num_levels; ++i) {
		cv::pyrUp(pyr[i + 1], tmp, pyr[i].size());
		cv::subtract(pyr[i], tmp, pyr[i]);
	}
}


void calib::Blender::feed(cv::InputArray _img, cv::InputArray mask, cv::Point tl) {
	cv::UMat img;
	img = _img.getUMat();
	// Keep source image in memory with small border
	int gap = 3 * (1 << num_bands_);
	cv::Point tl_new(std::max(dst_roi_.x, tl.x - gap),
		std::max(dst_roi_.y, tl.y - gap));
	cv::Point br_new(std::min(dst_roi_.br().x, tl.x + img.cols + gap),
		std::min(dst_roi_.br().y, tl.y + img.rows + gap));
	// Ensure coordinates of top-left, bottom-right corners are divided by (1 << num_bands_).
	// After that scale between layers is exactly 2.
	//
	// We do it to avoid interpolation problems when keeping sub-images only. There is no such problem when
	// image is bordered to have size equal to the final image size, but this is too memory hungry approach.
	tl_new.x = dst_roi_.x + (((tl_new.x - dst_roi_.x) >> num_bands_) << num_bands_);
	tl_new.y = dst_roi_.y + (((tl_new.y - dst_roi_.y) >> num_bands_) << num_bands_);
	int width = br_new.x - tl_new.x;
	int height = br_new.y - tl_new.y;
	width += ((1 << num_bands_) - width % (1 << num_bands_)) % (1 << num_bands_);
	height += ((1 << num_bands_) - height % (1 << num_bands_)) % (1 << num_bands_);
	br_new.x = tl_new.x + width;
	br_new.y = tl_new.y + height;
	int dy = std::max(br_new.y - dst_roi_.br().y, 0);
	int dx = std::max(br_new.x - dst_roi_.br().x, 0);
	tl_new.x -= dx; br_new.x -= dx;
	tl_new.y -= dy; br_new.y -= dy;

	int top = tl.y - tl_new.y;
	int left = tl.x - tl_new.x;
	int bottom = br_new.y - tl.y - img.rows + 1;
	int right = br_new.x - tl.x - img.cols + 1;

	// Create the source image Laplacian pyramid
	cv::UMat img_with_border;
	cv::copyMakeBorder(_img, img_with_border, top, bottom, left, right,
		cv::BORDER_CONSTANT);
	std::vector<cv::UMat> src_pyr_laplace;
	this->createLaplacePyr(img_with_border, num_bands_, src_pyr_laplace);
	// Create the weight map Gaussian pyramid
	cv::UMat weight_map;
	std::vector<cv::UMat> weight_pyr_gauss(num_bands_ + 1);

	mask.getUMat().convertTo(weight_map, CV_32F, 1. / 255.);

	copyMakeBorder(weight_map, weight_pyr_gauss[0], top, bottom, left, right, cv::BORDER_CONSTANT);
	for (int i = 0; i < num_bands_; ++i)
		pyrDown(weight_pyr_gauss[i], weight_pyr_gauss[i + 1]);

	int y_tl = tl_new.y - dst_roi_.y;
	int y_br = br_new.y - dst_roi_.y;
	int x_tl = tl_new.x - dst_roi_.x;
	int x_br = br_new.x - dst_roi_.x;

	// Add weighted layer of the source image to the final Laplacian pyramid layer
	for (int i = 0; i <= num_bands_; ++i) {
		cv::Rect rc(x_tl, y_tl, x_br - x_tl, y_br - y_tl);
		cv::Mat _src_pyr_laplace = src_pyr_laplace[i].getMat(cv::ACCESS_READ);
		cv::Mat _dst_pyr_laplace = dst_pyr_laplace_[i](rc).getMat(cv::ACCESS_RW);
		cv::Mat _weight_pyr_gauss = weight_pyr_gauss[i].getMat(cv::ACCESS_READ);
		cv::Mat _dst_band_weights = dst_band_weights_[i](rc).getMat(cv::ACCESS_RW);
		for (int y = 0; y < rc.height; ++y) {
			const cv::Point3_<float>* src_row = _src_pyr_laplace.ptr<cv::Point3_<float> >(y);
			cv::Point3_<float>* dst_row = _dst_pyr_laplace.ptr<cv::Point3_<float> >(y);
			const float* weight_row = _weight_pyr_gauss.ptr<float>(y);
			float* dst_weight_row = _dst_band_weights.ptr<float>(y);
			for (int x = 0; x < rc.width; ++x) {
				dst_row[x].x += static_cast<float>(src_row[x].x * weight_row[x]);
				dst_row[x].y += static_cast<float>(src_row[x].y * weight_row[x]);
				dst_row[x].z += static_cast<float>(src_row[x].z * weight_row[x]);
				dst_weight_row[x] += weight_row[x];
			}
		}
		x_tl /= 2; y_tl /= 2;
		x_br /= 2; y_br /= 2;
	}
}

void calib::Blender::normalizeUsingWeightMap(cv::InputArray _weight, cv::InputOutputArray _src) {
	cv::Mat src;
	cv::Mat weight;
	src = _src.getMat();
	weight = _weight.getMat();
	for (int y = 0; y < src.rows; ++y) {
		cv::Point3_<float> *row = src.ptr<cv::Point3_<float> >(y);
		const float *weight_row = weight.ptr<float>(y);
		for (int x = 0; x < src.cols; ++x) {
			row[x].x = static_cast<float>(row[x].x / (weight_row[x] + weight_eps));
			row[x].y = static_cast<float>(row[x].y / (weight_row[x] + weight_eps));
			row[x].z = static_cast<float>(row[x].z / (weight_row[x] + weight_eps));
		}
	}
}

void calib::Blender::restoreImageFromLaplacePyr(std::vector<cv::UMat> &pyr) {
	if (pyr.empty())
		return;
	cv::UMat tmp;
	for (size_t i = pyr.size() - 1; i > 0; --i) {
		pyrUp(pyr[i], tmp, pyr[i - 1].size());
		add(tmp, pyr[i - 1], pyr[i - 1]);
	}
}


void calib::Blender::blend(cv::InputOutputArray dst, cv::InputOutputArray dst_mask) {
	
	cv::Rect dst_rc(0, 0, dst_roi_final_.width, dst_roi_final_.height);
	cv::UMat dst_band_weights_0;
	for (int i = 0; i <= num_bands_; ++i)
		normalizeUsingWeightMap(dst_band_weights_[i], dst_pyr_laplace_[i]);
	restoreImageFromLaplacePyr(dst_pyr_laplace_);
	dst_ = dst_pyr_laplace_[0](dst_rc);
	dst_band_weights_0 = dst_band_weights_[0];
	dst_pyr_laplace_.clear();
	dst_band_weights_.clear();
	compare(dst_band_weights_0(dst_rc), weight_eps, dst_mask_, cv::CMP_GT);

	cv::UMat mask;
	compare(dst_mask_, 0, mask, cv::CMP_EQ);
	dst_.setTo(cv::Scalar::all(0), mask);
	dst.assign(dst_);
	dst_mask.assign(dst_mask_);
	dst_.release();
	dst_mask_.release();
}