#include "SphereMesh.h"

int main() {
	cv::Mat img = cv::imread("E:\\Project\\LightFieldGiga\\data\\data1\\result\\0035_left.jpg");
	cv::Mat depth = cv::imread("E:\\Project\\LightFieldGiga\\data\\data1\\result\\0035_depth.png",
		CV_LOAD_IMAGE_ANYDEPTH);
	cv::Mat_<float> K(3, 3);
	K.setTo(0);
	K(0, 0) = 3480;
	K(1, 1) = 3480;
	K(2, 2) = 1;
	K(0, 2) = 605.5;
	K(1, 2) = 505.5;

	three::SphereMesh mesh;
	mesh.setInput(img, depth, K);
	mesh.debug();

	return 0;
}