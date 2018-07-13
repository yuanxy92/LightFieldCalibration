/**
@brief main.cpp
@author Shane Yuan
@date May 26, 2018
*/

#include "Fragments.h"

int main(int argc, char* argv[]) {

	//argv[1] = "E:\\Project\\LightFieldGiga\\data\\data1\\result";
	//argv[2] = "0";
	//argv[3] = "29";
	//argv[4] = "14";
	//argv[5] = "E:\\Project\\LightFieldGiga\\data\\data1\\result_fusion\\fragment_0_29_14";

	std::string datapath = std::string(argv[1]);
	int startInd = atoi(argv[2]);
	int endInd = atoi(argv[3]);
	int representInd = atoi(argv[4]);
	std::string outpath = std::string(argv[5]);
	std::vector<std::string> colornames;
	std::vector<std::string> depthnames;
	for (size_t i = startInd; i <= endInd; i++) {
		colornames.push_back(cv::format("%s/%04d_left.jpg", datapath.c_str(), i));
		depthnames.push_back(cv::format("%s/%04d_depth.png", datapath.c_str(), i));
	}
	std::string cameraname = cv::format("%s/CameraParams_%04d_%04d.txt", outpath.c_str(), startInd, endInd);

	Fragments fragments;
	fragments.init(colornames, depthnames, cameraname, representInd - startInd);
	fragments.fusion();
	
	cv::Mat refineDepthMap, colorImg;
	fragments.getResults(refineDepthMap, colorImg);
	cv::imwrite(cv::format("%s/depth.png", outpath.c_str()), refineDepthMap);
	cv::imwrite(cv::format("%s/color.png", outpath.c_str()), colorImg);

	return 0;
}