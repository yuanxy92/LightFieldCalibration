/**
@brief main function file of registration
@author Shane Yuan
*/
#include "Registration.h"
#include "Triangulate.h"

int main(int argc, char* argv[]) {
	std::shared_ptr<calib::Registration> registrationPtr =
		std::make_shared<calib::Registration>();

	argv[1] = "E:\\Project\\LightFieldGiga\\data\\data1\\result_fusion\\final";
	argv[2] = "0";
	argv[3] = "22";
	argv[4] = "mesh_texture";

	std::string datapath = std::string(argv[1]);
	int startInd = atoi(argv[2]);
	int endInd = atoi(argv[3]);
	std::string outpath = std::string(cv::format("%s\\result", argv[1]));
	std::vector<std::string> colornames;
	std::vector<std::string> depthnames;
	for (size_t i = startInd; i <= endInd; i++) {
		colornames.push_back(cv::format("%s/%04d_left.jpg", datapath.c_str(), i));
		depthnames.push_back(cv::format("%s/%04d_depth.png", datapath.c_str(), i));
	}
	std::string cameraname = cv::format("%s/CameraParams_%04d_%04d.txt", datapath.c_str(),
		startInd, endInd);

	registrationPtr->init(cameraname, colornames, depthnames);

	registrationPtr->registrate();

	cv::imwrite(cv::format("%s/panorama_depth.png", argv[1]), registrationPtr->getDepthMap());

	std::shared_ptr<three::TexTriangleMesh> texmesh = registrationPtr->getTexMesh();
	three::OBJModelIO::save(argv[1], argv[4], texmesh);

	return 0;
}