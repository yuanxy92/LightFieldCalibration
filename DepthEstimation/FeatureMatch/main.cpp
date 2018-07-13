#include "FeatureMatch.h"

int main(int argc, char* argv[]) {
	//argv[1] = "E:\\Project\\LightFieldGiga\\data\\result\\0000_left.jpg";
	//argv[2] = "E:\\Project\\LightFieldGiga\\data\\result\\0000_right.jpg";
	//argv[3] = "0";
	//argv[4] = "90";
	//argv[5] = "match.tmp";

	cv::Mat leftImg = cv::imread(argv[1]);
	cv::Mat rightImg = cv::imread(argv[2]);
	int minDisparity = atoi(argv[3]);
	int maxDisparity = atoi(argv[4]);
	std::vector<fm::MatchingPoints> matchingPts = 
		fm::FeatureMatch::constructMatchingPointsZNCC(
			leftImg, rightImg, minDisparity, maxDisparity);
	// write into files
	std::string filename(argv[5]);
	std::fstream fs(filename, std::ios::out);
	for (size_t i = 0; i < matchingPts.size(); i++) {
		fs << matchingPts[i].leftPt.x << '\t';
		fs << matchingPts[i].leftPt.y << '\t';
		fs << matchingPts[i].rightPt.x << '\t';
		fs << matchingPts[i].rightPt.y << '\t';
		fs << matchingPts[i].confidence << '\t';
		fs << 1.0 << std::endl;
	}
	fs.close();

	if (argc > 6) {
		// generate depth map and confidence map for bilateral solver
		cv::Mat depth(leftImg.rows, leftImg.cols, CV_16U);
		cv::Mat confidence(leftImg.rows, leftImg.cols, CV_32F);
		confidence.setTo(0.001);
		for (size_t i = 0; i < matchingPts.size(); i++) {
			cv::Point2f p = matchingPts[i].leftPt;
			confidence.at<float>(p.y, p.x) = matchingPts[i].confidence;
		}
		// normalize
		double maxVal;
		cv::GaussianBlur(confidence, confidence, cv::Size(11, 11), 1.5, 0);
		cv::minMaxIdx(confidence, NULL, &maxVal);
		confidence = confidence / maxVal * (pow(2, 16) - 1);
		confidence.convertTo(confidence, CV_16U);
		cv::imwrite(argv[6], confidence);
	}
	return 0;
}