#include <iostream>
#include <string>
#include "stdlib.h"
#include <vector>


#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <iostream>
#include <string>
#include "stdlib.h"
#include <vector>

using namespace cv;
using namespace std;

#include "opticalFlow.h"
#include "omp.h"


/* show help information */
void help(){
    printf("USAGE: spm-bp image1 image2 outputfile [options]\n");
	printf("\n");
    printf("Estimate optical flow field between two images with SPM_BP and store it into a .flo file\n");
    printf("\n");
    printf("options:\n"); 
    printf("spm-bp parameters\n");
	printf(" -it_num	<int>(5)	number of iterations\n"); 
	printf(" -sp_num	<int>(500)	number of superpixels\n");
	printf(" -max_u		<int>(100)	motion range in pixels(ver)\n"); 
	printf(" -max_v		<int>(200)	motion range in pixels(hor)\n"); 
	printf(" -kn_size	<int>(9)	filter kerbel radius\n");
	printf(" -kn_tau	<int>(25)	filter smootheness\n");
	printf(" -lambda	<float>(2)	pairwise smoothness\n");
	printf(" -verbose	    		display intermediate result\n");
    printf("\n");
}

int main(int argc, char **argv){

	//cv::Mat img = cv::imread("E:/Project/LightFieldGiga/build/bin/Release/left.jpg");
	//cv::Mat img2;
	//cv::bilateralFilter(img, img2, 17, 150, 150, cv::BORDER_REPLICATE);
	//cv::ximgproc::l0Smooth(img2, img2, 0.01, 2);

	//return 0;


	std::string dir = "E:/data/giga_stereo/1";
	spm_bp_params params;
    spmbp_params_default(&params);
	// read optional arguments 
	params.display = false;
	params.max_u = 2;
	params.max_v = 40;
	params.sp_num = 1500;
	params.iter_num = 8;
	params.kn_size = 50;
	params.up_rate = 2;
	// optical flow estimation 
	opticalFlow of_est;
	for (size_t i = 0; i < 100; i++) {
		std::cout << cv::format("Process image index %04d ...", i) << std::endl;
		std::string leftname = cv::format("%s/data/1/%04d.jpg", dir.c_str(), i);
		std::string rightname = cv::format("%s/data/0/%04d.jpg", dir.c_str(), i);
		std::string visualname = cv::format("%s/result/%04d_visual.jpg", dir.c_str(), i);
		std::string depthname = cv::format("%s/result/%04d_depth.png", dir.c_str(), i);
		of_est.runFlowEstimator(leftname, rightname, depthname, visualname, &params);
	}
	return 0;
}
