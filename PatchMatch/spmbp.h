#pragma once

#include <vector>
using std::vector;

#include <bitset>
using std::bitset;

#include "opencv2/opencv.hpp"
using namespace cv;

#include <iostream>
using std::cout;
using std::endl;

#include "Common_Datastructure_Header.h"

#define UPSAMPLE 1
#define DO_LEFT 1
#define DO_RIGHT 0
#define NUM_TOP_K 3 ///////////3
#define NTHREADS 8
#define EPS 0.01
//data cost
#define DATA_COST_TRUNCATED_L1 0
#define DATA_COST_TRUNCATED_L2 0 //0
#define DATA_COST_ADCENSUS 1     //1
#define USE_POINTER_WISE 1

#define USE_CENSUS 1 //////

#define USE_CLMF0_TO_AGGREGATE_COST 1

//smooth
#define SMOOTH_COST_TRUNCATED_L1 0
#define SMOOTH_COST_TRUNCATED_L2 1

#define SAVE_DCOST 0

#define BP_RECT 1

const int CENSUS_WINSIZE = 11;
const int CENSUS_SIZE_OF = CENSUS_WINSIZE * CENSUS_WINSIZE;
const int CENSUS_LENGTH = CENSUS_WINSIZE * CENSUS_WINSIZE - 1;

class spm_bp {
public:
	int positive;
    //input
	bool display;
    int height1, width1, height2, width2;
    Mat_<Vec3f> im1f, im2f;
    Mat_<Vec3b> im1d, im2d;
    Mat_<float> im1Gray, im2Gray;
    // upsample
    int upScale;
    int height1_up, width1_up;
    Mat_<Vec3f> im1Up, im2Up;
    float expCensusDiffTable[CENSUS_SIZE_OF + 1];
    float expColorDiffTable[256];
    Mat_<Vec2f> im1GradUp, im2GradUp;

    // census bitset
    vector<vector<bitset<CENSUS_SIZE_OF>>> censusBS1;
    vector<vector<bitset<CENSUS_SIZE_OF>>> censusBS2;
    vector<vector<vector<bitset<CENSUS_SIZE_OF>>>> subCensusBS1;
    vector<vector<vector<bitset<CENSUS_SIZE_OF>>>> subCensusBS2;

    //optical flow
    int dispRange;
    int disp_range_u, disp_range_v;
    int range_u, range_v;
    int labelNum;
    float omega, alpha;
    float tau_c;
    float tau_s;

	// filter kernel related
	int g_filterKernelSize;
	int g_filterKernelBoundarySize;
	int g_filterKernelColorTau;
	float g_filterKernelEpsl;

    //guided filter
    int gf_r;
    float gf_eps;

    // crossmap of left and right images
    int crossArmLength, crossColorTau;
    cv::Mat_<cv::Vec4b> crossMap1, crossMap2;
    // sub-image buffer and sub-image cross map buffer
    vector<cv::Mat_<cv::Vec4b> > subCrossMap1;
    vector<cv::Mat_<cv::Vec4b> > subCrossMap2;
    // sub-image buffer
    vector<Mat_<Vec3f> > subImage1;
    vector<Mat_<Vec3f> > subImage2;

    //PMBP
    //int num_top_k;
    int iterNum;
    float lambda_smooth;
	Mat_<Vec2f> label_k;
	Mat_<float> dcost_k;

    // super pixel
    int createAndOrganizeSuperpixels();
    int g_spMethod;
    int g_spNumber;
    int g_spSize;
    int g_spSizeOrNumber;
    // sub-image range and super-pixel range
    // {0: left, 1: up, 2: right, 3: down}
    vector<Vec4i> subRange1, subRange2;
    vector<Vec4i> spRange1, spRange2;
    // number of super-pixels
    int numOfSP1, numOfSP2;

    // the label of each super pixel
    Mat_<int> segLabels1, segLabels2;
    // use for superpixel based graph traverse
    vector<int> repPixels1, repPixels2;
    GraphStructure spGraph1[2];
    // to random assign pixels
    vector<vector<int> > superpixelsList1;
    vector<vector<int> > superpixelsList2;
	

    void GetSuperpixelsListFromSegment(const Mat_<int>& segLabels, int numOfLabels, vector<vector<int> >& spPixelsList);
    void RandomAssignRepresentativePixel(const vector<vector<int> >& spPixelsList, int numOfLabels, vector<int>& rePixel);
    // super pixel graph
    void BuildSuperpixelsPropagationGraph(const Mat_<int>& refSegLabel, int numOfLabels, const Mat_<Vec3f>& refImg, GraphStructure& spGraphEven, GraphStructure& spGraphEvenOdd);
    void AssociateLeftImageItselfToEstablishNonlocalPropagation(int sampleNum, int topK);
    void ModifyCrossMapArmlengthToFitSubImage(const cv::Mat_<cv::Vec4b>& crMapIn, int maxArmLength, cv::Mat_<cv::Vec4b>& crMapOut);
    void getLocalDataCostPerlabel(int sp, const Vec2f& fl, Mat_<float>& localDataCost);
    //labelling init
    void init_label_super(Mat_<Vec2f>& label_k, Mat_<float>& dCost_super_k);
	void loadPairs(cv::Mat& in1, Mat& in2);
	void setParameters(spm_bp_params* params);
	void preProcessing();
	void initiateData();
	void runspm_bp(cv::Mat_<cv::Vec2f>& flowResult);
	void Show_WTA_Flow(int iter, Mat_<Vec2f>& label_k, Mat_<float>& dCost_k, Mat_<Vec<float, NUM_TOP_K> >& message, cv::Mat_<cv::Vec2f>& flowResult);

	spm_bp(int positive);
	~spm_bp();
};
