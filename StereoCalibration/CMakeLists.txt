# CMake file for GigaRender Calibrate submodule project
# Author: Shane Yuan
# E-mail: yuanxy92@gmail.com
#

##
##                     _oo0oo_
##                    o8888888o
##                    88" . "88
##                    (| -_- |)
##                    0\  =  /0
##                  ___/`---'\___
##                .' \\|     |// '.
##               / \\|||  :  |||// \
##              / _||||| -:- |||||- \
##             |   | \\\  -  /// |   |
##             | \_|  ''\---/''  |_/ |
##             \  .-\__  '-'  ___/-. /
##           ___'. .'  /--.--\  `. .'___
##        ."" '<  `.___\_<|>_/___.' >' "".
##       | | :  `- \`.;`\ _ /`;.`/ - ` : | |
##       \  \ `_.   \_ __\ /__ _/   .-` /  /
##   =====`-.____`.___ \_____/___.-`___.-'=====
##                     `=---='
##
##             佛祖保佑         永无BUG
##               Buddha Bless No Bug
##

set(stereo_exe_name "StereoCalibrate")

# C++ header file
set(MY_HEADER_FILES
    # C++ header file
    ${PROJECT_SOURCE_DIR}/Common/SysUtil.hpp
    StereoCalibration.h
)

set(MY_SOURCE_FILES 
	# C++ code
    StereoCalibration.cpp
)

cuda_add_library(${stereo_exe_name}
    ${MY_HEADER_FILES}
    ${MY_SOURCE_FILES}
    main.cpp
)

target_link_libraries(${stereo_exe_name}
    # OpenCV
    ${OpenCV_LIBS}
    # cuda libs
    ${CUDA_LIBRARY} 	
	${CUDA_npp_LIBRARY} 
	${CUDA_nppc_LIBRARY} 
	${CUDA_nppicc_LIBRARY}
	${CUDA_nppicom_LIBRARY}
	${CUDA_nppidei_LIBRARY}
	${CUDA_nppif_LIBRARY}
	${CUDA_nppig_LIBRARY}
	${CUDA_nppim_LIBRARY}
	${CUDA_nppist_LIBRARY}
	${CUDA_nppisu_LIBRARY}
	${CUDA_nppitc_LIBRARY}
	${CUDA_npps_LIBRARY} 
)
