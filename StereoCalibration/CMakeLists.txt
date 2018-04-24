# CMake file for stereo giga calibration project
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

# Camera Driver files
# camera SDK
# pointgrey camera sdk
set(PTGREY_SDK_INCLUDE_DIR "C:/Program Files/Point Grey Research/Spinnaker/include")
set(PTGREY_SDK_LIBRARY 
	"C:/Program Files/Point Grey Research/Spinnaker/lib64/vs2015/Spinnaker_v140.lib"
)
include_directories(${PTGREY_SDK_INCLUDE_DIR})
# XIMEA camera
set(XIMEA_SDK_INCLUDE_DIR "C:/XIMEA/API")
set(XIMEA_SDK_LIBRARY 
	"C:/XIMEA/API/x64/xiapi64.lib"
)
include_directories(${XIMEA_SDK_INCLUDE_DIR})
# General camera driver
set(CAMERA_DRIVER_DIR ${PROJECT_SOURCE_DIR}/GenCameraDriver)
set(CAMERA_DRIVER_FILE
	${CAMERA_DRIVER_DIR}/XIMEA/XIMEACamera.h
	${CAMERA_DRIVER_DIR}/FileCamera/FileCamera.h
	${CAMERA_DRIVER_DIR}/PointGrey/PointGreyCamera.h
	${CAMERA_DRIVER_DIR}/GenCameraDriver.h
	${CAMERA_DRIVER_DIR}/RealCameraDriver.h
	${CAMERA_DRIVER_DIR}/NPPJpegCoder.h
	${CAMERA_DRIVER_DIR}/XIMEA/XIMEACamera.cpp
	${CAMERA_DRIVER_DIR}/FileCamera/FileCamera.cpp
	${CAMERA_DRIVER_DIR}/PointGrey/PointGreyCamera.cpp
	${CAMERA_DRIVER_DIR}/GenCameraDriver.cpp
	${CAMERA_DRIVER_DIR}/GenCameraDriverImpl.cpp
	${CAMERA_DRIVER_DIR}/RealCameraDriver.cpp
	${CAMERA_DRIVER_DIR}/NPPJpegCoder.cpp
)
set(CAMERA_DRIVER_LIB
	${PTGREY_SDK_LIBRARY}
	${XIMEA_SDK_LIBRARY}
)
include_directories(
	${CAMERA_DRIVER_DIR}
	${PTGREY_SDK_INCLUDE_DIR}
	${XIMEA_SDK_INCLUDE_DIR}
)

# C++ header file
set(MY_HEADER_FILES
    # C++ header file
    ${PROJECT_SOURCE_DIR}/Common/SysUtil.hpp
    StereoCalibration.h
)

set(MY_SOURCE_FILES 
	# C++ code
    StereoCalibration.cpp
    StereoCalibration.cu
	${CAMERA_DRIVER_FILE}
)

cuda_add_executable(${stereo_exe_name}
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
	# camera libs
	${PTGREY_SDK_LIBRARY}
	${XIMEA_SDK_LIBRARY}
)
