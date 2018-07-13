# CMake file for GigaRender project
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

message( 
"##
##                     _oo0oo_
##                    o8888888o
##                    88\" . \"88
##                    (| -_- |)
##                    0\\  =  /0
##                  ___/`---'\\___
##                .' \\\\|     |// '.
##               / \\\\|||  :  |||// \\
##              / _||||| -:- |||||- \\
##             |   | \\\\\\  -  /// |   |
##             | \\_|  ''\\---/''  |_/ |
##             \\  .-\\__  '-'  ___/-. /
##           ___'. .'  /--.--\\  `. .'___
##        .\"\" '<  `.___\\_<|>_/___.' >' \"\".
##       | | :  `- \\`.;`\\ _ /`;.`/ - ` : | |
##       \\  \\ `_.   \\_ __\\ /__ _/   .-` /  /
##   =====`-.____`.___ \\_____/___.-`___.-'=====
##                     `=---='
##
##             佛祖保佑         永无BUG
##               Buddha Bless No Bug
##
"
)


set(exename "PoseEstimator")
set(libname "PoseEstimation")

# ceres solver for bundle adjustment
add_definitions(
    -DGOOGLE_GLOG_DLL_DECL=
    -DREFERENCE_STITCHER_EXPORT
)
include_directories(
    E:/libs/ceres_solver/include
    E:/libs/ceres_solver/include/ceres
    E:/libs/ceres_solver/include/glog
)
set (CERES_LIBS
    debug "E:/libs/ceres_solver/lib/ceresd.lib"
    debug "E:/libs/ceres_solver/lib/libglog_staticd.lib"
    optimized "E:/libs/ceres_solver/lib/ceres.lib"
    optimized "E:/libs/ceres_solver/lib/libglog_static.lib"
)

set(MY_HEADER_FILE
    SysUtil.hpp
    FeatureMatch.h
	CameraParamEstimator.h
    Compositor.h
    BundleAdjustment.h
    ReferenceStitcher.h
)

set(MY_SOURCE_FILE 
    FeatureMatch.cpp
	CameraParamEstimator.cpp
	Compositor.cpp
    BundleAdjustment.cpp
    ReferenceStitcher.cpp
)

cuda_add_executable( ${exename} 
    ${MY_HEADER_FILE} 
    ${MY_SOURCE_FILE}
    main.cpp
)

add_library(${libname} SHARED
    ${MY_HEADER_FILE} 
    ${MY_SOURCE_FILE} 
)

target_link_libraries( ${exename} 
    ${OpenCV_LIBS}
    ${CERES_LIBS} 
)

target_link_libraries( ${libname} 
    ${OpenCV_LIBS}
    ${CERES_LIBS} 
)
