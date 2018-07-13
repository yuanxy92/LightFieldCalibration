# CMake file for depth map estimation
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

set(exe_name "SPPMStereo")

file(GLOB FILTER_FILE 
    FGS*
    CF*)

file(GLOB SUPERPIXEL_FILE 
    SLIC* 
    Superpixels*)

file(GLOB FLOW_MANIPULATION_FILE 
    colorcode* 
    FlowInputOutput*)

file(GLOB COMMON_FILE 
    Common_Datastructure_*
    )

file(GLOB OF_FILE 
    opticalFlow*
    )

file(GLOB SPMBP_FILE 
    spmbp.h
    spmbp.cpp
    )

add_executable(${exe_name} 
    main.cpp
    ${FILTER_FILE}
    ${SUPERPIXEL_FILE}
    ${FLOW_MANIPULATION_FILE}
    ${COMMON_FILE}
    ${OF_FILE}
    ${SPMBP_FILE}
    )

source_group("filter" FILES ${FILTER_FILE})
source_group("super pixel" FILES ${SUPERPIXEL_FILE})
source_group("flow_io" FILES ${FLOW_MANIPULATION_FILE})
source_group("util" FILES ${COMMON_FILE})
source_group("optical flow" FILES ${OF_FILE})
source_group("spmbp" FILES ${SPMBP_FILE})

target_link_libraries(${exe_name} ${OpenCV_LIBS})