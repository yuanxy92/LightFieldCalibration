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

cmake_minimum_required (VERSION 3.10)
set(exename "SphereMeshGenerator")

# opengl
find_package(OpenGL REQUIRED)
if ( WIN32 )
	set(GLFW_INCLUDE_PATH "E:/libs/OpenGL/include/GLFW")
	set(GLFW_LIBRARY "E:/libs/OpenGL/lib/glfw3.lib" )
	set(GLEW_INCLUDE_PATH "E:/libs/OpenGL/include")
	set(GLEW_LIBRARY_DEBUG debug "E:/libs/OpenGL/lib/glew32d.lib")
	set(GLEW_LIBRARY_RELEASE optimized "E:/libs/OpenGL/lib/glew32.lib")
	set(GLM_INCLUDE_PATH "E:/libs/OpenGL/include")
endif()
include_directories(
	${GLFW_INCLUDE_PATH}
	${GLEW_INCLUDE_PATH}
	${GLM_INCLUDE_PATH}
)

# Open3D
set(Open3D_DIR "E:/libs/Open3D/include")
set(Open3D_LIBS 
    debug "E:/libs/Open3D/lib/Core_d.lib"
    debug "E:/libs/Open3D/lib/glew_d.lib"
    debug "E:/libs/Open3D/lib/glfw3_d.lib"
    debug "E:/libs/Open3D/lib/IO_d.lib"
    debug "E:/libs/Open3D/lib/jpeg_d.lib"
    debug "E:/libs/Open3D/lib/jsoncpp_d.lib"
    debug "E:/libs/Open3D/lib/png_d.lib"
    debug "E:/libs/Open3D/lib/py3d_d.lib"
    debug "E:/libs/Open3D/lib/realsense_d.lib"
    debug "E:/libs/Open3D/lib/tinyfiledialogs_d.lib"
    debug "E:/libs/Open3D/lib/Visualization_d.lib"
    debug "E:/libs/Open3D/lib/zlib_d.lib"

    optimized "E:/libs/Open3D/lib/Core.lib"
    optimized "E:/libs/Open3D/lib/glew.lib"
    optimized "E:/libs/Open3D/lib/glfw3.lib"
    optimized "E:/libs/Open3D/lib/IO.lib"
    optimized "E:/libs/Open3D/lib/jpeg.lib"
    optimized "E:/libs/Open3D/lib/jsoncpp.lib"
    optimized "E:/libs/Open3D/lib/png.lib"
    optimized "E:/libs/Open3D/lib/py3d.lib"
    optimized "E:/libs/Open3D/lib/realsense.lib"
    optimized "E:/libs/Open3D/lib/tinyfiledialogs.lib"
    optimized "E:/libs/Open3D/lib/Visualization.lib"
    optimized "E:/libs/Open3D/lib/zlib.lib"
)
include_directories(${Open3D_DIR})

set(SRC_FILES
    SphereMesh.h
    SphereMesh.cpp
    DepthRefine.h
    DepthRefine.cpp
    OBJModelIO.h
    OBJModelIO.cpp
    GraphCutMask.h
    GraphCutMask.cpp
    Warper.h
    Warper.cpp
    main.cpp
)

add_executable(${exename}
    ${SRC_FILES}
)

target_link_libraries(${exename}
    ${OpenCV_LIBS}
    ${Open3D_LIBS}
    ${OPENGL_LIBRARY}
    ${GLFW_LIBRARY}
	${GLEW_LIBRARY_DEBUG}
    ${GLEW_LIBRARY_RELEASE}
)
