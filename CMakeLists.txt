cmake_minimum_required(VERSION 2.8)
project("IVSUMM")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++0x")

if (CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -fopenmp")
    set(CMAKE_EXE_LINKER_FLAGS "-s")
endif()

find_package(PkgConfig)
pkg_check_modules(OPENCV3 REQUIRED opencv3)
pkg_check_modules(OPENBLAS REQUIRED blas)

set(OPENCV_INCLUDES "${OPENCV3_INCLUDE_DIRS}") # /usr/local/Cellar/3.4.1_6/include/ /usr/local/Cellar/3.4.1_6/include/opencv/
set(OPENCV_LIBPATH "${OPENCV3_LIBRARY_DIRS}") # /usr/local/Cellar/opencv/3.4.1_6/lib/
set(OPENCV_LIBS "${OPENCV3_LIBRARIES}") # [[Required Opencv libs : -lopencv_imgproc -lopencv_core -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_bgsegm -lopencv_video -lopencv_objdetect -lopencv_dnn]]

set(OPENBLAS_INCLUDES "${OPENBLAS_INCLUDE_DIRS}") # /usr/local/Cellar/openblas/0.3.1/include
set(OPENBLAS_LIBPATH "${OPENBLAS_LIBRARY_DIRS}") # /usr/local/Cellar/openblas/0.3.1/lib

set(IVSUMM_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/src/engine/" "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/" "${CMAKE_CURRENT_SOURCE_DIR}/src/isumm/" "${CMAKE_CURRENT_SOURCE_DIR}/src/vsumm/")

set(CAFFE_INCLUDES "/Users/rkiyer/Desktop/caffe/include")
set(CAFFE_LIBPATH "Users/rkiyer/Desktop/caffe/build/lib")

set(GLOG_INCLUDES "/usr/local/Cellar/glog/0.3.5_3/include")
set(GLOG_LIBPATH "/usr/local/Cellar/glog/0.3.5_3/lib")

# build summengine.a
add_library(summengine
  src/engine/set.cpp
	src/engine/SetFunctions.cc
	src/engine/SparseFeature.cc
	src/engine/DisparityMin.cc
	src/engine/DisparitySum.cc
	src/engine/FacilityLocation.cc
	src/engine/FeatureBasedFunctions.cc
	src/engine/GraphCutFunctions.cc
	src/engine/ProbabilisticSetCover.cc
	src/engine/SaturateCoverage.cc
	src/engine/SetCover.cc
  src/engine/mmr.cpp
	src/engine/increment.cc
  src/engine/lazyGreedyMax.cc
  src/engine/naiveGreedyMax.cc
	src/engine/streamGreedy.cc
  src/utils/error.cc
  src/utils/arguments.cc
)

######################################### Simple Video Summarizer Examples ########################################
include_directories(${OPENCV_INCLUDES} ${IVSUMM_INCLUDES})
link_directories(${OPENCV_LIBPATH})
add_executable(SimpleVideoSummExample src/utils/ShotDetector.cc src/vsumm/SimpleVideoSummarizer.cc src/utils/ColorUtils.cc src/utils/ImageUtils.cc examples/SimpleVideoSummExample.cc)
target_link_libraries(SimpleVideoSummExample
    summengine
    ${OPENCV_LIBS}
)

######################################### Simple Image Summarizer Examples ########################################
include_directories(${OPENCV_INCLUDES} ${IVSUMM_INCLUDES})
link_directories(${OPENCV_LIBPATH})
add_executable(SimpleImageSummExample src/utils/ShotDetector.cc src/isumm/SimpleImageSummarizer.cc src/utils/ColorUtils.cc src/utils/ImageUtils.cc examples/SimpleImageSummExample.cc)
target_link_libraries(SimpleImageSummExample
    summengine
    ${OPENCV_LIBS}
)

######################################### Deep Video Summarizer Example ########################################
include_directories(${OPENCV_INCLUDES} ${IVSUMM_INCLUDES} ${CAFFE_INCLUDES} ${OPENBLAS_INCLUDES} ${GLOG_INCLUDES})
link_directories(${OPENCV3_LIBRARY_DIRS} ${CAFFE_LIBPATH} ${OPENBLAS_LIBPATH} ${GLOG_LIBPATH})
add_executable(DeepVideoSummExample examples/DeepVideoSummExample.cc src/utils/ShotDetector.cc src/utils/ColorUtils.cc
    src/utils/caffeClassifier.cc src/vsumm/DeepSimVideoSummarizer.cc src/utils/ImageUtils.cc src/vsumm/DeepCoverVideoSummarizer.cc)
set_source_files_properties(examples/DeepVideoSummExample.cc PROPERTIES COMPILE_FLAGS "-DCPU_ONLY" )
set_source_files_properties(src/utils/caffeClassifier.cc PROPERTIES COMPILE_FLAGS "-DCPU_ONLY" )
set_source_files_properties(src/vsumm/DeepSimVideoSummarizer.cc PROPERTIES COMPILE_FLAGS "-DCPU_ONLY" )
set_source_files_properties(src/vsumm/DeepCoverVideoSummarizer.cc PROPERTIES COMPILE_FLAGS "-DCPU_ONLY" )
target_link_libraries(DeepVideoSummExample
    summengine ${OPENCV_LIBS} -lcaffe -lglog -lboost_system -lblas
)

######################################### Deep Image Summarizer Example ########################################
include_directories(${OPENCV_INCLUDES} ${IVSUMM_INCLUDES} ${CAFFE_INCLUDES} ${OPENBLAS_INCLUDES} ${GLOG_INCLUDES})
link_directories(${OPENCV_LIBPATH} ${CAFFE_LIBPATH} ${OPENBLAS_LIBPATH} ${GLOG_LIBPATH})
add_executable(DeepImageSummExample examples/DeepImageSummExample.cc src/utils/ShotDetector.cc src/utils/ColorUtils.cc
    src/utils/caffeClassifier.cc src/isumm/DeepSimImageSummarizer.cc src/utils/ImageUtils.cc src/isumm/DeepCoverImageSummarizer.cc)
set_source_files_properties(examples/DeepImageSummExample.cc PROPERTIES COMPILE_FLAGS "-DCPU_ONLY" )
set_source_files_properties(src/utils/caffeClassifier.cc PROPERTIES COMPILE_FLAGS "-DCPU_ONLY" )
set_source_files_properties(src/isumm/DeepSimImageSummarizer.cc PROPERTIES COMPILE_FLAGS "-DCPU_ONLY" )
set_source_files_properties(src/isumm/DeepCoverImageSummarizer.cc PROPERTIES COMPILE_FLAGS "-DCPU_ONLY" )
target_link_libraries(DeepImageSummExample
    summengine ${OPENCV_LIBS} -lcaffe -lglog -lboost_system -lblas

)

######################################### Query Video Summarizer Example ########################################
include_directories(/usr/local/Cellar/opencv/3.4.1_2/include/ /usr/local/Cellar/opencv/3.4.1_2/include/opencv/
/Users/pratik/Git/ivsumm/src/engine/ /Users/pratik/Git/ivsumm/src/utils/ /Users/pratik/Git/ivsumm/src/vsumm/
/Users/pratik/AITOE/aisaacv2_deps/caffe/include /usr/local/Cellar/openblas/0.2.20_1/include /usr/local/Cellar/glog/0.3.5_3/include)
link_directories(/usr/local/Cellar/opencv/3.4.1_2/lib/ /Users/pratik/AITOE/aisaacv2_deps/caffe/build/lib
/usr/local/Cellar/openblas/0.2.20_1/lib /usr/local/Cellar/glog/0.3.5_3/lib)
add_executable(QueryVideoSummExample examples/QueryVideoSummExample.cc src/utils/ShotDetector.cc src/utils/ColorUtils.cc
    src/utils/caffeClassifier.cc src/vsumm/QuerySimVideoSummarizer.cc src/utils/ImageUtils.cc)
set_source_files_properties(examples/QueryVideoSummExample.cc PROPERTIES COMPILE_FLAGS "-DCPU_ONLY" )
set_source_files_properties(src/utils/caffeClassifier.cc PROPERTIES COMPILE_FLAGS "-DCPU_ONLY" )
set_source_files_properties(src/vsumm/QuerySimVideoSummarizer.cc PROPERTIES COMPILE_FLAGS "-DCPU_ONLY" )
target_link_libraries(QueryVideoSummExample
    summengine -lopencv_imgproc -lcaffe -lglog
    -lopencv_core
    -lopencv_highgui
    -lopencv_videoio
    -lopencv_imgcodecs
    -lopencv_bgsegm
    -lopencv_video
    -lopencv_objdetect
)

######################################### Query Image Summarizer Example ########################################
include_directories(/usr/local/Cellar/opencv/3.4.1_2/include/ /usr/local/Cellar/opencv/3.4.1_2/include/opencv/
/Users/pratik/Git/ivsumm/src/engine/ /Users/pratik/Git/ivsumm/src/utils/ /Users/pratik/Git/ivsumm/src/isumm/
/Users/pratik/AITOE/aisaacv2_deps/caffe/include /usr/local/Cellar/openblas/0.2.20_1/include /usr/local/Cellar/glog/0.3.5_3/include)
link_directories(/usr/local/Cellar/opencv/3.4.1_2/lib/ /Users/pratik/AITOE/aisaacv2_deps/caffe/build/lib
/usr/local/Cellar/openblas/0.2.20_1/lib /usr/local/Cellar/glog/0.3.5_3/lib)
add_executable(QueryImageSummExample examples/QueryImageSummExample.cc src/utils/ColorUtils.cc
    src/utils/caffeClassifier.cc src/isumm/QuerySimImageSummarizer.cc src/utils/ImageUtils.cc)
set_source_files_properties(examples/QueryImageSummExample.cc PROPERTIES COMPILE_FLAGS "-DCPU_ONLY" )
set_source_files_properties(src/utils/caffeClassifier.cc PROPERTIES COMPILE_FLAGS "-DCPU_ONLY" )
set_source_files_properties(src/isumm/QuerySimImageSummarizer.cc PROPERTIES COMPILE_FLAGS "-DCPU_ONLY" )
target_link_libraries(QueryImageSummExample
    summengine -lopencv_imgproc -lcaffe -lglog
    -lopencv_core
    -lopencv_highgui
    -lopencv_videoio
    -lopencv_imgcodecs
    -lopencv_bgsegm
    -lopencv_video
    -lopencv_objdetect
)

######################################### Entity Face Video Summarizer Example ########################################
include_directories(${OPENCV_INCLUDES} ${IVSUMM_INCLUDES} ${CAFFE_INCLUDES} ${OPENBLAS_INCLUDES} ${GLOG_INCLUDES})
link_directories(${OPENCV_LIBPATH} ${CAFFE_LIBPATH} ${OPENBLAS_LIBPATH} ${GLOG_LIBPATH} /usr/local/lib/X11/)
add_executable(EntityFaceSummExample examples/EntityFaceSummExample.cc src/utils/caffeClassifier.cc src/utils/dnnClassifier.cc
src/utils/dlibClassifier.cc src/vsumm/EntitySimVideoSummarizer.cc src/utils/ImageUtils.cc src/utils/dlib/all/source.cpp)
set_source_files_properties(examples/EntityFaceSummExample.cc PROPERTIES COMPILE_FLAGS "-DCPU_ONLY" )
set_source_files_properties(src/utils/caffeClassifier.cc PROPERTIES COMPILE_FLAGS "-DCPU_ONLY" )
set_source_files_properties(src/vsumm/EntitySimVideoSummarizer.cc PROPERTIES COMPILE_FLAGS "-DCPU_ONLY" )
target_link_libraries(EntityFaceSummExample
    summengine ${OPENCV_LIBS} -lcaffe -lglog -lboost_system -lblas -lX11
)

######################################### Entity Object Video Summarizer Example ########################################
include_directories(${OPENCV_INCLUDES} ${IVSUMM_INCLUDES} ${CAFFE_INCLUDES} ${OPENBLAS_INCLUDES} ${GLOG_INCLUDES})
link_directories(${OPENCV_LIBPATH} ${CAFFE_LIBPATH} ${OPENBLAS_LIBPATH} ${GLOG_LIBPATH} /usr/local/lib/X11/)
add_executable(EntityObjectSummExample examples/EntityObjectSummExample.cc src/utils/caffeClassifier.cc src/utils/dnnClassifier.cc
src/utils/dlibClassifier.cc src/vsumm/EntitySimVideoSummarizer.cc src/utils/ImageUtils.cc src/utils/dlib/all/source.cpp)
set_source_files_properties(examples/EntityObjectSummExample.cc PROPERTIES COMPILE_FLAGS "-DCPU_ONLY" )
set_source_files_properties(src/utils/caffeClassifier.cc PROPERTIES COMPILE_FLAGS "-DCPU_ONLY" )
set_source_files_properties(src/vsumm/EntitySimVideoSummarizer.cc PROPERTIES COMPILE_FLAGS "-DCPU_ONLY" )
target_link_libraries(EntityObjectSummExample
    summengine ${OPENCV_LIBS} -lcaffe -lglog -lboost_system -lblas -lX11
)

######################################### Compute Distance Test ########################################
include_directories(/usr/local/Cellar/opencv/3.4.1_2/include/ /usr/local/Cellar/opencv/3.4.1_2/include/opencv/ /Users/pratik/Git/ivsumm/src/utils/ /Users/pratik/AITOE/aisaacv2_deps/caffe/include /usr/local/Cellar/openblas/0.2.20_1/include /usr/local/Cellar/glog/0.3.5_3/include)
link_directories(/usr/local/Cellar/opencv/3.4.1_2/lib/ /Users/pratik/AITOE/aisaacv2_deps/caffe/build/lib
/usr/local/Cellar/openblas/0.2.20_1/lib /usr/local/Cellar/glog/0.3.5_3/lib)
add_executable(computeDistance test/computeDistance.cc src/utils/caffeClassifier.cc)
set_source_files_properties(test/computeDistance.cc PROPERTIES COMPILE_FLAGS "-DCPU_ONLY" )
set_source_files_properties(src/utils/caffeClassifier.cc PROPERTIES COMPILE_FLAGS "-DCPU_ONLY" )
target_link_libraries(computeDistance
  -lopencv_imgproc -lcaffe -lglog
    -lopencv_core
    -lopencv_highgui
    -lopencv_videoio
    -lopencv_imgcodecs
    -lopencv_bgsegm
    -lopencv_video
    -lopencv_objdetect
)

SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "IVSUMM")
SET(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README")
SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/COPYING")
SET(CPACK_PACKAGE_VERSION_MAJOR "0")
SET(CPACK_PACKAGE_VERSION_MINOR "0")
SET(CPACK_PACKAGE_VERSION_PATCH "0")
SET(CPACK_PACKAGE_INSTALL_DIRECTORY "CMake ${CMake_VERSION_MAJOR}.${CMake_VERSION_MINOR}")
IF(WIN32 AND NOT UNIX)
  SET(CPACK_NSIS_HELP_LINK "")
  SET(CPACK_NSIS_URL_INFO_ABOUT "")
  SET(CPACK_NSIS_CONTACT "rishabhnad@gmail.com")
  SET(CPACK_NSIS_MODIFY_PATH ON)
ELSE(WIN32 AND NOT UNIX)
  SET(CPACK_SOURCE_STRIP_FILES "")
ENDIF(WIN32 AND NOT UNIX)
