cmake_minimum_required(VERSION 3.5)
project(prune_tiny_yolo)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
set(SRC_DIR "./src")









include_directories("./include" "./src")
#file(GLOB darknet_src ./src/*.c ./src/*.cu)
#add_library(darknet ${darknet_src} )
#target_link_libraries(darknet pthread m)



file(GLOB example_src ./examples/*.c)


#add_executable(object_detection ${example_src})
#target_link_libraries(object_detection darknet)
function(detect_cuDNN)
    set(CUDNN_ROOT "" CACHE PATH "CUDNN root folder")

    find_path(CUDNN_INCLUDE cudnn.h
            PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDA_TOOLKIT_INCLUDE}
            DOC "Path to cuDNN include directory." )

    # dynamic libs have different suffix in mac and linux
    if(APPLE)
        set(CUDNN_LIB_NAME "libcudnn.dylib")
    else()
        set(CUDNN_LIB_NAME "libcudnn.so")
    endif()

    get_filename_component(__libpath_hist ${CUDA_CUDART_LIBRARY} PATH)
    find_library(CUDNN_LIBRARY NAMES ${CUDNN_LIB_NAME}
            PATHS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDNN_INCLUDE} ${__libpath_hist} ${__libpath_hist}/../lib
            DOC "Path to cuDNN library.")

    if(CUDNN_INCLUDE AND CUDNN_LIBRARY)
        set(HAVE_CUDNN  TRUE PARENT_SCOPE)
        set(CUDNN_FOUND TRUE PARENT_SCOPE)

        file(READ ${CUDNN_INCLUDE}/cudnn.h CUDNN_VERSION_FILE_CONTENTS)

        # cuDNN v3 and beyond
        string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
                CUDNN_VERSION_MAJOR "${CUDNN_VERSION_FILE_CONTENTS}")
        string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
                CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR}")
        string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
                CUDNN_VERSION_MINOR "${CUDNN_VERSION_FILE_CONTENTS}")
        string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
                CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR}")
        string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
                CUDNN_VERSION_PATCH "${CUDNN_VERSION_FILE_CONTENTS}")
        string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
                CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH}")

        if(NOT CUDNN_VERSION_MAJOR)
            set(CUDNN_VERSION "???")
        else()
            set(CUDNN_VERSION "${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
        endif()

        message(STATUS "Found cuDNN: ver. ${CUDNN_VERSION} found (include: ${CUDNN_INCLUDE}, library: ${CUDNN_LIBRARY})")

        string(COMPARE LESS "${CUDNN_VERSION_MAJOR}" 3 cuDNNVersionIncompatible)
        if(cuDNNVersionIncompatible)
            message(FATAL_ERROR "cuDNN version >3 is required.")
        endif()

        set(CUDNN_VERSION "${CUDNN_VERSION}" PARENT_SCOPE)
        mark_as_advanced(CUDNN_INCLUDE CUDNN_LIBRARY CUDNN_ROOT)

    endif()
endfunction()

detect_cuDNN()

find_package(CUDA QUIET REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
if (CUDA_FOUND)
    message(STATUS "CUDA Version: " ${CUDA_VERSION_STRINGS})
    message(STATUS "CUDA Libararies: " ${CUDA_LIBRARIES})
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;-O3;-Xcompiler;-fPIC;")
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS} ${CUDNN_INCLUDE})
    list(APPEND LIBRARIES ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY} ${CUDA_cusparse_LIBRARY} ${CUDNN_LIBRARY})
    add_definitions(-DGPU  -DCUDNN)
    cuda_include_directories(src)
else()
    list(APPEND LIBRARIES "m")
endif()


if (CUDA_FOUND)
    FILE(GLOB CU_SRC_FILES ${SRC_DIR}/*.cu ${SRC_DIR}/*.c)

    CUDA_INCLUDE_DIRECTORIES( ${SRC_DIR} )

    set(CUDA_ARCH_BIN "20 30 32 35 37 50 52" CACHE STRING "Specify 'real' GPU arch to build binaries for, BIN(PTX) format is supported. Example: 1.3 2.1(1.3) or 13 21(13)")
    set(CUDA_ARCH_PTX "" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")

    include(CudaComputeTargetFlags.cmake)
    APPEND_TARGET_ARCH_FLAGS()

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false")

    CUDA_COMPILE(cuda_objs ${CU_SRC_FILES})

    message(STATUS "CUDA FOUND")
endif()







cuda_add_executable(darknet ${example_src})
add_library(libdarknet  ${SRC_FILES} ${CU_SRC_FILES} ${cuda_objs})
target_link_libraries( darknet  libdarknet ${LIBRARIES})