project(CUDA_TEST)
cmake_minimum_required(VERSION 3.5)
set (TARGET_NAME "CUDA_TEST")

# packages
find_package(CUDA)

# nvcc flags
set(CUDA_NVCC_FLAGS -O3;-G;-g)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;-std=c++11;)
list(APPEND SRC_FILES
    ${PROJECT_SOURCE_DIR}/main.cu
    ${PROJECT_SOURCE_DIR}/FacePatchProc.cu
    ${PROJECT_SOURCE_DIR}/simple_pre_process.cu
)
link_directories(/usr/local/opencv-3.4/lib)
CUDA_ADD_EXECUTABLE(${TARGET_NAME} ${SRC_FILES})

target_link_libraries(${TARGET_NAME}
        -lopencv_core
        -lopencv_imgproc
        -lopencv_imgcodecs
        -lglog
        )