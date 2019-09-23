//
// Created by czx on 19-9-9.
//
#include <stdio.h>
#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include<fstream>
#include <boost/algorithm/string.hpp>
#include "FacePatchProc.h"
#include <glog/logging.h>
int load_image_names_and_landmarks(std::string input_file ,std::vector<std::vector<float>> &landmark, std::vector<std::string> &image_names,std::vector<cv::Rect> &bbox);

__global__ void add( int a, int b, int *c ) {
    *c = a + b;
}
// 两个向量加法kernel，grid和block均为一维
__global__ void add(float* x, float * y, float* z, int n)
{
    // 获取全局索引
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    // 步长
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < n; i += stride)
    {
        z[i] = x[i] + y[i];
    }
}
#if 0
int main()
{
    int N = 1 << 20;
    int nBytes = N * sizeof(float);
    // 申请host内存
    float *x, *y, *z;
    x = (float*)malloc(nBytes);
    y = (float*)malloc(nBytes);
    z = (float*)malloc(nBytes);

    // 初始化数据
    for (int i = 0; i < N; ++i)
    {
        x[i] = 10.0;
        y[i] = 20.0;
    }

    // 申请device内存
    float *d_x, *d_y, *d_z;
    cudaMalloc((void**)&d_x, nBytes);
    cudaMalloc((void**)&d_y, nBytes);
    cudaMalloc((void**)&d_z, nBytes);

    // 将host数据拷贝到device
    cudaMemcpy((void*)d_x, (void*)x, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_y, (void*)y, nBytes, cudaMemcpyHostToDevice);
    // 定义kernel的执行配置
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    // 执行kernel
    add << < gridSize, blockSize >> >(d_x, d_y, d_z, N);

    // 将device得到的结果拷贝到host
    cudaMemcpy((void*)z, (void*)d_z, nBytes, cudaMemcpyHostToDevice);

    // 检查执行结果
    float maxError = 0.0;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(z[i] - 30.0));
    std::cout << "最大误差: " << maxError << std::endl;

    // 释放device内存
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    // 释放host内存
    free(x);
    free(y);
    free(z);

    return 0;
}
#endif
#if 0
int main( void )
{
    int c;
    int *dev_c;
    int dev = 0;
    cudaDeviceProp devProp;
    //CHECK(cudaGetDeviceProperties(&devProp, dev));
    cudaGetDeviceProperties(&devProp, dev);
    std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
    std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
    std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "每个EM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "每个EM的最大线程束数：" << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
    //cudaMalloc()
    cudaMalloc( (void**)&dev_c, sizeof(int) );
    //核函数执行
    add<<<1,1>>>( 2, 7, dev_c );
    //cudaMemcpy()
    cudaMemcpy( &c, dev_c, sizeof(int),cudaMemcpyDeviceToHost ) ;
    printf( "2 + 7 = %d\n", c );
    //cudaFree()
    cudaFree( dev_c );

    return 0;
}
#endif
#if 1
int main(int argc, char** argv)
{
    cv::Mat src;
    cv::cuda::GpuMat cu_src;

    std::string inputFile = argv[1];
    std::vector<std::vector<float>> landmarks;
    std::vector<std::string> image_names;
    std::vector<cv::Rect> bbox;
    load_image_names_and_landmarks(inputFile ,landmarks, image_names,bbox);
    src = cv::imread(image_names[0]);
    std::cout<<"imread file="<<image_names[0]<<std::endl;
    std::vector<cv::Mat> face_patchs;
    std::vector<int> _patch_ids={0,1,2,3,6,7};
    std::vector<float> landmark=landmarks[0];
   // ReadFacePatchImageToData(src,landmark,face_patchs,_patch_ids);

    std::vector<CROP_FACE_RECT> face_patchs_rect;
    ReadFacePatchCropRect(src,landmark,face_patchs_rect,_patch_ids);
    cudaStream_t stream1;
    cudaStreamCreate ( &stream1) ;
    // 初始化一个黑色的GpuMat
    cu_src.upload(src);
    cv::Size dst_size(108,108);
    float alpha=1;
    float beta[3]={0,0,0};
    float *dst_data;
    //cudaMalloc()
    cudaMalloc( (void**)&dst_data, sizeof(float)*108*108*3*6);

    LOG(ERROR) << "InValid_Roi roi_rect1.x=" << face_patchs_rect[3].src.x << "roi_rect1.y=" << face_patchs_rect[3].src.y << "roi_rect1.width="
               << face_patchs_rect[3].src.width << "roi_rect1.height=" << face_patchs_rect[3].src.height
               << "roi_rect2.x=" << face_patchs_rect[3].dst.x << "roi_rect2.y=" << face_patchs_rect[3].dst.y << "roi_rect2.width="
               << face_patchs_rect[3].dst.width << "face_patchs_rect[3].dst.height=" << face_patchs_rect[3].dst.height;
    SimplePreProcessKernel_bgr(cu_src, face_patchs_rect[3].src, dst_data, dst_size, face_patchs_rect[3].dst,
                               alpha, beta, false, stream1);
//    SimplePreProcessKernel_bgr(gpuMat, roi,
//                               (float *)dst[i]->data(), dst_size, dst_roi,
//                               alpha, beta,
//                               toRgb,
//                               cuda_stream);
    cudaStreamSynchronize(stream1);
    cv::Mat cv_dst(108,108,CV_32FC3);
    cudaMemcpy(cv_dst.data,dst_data,sizeof(float)*108*108*3,cudaMemcpyDeviceToHost);
    cv::imwrite("108x108_3.jpg", cv_dst);
    cudaStreamDestroy(stream1);
    std::cout<<"face_patchs.size="<<face_patchs_rect.size()<<std::endl;
    cudaFree( dst_data );
}
#endif
#define LANDMARK_LEN                                144
#define BBOX_LEN                                    4

int load_image_names_and_landmarks(std::string input_file ,std::vector<std::vector<float>> &landmark, std::vector<std::string> &image_names,std::vector<cv::Rect> &bbox)
{
    std::ifstream fp(input_file);
    if (!fp)
    {
        std::cout << "Can't open list file " << input_file << std::endl;
        exit(-1);
    }
    int job_count = 0;
    std::string one_job;
    while (getline(fp, one_job))
    {
        int count = 0;
        std::vector<std::string> vec_str;
        boost::split(vec_str, one_job, boost::is_any_of(" "));
		std::cout << "vec_str.size(): " << vec_str.size() << std::endl;
        assert(vec_str.size() == 1 + BBOX_LEN + LANDMARK_LEN);

        image_names.push_back(vec_str[0]);

        cv::Rect one_bbox;
        one_bbox.x = std::stoi(vec_str[count + 1]);
        one_bbox.y = std::stoi(vec_str[count + 2]);
        one_bbox.width = std::stoi(vec_str[count + 3]);
        one_bbox.height = std::stoi(vec_str[count + 4]);
        bbox.push_back(one_bbox);
        count = count + 1 + BBOX_LEN;

        std::vector<float> one_landmark;
        for (int i = count; i < count + LANDMARK_LEN; ++i)
        {
		    one_landmark.push_back(std::stof(vec_str[i]));
        }
		landmark.push_back(one_landmark);
        job_count++;
    }
    return job_count;
}
