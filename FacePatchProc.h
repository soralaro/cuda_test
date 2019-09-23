//
// Created by czx on 19-9-17.
//

#ifndef CUDA_TEST_FACEPATCHPROC_H
#define CUDA_TEST_FACEPATCHPROC_H
#include "opencv2/opencv.hpp"
typedef struct {
    cv::Rect  src;
    cv::Rect  dst;
} CROP_FACE_RECT;
void ReadFacePatchImageToData(const cv::Mat &img,const std::vector<float> &landmarks,std::vector<cv::Mat> &face_patchs,std::vector<int> _patch_ids);
void ReadFacePatchCropRect(const cv::Mat& img,const std::vector<float> & landmarks,std::vector<CROP_FACE_RECT> &face_patchs,std::vector<int > _patch_ids);
int SimplePreProcessKernel_bgr(cv::cuda::GpuMat &src, cv::Rect srcROI, float *dst, cv::Size dstSize, cv::Rect dst_roi,
                               float alpha, float *beta, bool do_bgr2rgb, cudaStream_t cuda_stream);
int SimplePreProcessKernel_nv12(uchar *src, cv::Size srcSize, cv::Rect srcROI, float *dst, cv::Size dstSize, cv::Rect dst_roi,
                                float alpha, float *beta, bool do_bgr2rgb, cudaStream_t cuda_stream);
#endif //CUDA_TEST_FACEPATCHPROC_H
