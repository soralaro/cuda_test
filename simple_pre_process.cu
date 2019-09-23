
// System includes
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>

//#include "dg_types.h"
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#define clip(x, a, b) x >= a ? (x < b ? x : b-1) : a;

        static __host__ __device__ __forceinline__ int DivUp(int total, int grain)
        {
            return (total + grain - 1) / grain;
        }

        __device__ __forceinline__ int clip_ex(int x, int a, int b)
        {
            return (x >= a ? (x < b ? x : b-1) : a);
        }

        __device__ __forceinline__ float clamp_0_255f(float x)
        {
            return (x < 0.0f) ? 0.0f : ((x > 255.0f) ? 255.0f : x);
        }

        __device__ __forceinline__ uchar3 yuv2rgb(uchar y, uchar u, uchar v)
        {
            uchar3 bgr;
            bgr.z = clamp_0_255f( (1.1644f * (y - 0.0f) + 1.5960f * (v - 128.0f))); // r
            bgr.y = clamp_0_255f((1.1644f * (y - 0.0f) - 0.3918f * (u - 128.0f) - 0.8130f * (v - 128.0f))); // g
            bgr.x = clamp_0_255f((1.1644f * (y - 0.0f) + 2.0172f * (u - 128.0f))); // b
            return bgr;
        }


// dxy : (x,y,width,height)
// roi : (x,y,width,height)
        __device__ __forceinline__ float2 calc_fxfy(float4 dxy, float4 roi)
        {
            float scale_x = roi.z / dxy.z;
            float scale_y = roi.w / dxy.w;

            float fx = (dxy.x + 0.5f) * scale_x - 0.5f + roi.x;
            float fy = (dxy.y + 0.5f) * scale_y - 0.5f + roi.y;
            return make_float2(fx, fy);
        }

// return : 4 pixels place ( (x1, y1), (x2, y2) )
        __device__ __forceinline__ int4 get_linear_interpolation(float4 *fxfy, float fx, float fy, int xmax, int dy, int width_plus_one, int height_plus_one)
        {
            float sx = floor(fx);
            fx = fx - sx;
            int isx1 = (int) sx;
            if (isx1 < 0) {
                fx = 0.f;
                isx1 = 0;
            }
            if (isx1 > width_plus_one) {
		xmax = ::min( xmax, dy);
                fx = 0.f;
                isx1 = width_plus_one;
            }

            float sy = floor(fy);
            fy = fy - sy;

            int isy1 = clip(sy - 1 + 1 + 0, 0, (height_plus_one + 1));
            int isy2 = clip(sy - 1 + 1 + 1, 0, (height_plus_one + 1));
            int isx2 = isx1 + 1;

            if( dy > xmax - 1)
	        fxfy[0] = make_float4(1.0, 0.0f, (1.0f-fy), fy);
	    else
		fxfy[0] = make_float4((1.0f-fx), fx, (1.0f-fy), fy);

            return make_int4(isx1, isy1, isx2, isy2);
        }


        __device__ __forceinline__ uchar3 fetch_pixel_bgr_uchar3(cv::cuda::PtrStepSz<uchar3> BGR, int x, int y)
        {
            if (x < BGR.cols && y < BGR.rows) {
                return BGR(y, x);
            } else {
                return(make_uchar3(0, 0, 0));
            }
        }


// fxy4  : ((1-fx), fx, (1-fy), fy)
        __device__ __forceinline__ float3 bgr_uchar3_inter_rgb_float3(float4 fxy, uchar3 pix11, uchar3 pix12, uchar3 pix21, uchar3 pix22)
        {
            float fx_ = fxy.x;
            float fx  = fxy.y;
            float fy_ = fxy.z;
            float fy  = fxy.w;

            float h_rst00, h_rst01;
            float4 pix;

            // B
            pix = make_float4((float)pix11.x, (float)pix12.x, (float)pix21.x, (float)pix22.x);
            h_rst00 = pix.x * fx_ + pix.y * fx;
            h_rst01 = pix.z * fx_ + pix.w * fx;
            float x = h_rst00 * fy_ + h_rst01 * fy;
            // G
            pix = make_float4((float)pix11.y, (float)pix12.y, (float)pix21.y, (float)pix22.y);
            h_rst00 = pix.x * fx_ + pix.y * fx;
            h_rst01 = pix.z * fx_ + pix.w * fx;
            float y = h_rst00 * fy_ + h_rst01 * fy;
            // R
            pix = make_float4((float)pix11.z, (float)pix12.z, (float)pix21.z, (float)pix22.z);
            h_rst00 = pix.x * fx_ + pix.y * fx;
            h_rst01 = pix.z * fx_ + pix.w * fx;
            float z = h_rst00 * fy_ + h_rst01 * fy;

            return make_float3(z, y, x);
        }

        __device__ __forceinline__ float3 bgr_uchar3_inter_bgr_float3(float4 fxy, uchar3 pix11, uchar3 pix12, uchar3 pix21, uchar3 pix22)
        {
            float fx_ = fxy.x;
            float fx  = fxy.y;
            float fy_ = fxy.z;
            float fy  = fxy.w;

            float h_rst00, h_rst01;
            float4 pix;

            // B
            pix = make_float4((float)pix11.x, (float)pix12.x, (float)pix21.x, (float)pix22.x);
            h_rst00 = pix.x * fx_ + pix.y * fx;
            h_rst01 = pix.z * fx_ + pix.w * fx;
            float x = h_rst00 * fy_ + h_rst01 * fy;
            // G
            pix = make_float4((float)pix11.y, (float)pix12.y, (float)pix21.y, (float)pix22.y);
            h_rst00 = pix.x * fx_ + pix.y * fx;
            h_rst01 = pix.z * fx_ + pix.w * fx;
            float y = h_rst00 * fy_ + h_rst01 * fy;
            // R
            pix = make_float4((float)pix11.z, (float)pix12.z, (float)pix21.z, (float)pix22.z);
            h_rst00 = pix.x * fx_ + pix.y * fx;
            h_rst01 = pix.z * fx_ + pix.w * fx;
            float z = h_rst00 * fy_ + h_rst01 * fy;

            return make_float3(x, y, z);
        }


        __device__ __forceinline__ uchar3 fetch_pixel_nv12_uchar3(uchar * nv12, int x, int y, int width, int y_area)
        {
            if (x < width && y < (float)y_area/(float)width) {
                // YUV --> BGR
                uchar y0 = nv12[y * width + x];

                uchar2 uv0;

                int uvAddr = (y / 2) * width + (x & (~0x1)) + y_area;
                reinterpret_cast<unsigned short*>(&uv0)[0] = reinterpret_cast<unsigned short*>(&nv12[uvAddr])[0];

                return yuv2rgb(y0, uv0.x, uv0.y);
            } else {
                return (make_uchar3(0, 0, 0));
            }
        }

// m(alpha_beta) : (alpha, beta[0], beta[1], beta[2])
        __device__ __forceinline__ float3 scale_float3(float3 d0, float4 m)
        {
            float3 d = make_float3( (m.x*d0.x+m.y), (m.x*d0.y+m.z), (m.x*d0.z+m.w));
            return d;
        }

        __device__ __forceinline__ float3 scale_float3_bgr_2_rgb(float3 d0, float4 m)
        {
            float3 d = make_float3( (m.x*d0.z+m.w), (m.x*d0.y+m.z),  (m.x*d0.x+m.y));
            return d;
        }

        __device__ __forceinline__ void store_planar_float3(float *dst, float3 d0, int x, int y, int width, int height)
        {
            int area = width * height;
            int addr = y * width + x;
            dst[addr] = d0.x;
            addr += area;
            dst[addr] = d0.y;
            addr += area;
            dst[addr] = d0.z;
        }

#define Rtype  double

        __global__ void cuda_cv_bgr_uchar3_2_rgb_float3_linear_kernel(
                cv::cuda::PtrStepSz<uchar3> srcBGR, cv::Rect srcROI, float *dstPlanar, cv::Size dstSize, cv::Rect dst_roi, float4 alpha_beta)
        {
            const int dx = blockDim.x * blockIdx.x + threadIdx.x;
            const int dy = blockDim.y * blockIdx.y + threadIdx.y;
            if (dx < dstSize.width && dy < dstSize.height) {
                float3 d0 = make_float3(0.f, 0.f, 0.f);
                if ((dx > dst_roi.x) && (dx < dst_roi.x+dst_roi.width) &&
                    (dy > dst_roi.y) && (dy < dst_roi.y+dst_roi.height))
                {
                    double scale_x = (double)srcROI.width / (double)dst_roi.width;
                    double scale_y = (double)srcROI.height / (double)dst_roi.height;

                    // x1 / x2
                    Rtype fx = (Rtype)((dx+0.5)*scale_x - 0.5);
                    int sx = floor(fx);
                    fx = fx - (Rtype)sx;
                    if ( sx < 0 ){
                        fx = 0.f;
                        sx = 0;
                    }
                    if ( sx > (srcROI.width-1) ){
                        fx = 0.;
                        sx = srcROI.width - 1;
                    }

                    Rtype ialpha0 = (Rtype)1. - fx;
                    Rtype ialpha1 = fx;

                    int sx2 = min(sx+1, srcROI.width-1);

                    // y1 / y2
                    Rtype fy = (Rtype)((dy+0.5)*scale_y - 0.5);
                    int sy0 = floor(fy);
                    fy = fy - (Rtype)sy0;

                    Rtype ibeta0 = (Rtype)1. - fy;
                    Rtype ibeta1 = fy;

                    int sy1 = clip_ex(sy0 + 0, 0, srcROI.height); // [ 0 -- ssize.height-1 ]
                    int sy2 = clip_ex(sy0 + 1, 0, srcROI.height);

                    // 2 - fetch all pixels
                    uchar3 BGR11 = fetch_pixel_bgr_uchar3(srcBGR, (sx+srcROI.x), (sy1+srcROI.y));
                    uchar3 BGR12 = fetch_pixel_bgr_uchar3(srcBGR, (sx2+srcROI.x), (sy1+srcROI.y));
                    uchar3 BGR21 = fetch_pixel_bgr_uchar3(srcBGR, (sx+srcROI.x), (sy2+srcROI.y));
                    uchar3 BGR22 = fetch_pixel_bgr_uchar3(srcBGR, (sx2+srcROI.x), (sy2+srcROI.y));

                    float3 pix11 = make_float3(((float)BGR11.x), ((float)BGR11.y), ((float)BGR11.z));
                    float3 pix12 = make_float3(((float)BGR12.x), ((float)BGR12.y), ((float)BGR12.z));
                    float3 pix21 = make_float3(((float)BGR21.x), ((float)BGR21.y), ((float)BGR21.z));
                    float3 pix22 = make_float3(((float)BGR22.x), ((float)BGR22.y), ((float)BGR22.z));


                    Rtype rst00, rst01;

                    rst00 = (Rtype)pix11.x * ialpha0 + (Rtype)pix12.x * ialpha1;
                    rst01 = (Rtype)pix21.x * ialpha0 + (Rtype)pix22.x * ialpha1;
                    d0.x = (float)(ibeta0 * rst00 + ibeta1 * rst01);

                    rst00 = (Rtype)pix11.y * ialpha0 + (Rtype)pix12.y * ialpha1;
                    rst01 = (Rtype)pix21.y * ialpha0 + (Rtype)pix22.y * ialpha1;
                    d0.y = (float)(ibeta0 * rst00 + ibeta1 * rst01);

                    rst00 = (Rtype)pix11.z * ialpha0 + (Rtype)pix12.z * ialpha1;
                    rst01 = (Rtype)pix21.z * ialpha0 + (Rtype)pix22.z * ialpha1;
                    d0.z = (float)(ibeta0 * rst00 + ibeta1 * rst01);
                }
                float3 d1 = scale_float3_bgr_2_rgb(d0, alpha_beta);
                store_planar_float3(dstPlanar, d1, dx, dy, dstSize.width, dstSize.height);
            }
        }


        __global__ void cuda_cv_bgr_uchar3_2_bgr_float3_linear_kernel(
                cv::cuda::PtrStepSz<uchar3> srcBGR, cv::Rect srcROI, float *dstPlanar, cv::Size dstSize, cv::Rect dst_roi, float4 alpha_beta)
        {
            const int dx = blockDim.x * blockIdx.x + threadIdx.x;
            const int dy = blockDim.y * blockIdx.y + threadIdx.y;
            if (dx < dstSize.width && dy < dstSize.height) {
                float3 d0 = make_float3(0.f, 0.f, 0.f);
                if ((dx > dst_roi.x) && (dx < dst_roi.x+dst_roi.width) &&
                    (dy > dst_roi.y) && (dy < dst_roi.y+dst_roi.height))
                {
                    double scale_x = (double)srcROI.width / (double)dst_roi.width;
                    double scale_y = (double)srcROI.height / (double)dst_roi.height;

                    // x1 / x2
                    Rtype fx = (Rtype)((dx+0.5)*scale_x - 0.5);
                    int sx = floor(fx);
                    fx = fx - (Rtype)sx;
                    if ( sx < 0 ){
                        fx = 0.f;
                        sx = 0;
                    }
                    if ( sx > (srcROI.width-1) ){
                        fx = 0.;
                        sx = srcROI.width - 1;
                    }

                    Rtype ialpha0 = (Rtype)1. - fx;
                    Rtype ialpha1 = fx;

                    int sx2 = min(sx+1, srcROI.width-1);

                    // y1 / y2
                    Rtype fy = (Rtype)((dy+0.5)*scale_y - 0.5);
                    int sy0 = floor(fy);
                    fy = fy - (Rtype)sy0;

                    Rtype ibeta0 = (Rtype)1. - fy;
                    Rtype ibeta1 = fy;

                    int sy1 = clip_ex(sy0 + 0, 0, srcROI.height); // [ 0 -- ssize.height-1 ]
                    int sy2 = clip_ex(sy0 + 1, 0, srcROI.height);

                    // 2 - fetch all pixels
                    uchar3 BGR11 = fetch_pixel_bgr_uchar3(srcBGR, (sx+srcROI.x), (sy1+srcROI.y));
                    uchar3 BGR12 = fetch_pixel_bgr_uchar3(srcBGR, (sx2+srcROI.x), (sy1+srcROI.y));
                    uchar3 BGR21 = fetch_pixel_bgr_uchar3(srcBGR, (sx+srcROI.x), (sy2+srcROI.y));
                    uchar3 BGR22 = fetch_pixel_bgr_uchar3(srcBGR, (sx2+srcROI.x), (sy2+srcROI.y));

                    float3 pix11 = make_float3(((float)BGR11.x), ((float)BGR11.y), ((float)BGR11.z));
                    float3 pix12 = make_float3(((float)BGR12.x), ((float)BGR12.y), ((float)BGR12.z));
                    float3 pix21 = make_float3(((float)BGR21.x), ((float)BGR21.y), ((float)BGR21.z));
                    float3 pix22 = make_float3(((float)BGR22.x), ((float)BGR22.y), ((float)BGR22.z));


                    Rtype rst00, rst01;

                    rst00 = (Rtype)pix11.x * ialpha0 + (Rtype)pix12.x * ialpha1;
                    rst01 = (Rtype)pix21.x * ialpha0 + (Rtype)pix22.x * ialpha1;
                    d0.x = (float)(ibeta0 * rst00 + ibeta1 * rst01);

                    rst00 = (Rtype)pix11.y * ialpha0 + (Rtype)pix12.y * ialpha1;
                    rst01 = (Rtype)pix21.y * ialpha0 + (Rtype)pix22.y * ialpha1;
                    d0.y = (float)(ibeta0 * rst00 + ibeta1 * rst01);

                    rst00 = (Rtype)pix11.z * ialpha0 + (Rtype)pix12.z * ialpha1;
                    rst01 = (Rtype)pix21.z * ialpha0 + (Rtype)pix22.z * ialpha1;
                    d0.z = (float)(ibeta0 * rst00 + ibeta1 * rst01);
                 }

                float3 d1 = scale_float3(d0, alpha_beta);
                store_planar_float3(dstPlanar, d1, dx, dy, dstSize.width, dstSize.height);

            }
        }

        __global__ void cuda_cv_nv12_uchar_2_rgb_float3_linear_kernel(
                uchar * srcNv12, cv::Size srcSize, cv::Rect srcROI, float *dstPlanar, cv::Size dstSize, cv::Rect dst_roi, float4 alpha_beta)
        {
            const int dx = blockDim.x * blockIdx.x + threadIdx.x;
            const int dy = blockDim.y * blockIdx.y + threadIdx.y;
            if (dx < dstSize.width && dy < dstSize.height) {
                float3 d0=make_float3(0.f, 0.f, 0.f);
                if ((dx > dst_roi.x) && (dx < dst_roi.x+dst_roi.width) &&
                    (dy > dst_roi.y) && (dy < dst_roi.y+dst_roi.height))
                {
                    double scale_x = (double)srcROI.width / (double)dst_roi.width;
                    double scale_y = (double)srcROI.height / (double)dst_roi.height;

                    // x1 / x2
                    Rtype fx = (Rtype)((dx+0.5)*scale_x - 0.5);
                    int sx = floor(fx);
                    fx = fx - (Rtype)sx;
                    if ( sx < 0 ){
                        fx = 0.f;
                        sx = 0;
                    }
                    if ( sx > (srcROI.width-1) ){
                        fx = 0.;
                        sx = srcROI.width - 1;
                    }

                    Rtype ialpha0 = (Rtype)1. - fx;
                    Rtype ialpha1 = fx;

                    int sx2 = min(sx+1, srcROI.width-1);

                    // y1 / y2
                    Rtype fy = (Rtype)((dy+0.5)*scale_y - 0.5);
                    int sy0 = floor(fy);
                    fy = fy - (Rtype)sy0;

                    Rtype ibeta0 = (Rtype)1. - fy;
                    Rtype ibeta1 = fy;

                    int sy1 = clip_ex(sy0 + 0, 0, srcROI.height); // [ 0 -- ssize.height-1 ]
                    int sy2 = clip_ex(sy0 + 1, 0, srcROI.height);

                    // 2 - fetch all pixels
                    int y_area = srcSize.width * srcSize.height;
                    int width = srcSize.width;
                    uchar3 BGR11 = fetch_pixel_nv12_uchar3(srcNv12, (sx+srcROI.x), (sy1+srcROI.y), width, y_area);
                    uchar3 BGR12 = fetch_pixel_nv12_uchar3(srcNv12, (sx2+srcROI.x), (sy1+srcROI.y), width, y_area);
                    uchar3 BGR21 = fetch_pixel_nv12_uchar3(srcNv12, (sx+srcROI.x), (sy2+srcROI.y), width, y_area);
                    uchar3 BGR22 = fetch_pixel_nv12_uchar3(srcNv12, (sx2+srcROI.x), (sy2+srcROI.y), width, y_area);

                    float3 pix11 = make_float3(((float)BGR11.x), ((float)BGR11.y), ((float)BGR11.z));
                    float3 pix12 = make_float3(((float)BGR12.x), ((float)BGR12.y), ((float)BGR12.z));
                    float3 pix21 = make_float3(((float)BGR21.x), ((float)BGR21.y), ((float)BGR21.z));
                    float3 pix22 = make_float3(((float)BGR22.x), ((float)BGR22.y), ((float)BGR22.z));


                    Rtype rst00, rst01;

                    rst00 = (Rtype)pix11.x * ialpha0 + (Rtype)pix12.x * ialpha1;
                    rst01 = (Rtype)pix21.x * ialpha0 + (Rtype)pix22.x * ialpha1;
                    d0.x = (float)(ibeta0 * rst00 + ibeta1 * rst01);

                    rst00 = (Rtype)pix11.y * ialpha0 + (Rtype)pix12.y * ialpha1;
                    rst01 = (Rtype)pix21.y * ialpha0 + (Rtype)pix22.y * ialpha1;
                    d0.y = (float)(ibeta0 * rst00 + ibeta1 * rst01);

                    rst00 = (Rtype)pix11.z * ialpha0 + (Rtype)pix12.z * ialpha1;
                    rst01 = (Rtype)pix21.z * ialpha0 + (Rtype)pix22.z * ialpha1;
                    d0.z = (float)(ibeta0 * rst00 + ibeta1 * rst01);
                }

                float3 d1 = scale_float3_bgr_2_rgb(d0, alpha_beta);
                store_planar_float3(dstPlanar, d1, dx, dy, dstSize.width, dstSize.height);
            }
        }

        __global__ void cuda_cv_nv12_uchar_2_bgr_float3_linear_kernel(
                uchar * srcNv12, cv::Size srcSize, cv::Rect srcROI, float *dstPlanar, cv::Size dstSize, cv::Rect dst_roi, float4 alpha_beta)
        {
            const int dx = blockDim.x * blockIdx.x + threadIdx.x;
            const int dy = blockDim.y * blockIdx.y + threadIdx.y;
            if (dx < dstSize.width && dy < dstSize.height) {
                float3 d0 = make_float3(0.f, 0.f, 0.f);
                if ((dx > dst_roi.x) && (dx < dst_roi.x+dst_roi.width) &&
                    (dy > dst_roi.y) && (dy < dst_roi.y+dst_roi.height))
                {
                    double scale_x = (double)srcROI.width / (double)dst_roi.width;
                    double scale_y = (double)srcROI.height / (double)dst_roi.height;

                    // x1 / x2
                    Rtype fx = (Rtype)((dx+0.5)*scale_x - 0.5);
                    int sx = floor(fx);
                    fx = fx - (Rtype)sx;
                    if ( sx < 0 ){
                        fx = 0.f;
                        sx = 0;
                    }
                    if ( sx > (srcROI.width-1) ){
                        fx = 0.;
                        sx = srcROI.width - 1;
                    }

                    Rtype ialpha0 = (Rtype)1. - fx;
                    Rtype ialpha1 = fx;

                    int sx2 = min(sx+1, srcROI.width-1);

                    // y1 / y2
                    Rtype fy = (Rtype)((dy+0.5)*scale_y - 0.5);
                    int sy0 = floor(fy);
                    fy = fy - (Rtype)sy0;

                    Rtype ibeta0 = (Rtype)1. - fy;
                    Rtype ibeta1 = fy;

                    int sy1 = clip_ex(sy0 + 0, 0, srcROI.height); // [ 0 -- ssize.height-1 ]
                    int sy2 = clip_ex(sy0 + 1, 0, srcROI.height);

                    // 2 - fetch all pixels
                    int y_area = srcSize.width * srcSize.height;
                    int width = srcSize.width;
                    uchar3 BGR11 = fetch_pixel_nv12_uchar3(srcNv12, (sx+srcROI.x), (sy1+srcROI.y), width, y_area);
                    uchar3 BGR12 = fetch_pixel_nv12_uchar3(srcNv12, (sx2+srcROI.x), (sy1+srcROI.y), width, y_area);
                    uchar3 BGR21 = fetch_pixel_nv12_uchar3(srcNv12, (sx+srcROI.x), (sy2+srcROI.y), width, y_area);
                    uchar3 BGR22 = fetch_pixel_nv12_uchar3(srcNv12, (sx2+srcROI.x), (sy2+srcROI.y), width, y_area);

                    float3 pix11 = make_float3(((float)BGR11.x), ((float)BGR11.y), ((float)BGR11.z));
                    float3 pix12 = make_float3(((float)BGR12.x), ((float)BGR12.y), ((float)BGR12.z));
                    float3 pix21 = make_float3(((float)BGR21.x), ((float)BGR21.y), ((float)BGR21.z));
                    float3 pix22 = make_float3(((float)BGR22.x), ((float)BGR22.y), ((float)BGR22.z));


                    Rtype rst00, rst01;

                    rst00 = (Rtype)pix11.x * ialpha0 + (Rtype)pix12.x * ialpha1;
                    rst01 = (Rtype)pix21.x * ialpha0 + (Rtype)pix22.x * ialpha1;
                    d0.x = (float)(ibeta0 * rst00 + ibeta1 * rst01);

                    rst00 = (Rtype)pix11.y * ialpha0 + (Rtype)pix12.y * ialpha1;
                    rst01 = (Rtype)pix21.y * ialpha0 + (Rtype)pix22.y * ialpha1;
                    d0.y = (float)(ibeta0 * rst00 + ibeta1 * rst01);

                    rst00 = (Rtype)pix11.z * ialpha0 + (Rtype)pix12.z * ialpha1;
                    rst01 = (Rtype)pix21.z * ialpha0 + (Rtype)pix22.z * ialpha1;
                    d0.z = (float)(ibeta0 * rst00 + ibeta1 * rst01);
                }
                float3 d1 = scale_float3(d0, alpha_beta);
                store_planar_float3(dstPlanar, d1, dx, dy, dstSize.width, dstSize.height);
            }
        }


// input bgr : uchar3
        int SimplePreProcessKernel_bgr(cv::cuda::GpuMat &src, cv::Rect srcROI, float *dst, cv::Size dstSize, cv::Rect dst_roi,
                                       float alpha, float *beta, bool do_bgr2rgb, cudaStream_t cuda_stream)
        {
            dim3 blockS(16, 16);
            dim3 gridS = dim3(DivUp(dstSize.width, blockS.x), DivUp(dstSize.height, blockS.y));

            float4 alpha_beta = make_float4(alpha, beta[0], beta[1], beta[2]);

            if (do_bgr2rgb){
                cuda_cv_bgr_uchar3_2_rgb_float3_linear_kernel <<< gridS, blockS, 0, cuda_stream >>>
                                                                                    (src, srcROI, (float *)dst, dstSize, dst_roi, alpha_beta);
            }
            else {
                cuda_cv_bgr_uchar3_2_bgr_float3_linear_kernel <<< gridS, blockS, 0, cuda_stream >>>
                                                                                    (src, srcROI, (float *)dst, dstSize, dst_roi, alpha_beta);
            }

            return 0;
        }

// input yuv420 : uchar
        int SimplePreProcessKernel_nv12(uchar *src, cv::Size srcSize, cv::Rect srcROI, float *dst, cv::Size dstSize, cv::Rect dst_roi,
                                        float alpha, float *beta, bool do_bgr2rgb, cudaStream_t cuda_stream)
        {
            dim3 blockS(16, 16);
            dim3 gridS = dim3(DivUp(dstSize.width, blockS.x), DivUp(dstSize.height, blockS.y));

            float4 alpha_beta = make_float4(alpha, beta[0], beta[1], beta[2]);

            if (do_bgr2rgb){
                cuda_cv_nv12_uchar_2_rgb_float3_linear_kernel <<< gridS, blockS, 0, cuda_stream >>>
                                                                                    (src, srcSize, srcROI, (float *)dst, dstSize, dst_roi, alpha_beta);
            }
            else {
                cuda_cv_nv12_uchar_2_bgr_float3_linear_kernel <<< gridS, blockS, 0, cuda_stream >>>
                                                                                    (src, srcSize, srcROI, (float *)dst, dstSize, dst_roi, alpha_beta);
            }
            return 0;
        }
