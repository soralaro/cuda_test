//
// Created by czx on 19-9-17.
//

#include "FacePatchProc.h"
#include <glog/logging.h>


bool Valid_roi(const cv::Mat m,cv::Rect roi)
{
    if(0 <= roi.x && 0 < roi.width && roi.x + roi.width <= m.cols &&0 <= roi.y && 0 < roi.height && roi.y + roi.height <= m.rows)
    {
        return true;
    }
    return false;
}
bool Valid_roi(const int max_w,const int max_h,cv::Rect roi)
{
    if(0 <= roi.x && 0 < roi.width && roi.x + roi.width <=max_w &&0 <= roi.y && 0 < roi.height && roi.y + roi.height <= max_h)
    {
        return true;
    }
    return false;
}

cv::Mat CropImage(const cv::Mat &image, int crop_height, int crop_width, int center_x, int center_y)
{

    cv::Mat result(cv::Mat::zeros(crop_height, crop_width, image.type()));
    assert(!image.empty() && image.rows >= crop_height && image.cols >= crop_width);
    if (center_x <= 0 || center_y <= 0)
    {
        // TODO: if x < 0 || y < 0, why center should be set to the image center?
        center_x = image.cols / 2;
        center_y = image.rows / 2;
    }

    int left1, top1, right1, bottom1;
    int left2, top2, right2, bottom2;
    int overlop_left, overlop_top;
    int overlop_right, overlop_bottom;
    left1 = 0, top1 = 0;
    right1 = image.cols, bottom1 = image.rows;
    left2 = center_x - crop_width / 2;
    top2 = center_y - crop_height / 2;
    right2 = left2 + crop_width;
    bottom2 = top2 + crop_height;

    overlop_left = MAX(left1, left2);
    overlop_right = MIN(right1, right2);
    overlop_top = MAX(top1, top2);
    overlop_bottom = MIN(bottom1, bottom2);

    cv::Rect roi_rect1(overlop_left, overlop_top, overlop_right - overlop_left,
                       overlop_bottom - overlop_top);

    left1 = 0, top1 = 0;
    right1 = crop_width, bottom1 = crop_height;
    left2 = crop_width / 2 - center_x;
    top2 = crop_height /2 - center_y;
    right2 = left2 + image.cols;
    bottom2 = top2 + image.rows;

    overlop_left = MAX(left1, left2);
    overlop_right = MIN(right1, right2);
    overlop_top = MAX(top1, top2);
    overlop_bottom = MIN(bottom1, bottom2);

    cv::Rect roi_rect2(overlop_left, overlop_top, overlop_right - overlop_left,
                       overlop_bottom - overlop_top);
    if(Valid_roi(image,roi_rect1)&&Valid_roi(result,roi_rect2))
        image(roi_rect1).copyTo(result(roi_rect2));
    else
        LOG(ERROR) << "InValid_Roi roi_rect1.x="<<roi_rect1.x<<"roi_rect1.y="<<roi_rect1.y<<"roi_rect1.width="<<roi_rect1.width<<"roi_rect1.height="<<roi_rect1.height
                   <<"roi_rect2.x="<<roi_rect2.x<<"roi_rect2.y="<<roi_rect2.y<<"roi_rect2.width="<<roi_rect2.width<<"roi_rect2.height="<<roi_rect2.height;
    return result;
}

void ReadImageToData(const cv::Mat& img, const std::vector<float>& patch_landmarks, std::vector<cv::Mat> &face_patchs)
{
    // assert(img.rows == _ref_img_size && img.cols == _ref_img_size);
    const int output_size = 108;

    for(size_t i = 0; i < patch_landmarks.size() / 2; ++i)
    {
        int temp_patch_size = 0;
        if (patch_landmarks[i * 2] == 0 && patch_landmarks[i * 2 + 1] == 0)
        {
            temp_patch_size = 225;
        }
        else if (patch_landmarks[i * 2] == -1 && patch_landmarks[i * 2 + 1] == -1)
        {
            temp_patch_size = 169;
        }
        else
        {
            temp_patch_size = 108;
        }

        cv::Mat crop_img;
        if (i < patch_landmarks.size() /2 - 1)
        {
            crop_img = CropImage(img, temp_patch_size, temp_patch_size,
                                 patch_landmarks[i*2], patch_landmarks[i*2+1]);
        }
        else
        {
            crop_img = CropImage(img, temp_patch_size, temp_patch_size,
                                 patch_landmarks[i*2], patch_landmarks[i*2+1]);
        }

        cv::Mat resize_img;
        if (temp_patch_size == output_size)
        {
            resize_img = crop_img;
        }
        else
        {
            cv::resize(crop_img, resize_img, cv::Size(output_size, output_size));
        }
        std::string patch_img_name = "patch_" + std::to_string(i) + ".jpg";
        imwrite(patch_img_name, resize_img);

        face_patchs.push_back(resize_img);
    }

}
void ReadFacePatchImageToData(const cv::Mat& img,const std::vector<float> & landmarks,std::vector<cv::Mat> &face_patchs,std::vector<int > _patch_ids)
{
    std::vector<float> patch_landmarks;
    float mark_x, mark_y;
    enum face_patch {main_face = 0, left_eye, right_eye, nose, mouth, middle_eye, left_mouth, right_mouth, mini_face, left_brow, right_brow, middle_brow};
    for (unsigned int i = 0; i < _patch_ids.size(); i++)
    {
        switch(_patch_ids[i])
        {
            case main_face:
                patch_landmarks.push_back(0);
                patch_landmarks.push_back(0);
                break;
            case left_eye:
                patch_landmarks.push_back(landmarks[21 * 2 + 0]);
                patch_landmarks.push_back(landmarks[21 * 2 + 1]);
                break;
            case right_eye:
                patch_landmarks.push_back(landmarks[38 * 2 + 0]);
                patch_landmarks.push_back(landmarks[38 * 2 + 1]);
                break;
            case nose:
                patch_landmarks.push_back(landmarks[57 * 2 + 0]);
                patch_landmarks.push_back(landmarks[57 * 2 + 1]);
                break;
            case mouth:
                mark_x = (landmarks[58 * 2] + landmarks[62 * 2])/2 ;
                mark_y = (landmarks[58 * 2 + 1] + landmarks[62 * 2 + 1])/2 ;
                patch_landmarks.push_back(mark_x);
                patch_landmarks.push_back(mark_y);
                break;
            case middle_eye:
                mark_x = (landmarks[21 * 2] + landmarks[38 * 2])/2 ;
                mark_y = (landmarks[21 * 2 + 1] + landmarks[38 * 2 + 1])/2 ;
                patch_landmarks.push_back(mark_x);
                patch_landmarks.push_back(mark_y);
                break;
            case left_mouth:
                patch_landmarks.push_back(landmarks[58 * 2 + 0]);
                patch_landmarks.push_back(landmarks[58 * 2 + 1]);
                break;
            case right_mouth:
                patch_landmarks.push_back(landmarks[62 * 2 + 0]);
                patch_landmarks.push_back(landmarks[62 * 2 + 1]);
                break;
            case mini_face:
                patch_landmarks.push_back(-1);
                patch_landmarks.push_back(-1);
                break;
            case left_brow:
                patch_landmarks.push_back(landmarks[24*2]);
                patch_landmarks.push_back(landmarks[24*2+1]);
                break;
            case right_brow:
                patch_landmarks.push_back(landmarks[41*2]);
                patch_landmarks.push_back(landmarks[41*2+1]);
                break;
            case middle_brow:
                mark_x = (landmarks[24 * 2] + landmarks[41 * 2])/2 ;
                mark_y = (landmarks[24 * 2 + 1] + landmarks[41 * 2 + 1])/2 ;
                patch_landmarks.push_back(mark_x);
                patch_landmarks.push_back(mark_y);
                break;
            default:
                patch_landmarks.push_back(0);
                patch_landmarks.push_back(0);
                break;
        }
    }
    // ReadImageToData(img, 128, 128, 192, 192, 108,
    //     patch_landmarks, transformed_data);
    ReadImageToData(img, patch_landmarks, face_patchs);
}

int CropDstRet(const cv::Mat &image, int crop_height, int crop_width, int center_x, int center_y,CROP_FACE_RECT &src_dst_rect)
{

    assert(!image.empty() && image.rows >= crop_height && image.cols >= crop_width);
    if (center_x <= 0 || center_y <= 0)
    {
        // TODO: if x < 0 || y < 0, why center should be set to the image center?
        center_x = image.cols / 2;
        center_y = image.rows / 2;
    }

    int left1, top1, right1, bottom1;
    int left2, top2, right2, bottom2;
    int overlop_left, overlop_top;
    int overlop_right, overlop_bottom;
    left1 = 0, top1 = 0;
    right1 = image.cols, bottom1 = image.rows;
    left2 = center_x - crop_width / 2;
    top2 = center_y - crop_height / 2;
    right2 = left2 + crop_width;
    bottom2 = top2 + crop_height;

    overlop_left = MAX(left1, left2);
    overlop_right = MIN(right1, right2);
    overlop_top = MAX(top1, top2);
    overlop_bottom = MIN(bottom1, bottom2);

    cv::Rect roi_rect1(overlop_left, overlop_top, overlop_right - overlop_left,
                       overlop_bottom - overlop_top);

    left1 = 0, top1 = 0;
    right1 = crop_width, bottom1 = crop_height;
    left2 = crop_width / 2 - center_x;
    top2 = crop_height /2 - center_y;
    right2 = left2 + image.cols;
    bottom2 = top2 + image.rows;

    overlop_left = MAX(left1, left2);
    overlop_right = MIN(right1, right2);
    overlop_top = MAX(top1, top2);
    overlop_bottom = MIN(bottom1, bottom2);

    cv::Rect roi_rect2(overlop_left, overlop_top, overlop_right - overlop_left,
                       overlop_bottom - overlop_top);
    if(Valid_roi(image,roi_rect1)&&Valid_roi(crop_width,crop_height,roi_rect2)) {
        src_dst_rect.src=roi_rect1;
        src_dst_rect.dst=roi_rect2;
       // return 0;
    }
  //else {
        LOG(ERROR) << "InValid_Roi roi_rect1.x=" << roi_rect1.x << "roi_rect1.y=" << roi_rect1.y << "roi_rect1.width="
                   << roi_rect1.width << "roi_rect1.height=" << roi_rect1.height
                   << "roi_rect2.x=" << roi_rect2.x << "roi_rect2.y=" << roi_rect2.y << "roi_rect2.width="
                   << roi_rect2.width << "roi_rect2.height=" << roi_rect2.height;
   //     return -1;
   return 0;
    //}
}

void PatchRect(const cv::Mat& img, const std::vector<float>& patch_landmarks, std::vector<CROP_FACE_RECT> &face_patchs)
{
    // assert(img.rows == _ref_img_size && img.cols == _ref_img_size);
    const int output_size = 108;

    for(size_t i = 0; i < patch_landmarks.size() / 2; ++i)
    {
        int temp_patch_size = 0;
        if (patch_landmarks[i * 2] == 0 && patch_landmarks[i * 2 + 1] == 0)
        {
            temp_patch_size = 225;
        }
        else if (patch_landmarks[i * 2] == -1 && patch_landmarks[i * 2 + 1] == -1)
        {
            temp_patch_size = 169;
        }
        else
        {
            temp_patch_size = 108;
        }

        CROP_FACE_RECT src_dst_rect;
        if(0==CropDstRet(img,temp_patch_size, temp_patch_size, patch_landmarks[i*2],patch_landmarks[i*2+1],src_dst_rect))
        {
            face_patchs.push_back(src_dst_rect);
        }
    }

}
void ReadFacePatchCropRect(const cv::Mat& img,const std::vector<float> & landmarks,std::vector<CROP_FACE_RECT> &face_patchs,std::vector<int > _patch_ids)
{
    std::vector<float> patch_landmarks;
    float mark_x, mark_y;
    enum face_patch {main_face = 0, left_eye, right_eye, nose, mouth, middle_eye, left_mouth, right_mouth, mini_face, left_brow, right_brow, middle_brow};
    for (unsigned int i = 0; i < _patch_ids.size(); i++)
    {
        switch(_patch_ids[i])
        {
            case main_face:
                patch_landmarks.push_back(0);
                patch_landmarks.push_back(0);
                break;
            case left_eye:
                patch_landmarks.push_back(landmarks[21 * 2 + 0]);
                patch_landmarks.push_back(landmarks[21 * 2 + 1]);
                break;
            case right_eye:
                patch_landmarks.push_back(landmarks[38 * 2 + 0]);
                patch_landmarks.push_back(landmarks[38 * 2 + 1]);
                break;
            case nose:
                patch_landmarks.push_back(landmarks[57 * 2 + 0]);
                patch_landmarks.push_back(landmarks[57 * 2 + 1]);
                break;
            case mouth:
                mark_x = (landmarks[58 * 2] + landmarks[62 * 2])/2 ;
                mark_y = (landmarks[58 * 2 + 1] + landmarks[62 * 2 + 1])/2 ;
                patch_landmarks.push_back(mark_x);
                patch_landmarks.push_back(mark_y);
                break;
            case middle_eye:
                mark_x = (landmarks[21 * 2] + landmarks[38 * 2])/2 ;
                mark_y = (landmarks[21 * 2 + 1] + landmarks[38 * 2 + 1])/2 ;
                patch_landmarks.push_back(mark_x);
                patch_landmarks.push_back(mark_y);
                break;
            case left_mouth:
                patch_landmarks.push_back(landmarks[58 * 2 + 0]);
                patch_landmarks.push_back(landmarks[58 * 2 + 1]);
                break;
            case right_mouth:
                patch_landmarks.push_back(landmarks[62 * 2 + 0]);
                patch_landmarks.push_back(landmarks[62 * 2 + 1]);
                break;
            case mini_face:
                patch_landmarks.push_back(-1);
                patch_landmarks.push_back(-1);
                break;
            case left_brow:
                patch_landmarks.push_back(landmarks[24*2]);
                patch_landmarks.push_back(landmarks[24*2+1]);
                break;
            case right_brow:
                patch_landmarks.push_back(landmarks[41*2]);
                patch_landmarks.push_back(landmarks[41*2+1]);
                break;
            case middle_brow:
                mark_x = (landmarks[24 * 2] + landmarks[41 * 2])/2 ;
                mark_y = (landmarks[24 * 2 + 1] + landmarks[41 * 2 + 1])/2 ;
                patch_landmarks.push_back(mark_x);
                patch_landmarks.push_back(mark_y);
                break;
            default:
                patch_landmarks.push_back(0);
                patch_landmarks.push_back(0);
                break;
        }
    }
    // ReadImageToData(img, 128, 128, 192, 192, 108,
    //     patch_landmarks, transformed_data);
    PatchRect(img, patch_landmarks, face_patchs);
}