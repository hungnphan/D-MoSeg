//
// @file    : util.h
// @purpose : A header class for data conversion between torch::Tensor and cv::Mat
// @author  : Hung Ngoc Phan
// @project : Distributed training of Foreground Detection Network with Event-Triggered Data Parallelism
// @licensed: N/A
// @created : 10/05/2021
// @modified: 13/05/2021
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _UTIL_H_
#define _UTIL_H_

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

///////////////////////////////////////////////////////////////////////
// Data conversion: cv::Mat <-----------> torch::Tensor
///////////////////////////////////////////////////////////////////////

cv::Mat cvt_torchTensor_to_cvMat(torch::Tensor& tsr){
    cv::Mat mat;
    // BGR image
    if(tsr.dim() == 3)        
        mat = cv::Mat(tsr.size(1), tsr.size(0), CV_8UC3);
    // Grayscale image
    else if(tsr.dim() == 2)
        mat = cv::Mat(tsr.size(1), tsr.size(0), CV_8U);
        
    torch::Tensor flatten_tsr = tsr.contiguous();
    std::memcpy(mat.data, (uint8_t*) flatten_tsr.data_ptr(), sizeof(unsigned char) * tsr.numel());
    return mat;
}

torch::Tensor cvt_cvMat_to_torchTensor(cv::Mat& mat){
    torch::Tensor tsr;
    // BGR image
    if(mat.channels() == 3){    
        tsr = torch::zeros({mat.cols, mat.rows, mat.channels()},
                                         torch::TensorOptions(torch::kUInt8) );
        memcpy(tsr.data_ptr(), mat.data, tsr.numel() * sizeof(unsigned char));    
    }
    // Grayscale image
    else if(mat.channels() == 1){   
        tsr = torch::zeros({mat.cols, mat.rows},
                                         torch::TensorOptions(torch::kUInt8) );
        memcpy(tsr.data_ptr(), mat.data, tsr.numel() * sizeof(unsigned char));
    }
    return tsr;
}

#endif