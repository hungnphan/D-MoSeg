//
// @file    : custom_dataset.h
// @purpose : A header file for data loader using foreground-training custom dataset
// @author  : Hung Ngoc Phan
// @project : Distributed training of Foreground Detection Network with Event-Triggered Data Parallelism
// @licensed: N/A
// @created : 10/05/2021
// @modified: 15/05/2021
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _CUSTOM_DATASET_H_
#define _CUSTOM_DATASET_H_

#include "util.h"

///////////////////////////////////////////////////////////////////////
// Custom dataset class
///////////////////////////////////////////////////////////////////////
class CustomDataset : public torch::data::datasets::Dataset<CustomDataset> {

public:
    CustomDataset(int            proc_id, 
                  int            num_proc,
                  std::string    data_dir, 
                  std::string    scenario_name, 
                  std::string    sequence_name)
      : proc_id          {proc_id},    
        num_proc         {num_proc},
        data_dir         {data_dir},
        scenario_name    {scenario_name},
        sequence_name    {sequence_name},
        sample_indices   {std::vector<int>()}{
            
        // split number of frames for each worker
        this->split_data_for_workers();
    }

    torch::data::Example<> get(size_t index) override{
        int data_idx = sample_indices[index];
        // std::cout << "Getting data of index #" << data_idx << std::endl;
        
        // Convert int to string with leading zeros
        std::stringstream ss;
        ss << std::setw(6) << std::setfill('0') << data_idx;
        std::string index_str = ss.str();
        
        // Path to frame data
        std::string path_to_data = this->data_dir + "/" 
            + this->scenario_name + "/" 
            + this->sequence_name + "/"
            + index_str;
                
        // Read in, bg, fg images with opencv
        cv::Mat in_img = cv::imread(path_to_data + "_in.png", cv::IMREAD_COLOR);
        cv::Mat bg_img = cv::imread(path_to_data + "_bg.png", cv::IMREAD_COLOR);
        cv::Mat fg_img = cv::imread(path_to_data + "_fg.png", cv::IMREAD_GRAYSCALE);
                
        // Convert cv::Mat to torch::Tensor
        torch::Tensor in_tsr = cvt_cvMat_to_torchTensor(in_img);                // [W, H, 3]
        torch::Tensor bg_tsr = cvt_cvMat_to_torchTensor(bg_img);                // [W, H, 3]
        torch::Tensor fg_tsr = cvt_cvMat_to_torchTensor(fg_img).unsqueeze(-1);  // [W, H, 1]
        
        // Concat data tensor: [input_img, bg_img, fg_img] with shape [W, H, 7]
        torch::Tensor data = torch::cat({in_tsr, bg_tsr}, -1);  
         
        // Transpose: [W,H,C] -> [C,W,H]
        data = data.permute({2,0,1}).contiguous();
        fg_tsr = fg_tsr.permute({2,0,1}).contiguous();       
        
        return {data, fg_tsr};  
    }

    torch::optional<size_t> size() const override {
        return sample_indices.size();
    }
    
private:

    void split_data_for_workers(){
        // There are 200 labelled frames in CDnet dataset for foreground training
        int nFrames = 200;
        
        // Iterate through a data sequence:
        // Frame index: 1,2,3,4,...200
        for(int frame_idx=1;frame_idx<=nFrames;frame_idx++){
            if((frame_idx-1) % num_proc == proc_id)
                sample_indices.push_back(frame_idx);
        }
        return;
    }    
    
    int                 proc_id;
    int                 num_proc;
    
    std::string         data_dir;
    std::string         scenario_name;
    std::string         sequence_name;
    
    std::vector<int>    sample_indices;
};

#endif