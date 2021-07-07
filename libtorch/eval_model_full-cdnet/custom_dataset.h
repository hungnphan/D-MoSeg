//
// @file    : custom_dataset.h
// @purpose : A header file for data loader using foreground-evaluation custom dataset
// @author  : Hung Ngoc Phan
// @project : Distributed training of Foreground Detection Network with Event-Triggered Data Parallelism
// @licensed: N/A
// @created : 03/07/2021
// @modified: 07/07/2021
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
    CustomDataset(int           proc_id, 
                  int           num_proc,
                  int           batch_sz,
                  std::string           data_dir, 
                  std::vector<std::pair<std::string,std::string>> data_sequences)
      : proc_id          {proc_id},    
        num_proc         {num_proc},
        batch_sz         {batch_sz},
        data_dir         {data_dir},
        data_sequences   {data_sequences} {
            
        // Init the iterative indices
        this->sample_idx    = 0;
        this->sequence_idx  = 0;      

        this->sequence_size = {
            6760, 3660, 6260, 3260, 1460, 1810, 859, 960, 910, 2260, 960, 1330, 7759, 
            949, 3760, 944, 1259, 2760, 4260, 2260, 2510, 2960, 2260, 2760, 660, 3760, 
            1260, 2260, 2520, 1124, 4960, 2760, 1545, 1760, 1460, 1010, 3160, 7160, 959, 
            5160, 3460, 6260, 4660, 360, 4760, 3760, 4260, 1960
        };  
    }

    torch::data::Example<> get(size_t index) override{

        int data_idx = index;
        
        // Set up scenario and sequence name
        this->scenario_name = this->data_sequences[this->sequence_idx].first;
        this->sequence_name = this->data_sequences[this->sequence_idx].second;

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
        
        // Debug the file loading from dataloader
        // std::cout << data.sizes() << " " << fg_tsr.sizes() << "\t" << path_to_data << "\n";

        // Transpose: [W,H,C] -> [C,W,H]
        data = data.permute({2,0,1}).contiguous();
        fg_tsr = fg_tsr.permute({2,0,1}).contiguous();       
        

        /////////////////////////////////////////////////////////
        // USE MULTIPLE DATA SEQUENCES
        /////////////////////////////////////////////////////////

        // Iterative through multiple sequences
        this->sample_idx += 1;

        // Jump to next sequence
        if(this->sample_idx == (this->sequence_size[this->sequence_idx]-1)){            
            this->sequence_idx = (this->sequence_idx + 1) % (this->data_sequences.size());
            this->sample_idx = 0;
        }


        return {data, fg_tsr};  
    }

    torch::optional<size_t> size() const override {
        return this->sequence_size[this->sequence_idx];
    }
    
private:
    
    int                 proc_id;
    int                 num_proc;
    int                 batch_sz;

    int                 sample_idx;
    int                 sequence_idx;
    
    std::string         data_dir;
    std::string         scenario_name;
    std::string         sequence_name;
    
    std::vector<std::pair<std::string,std::string>> data_sequences;
    std::vector<int>    sequence_size;
};

#endif