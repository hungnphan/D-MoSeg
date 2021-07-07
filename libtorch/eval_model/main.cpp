//
// @file    : main.cpp
// @purpose : An evaluating scheme to evaluate the model training on CDnet-2014
// @author  : Hung Ngoc Phan
// @project : Distributed training of Foreground Detection Network with Event-Triggered Data Parallelism
// @licensed: N/A
// @created : 04/07/2021
// @modified: 07/07/2021
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include <iostream>             // standard I/O
#include <cstdio>
#include <cmath>

#include <algorithm>            // STL data structures and algorithms
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include <tuple>

#include <fstream>              // file stream and os stream
#include <sstream>

#include <iomanip>              // formating and misc
#include <stdint.h>

#include <chrono>               // timing
#include <ctime>

#include <sys/types.h>          // file system linux
#include <sys/stat.h>
#include <unistd.h>

#include "custom_dataset.h"     // data loader for custom dataset
#include "util.h"               // data conversion: torch::Tensor and cv::Mat
#include "model.h"              // implementation of FDN on TorchLib
#include "config.h"             // configuration of data and training

#define bug(x) std::cout << #x << " = " << x << "\n"

// Dictionary of dataset CDnet2014
std::map<std::string, std::vector<std::string>> const cdnet_data {
   { "badWeather",                  { "blizzard","skating","snowFall","wetSnow" } },
   { "baseline",                    { "highway","office","pedestrians","PETS2006" } },
   { "cameraJitter",                { "badminton","boulevard","sidewalk","traffic" } },
   { "dynamicBackground",           { "boats","canoe","fall","fountain01","fountain02","overpass" } },
   { "intermittentObjectMotion",    { "abandonedBox","parking","sofa","tramstop","winterDriveway" } }, // "streetLight",
   { "lowFramerate",                { "port_0_17fps","tramCrossroad_1fps","tunnelExit_0_35fps","turnpike_0_5fps" } },
   { "nightVideos",                 { "bridgeEntry","busyBoulvard","fluidHighway","streetCornerAtNight","tramStation","winterStreet" } },
//    { "PTZ",                         { "continuousPan","intermittentPan","twoPositionPTZCam","zoomInZoomOut" } },
   { "shadow",                      { "backdoor","bungalows","busStation","copyMachine","cubicle","peopleInShade" } },
   { "thermal",                     { "corridor","diningRoom","lakeSide","library","park" } },
   { "turbulence",                  { "turbulence0","turbulence1","turbulence2","turbulence3" } }
};

const int kNumSequence = 48;

std::vector<std::pair<std::string,std::string>> scatter_data_for_eval();


////////////////////////////////////////////////////////////////////////////////////////////////
// ------- ARGUMENT DESCRIPTION -------
// ------------------------------------
// <num_proc>           : int : The number of processes used in model training
// <thres_value>        : int : threshold value
// <train_batch_size>   : int : batch size for training
// <gpu_index>          : int : The index of gpu used for evaluating, default is 0
// <eval_batch_size>    : int : batch size for evaluation
////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[]){

    /////////////////////////////////////////////////////////////////////
    // ARGUMENT ASSESSMENT
    /////////////////////////////////////////////////////////////////////
    if(argc < 5){
        std::cout << "Please input all necessary arguments:\n";
        std::cout << "------------------------------------\n";
        std::cout << "./run <num_proc> <thres_value> <batch_size> <gpu_index>\n";
        std::cout << "------------------------------------\n";
        std::cout << " <num_proc>           : int : The number of processes used in model training\n";
        std::cout << " <thres_value>        : int : threshold value\n";
        std::cout << " <train_batch_size>   : int : batch size for training\n";
        std::cout << " <gpu_index>          : int : The index of gpu used for evaluating, default is 0\n";
        std::cout << " <eval_batch_size>    : int : batch size for evaluation\n";
        std::cout << "------------------------------------\n";

        return 0;
    }

    /////////////////////////////////////////////////////////////////////
    // CONFIGURE MODEL PARAMS
    /////////////////////////////////////////////////////////////////////
    Config config;
    std::vector<std::pair<std::string,std::string>> data_for_PE = scatter_data_for_eval();

   
    /////////////////////////////////////////////////////////////////////
    // INITIALIZE DATASET AND LOADER FOR TRAINING
    /////////////////////////////////////////////////////////////////////

    // Custom dataset for foreground: 200 samples/sequence from FgSegNet
    int batch_size = (int) std::atoi(argv[5]);;
    auto dataset = CustomDataset(
        0,                           // proc_id
        1,                           // num_proc   
        batch_size,                  // batch_sz
        config.FG_TRAINING_FRAME,    // data_dir
        data_for_PE                  // data_sequences
    ).map(torch::data::transforms::Stack<>());

    // Data sampler to load batch of samples    
    auto data_loader_options =  torch::data::DataLoaderOptions(batch_size).workers(1).max_jobs(1);

    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        dataset,
        data_loader_options
        // batch_size
    );


    /////////////////////////////////////////////////////////////////////
    // SPECIFY DEVICE FOR MODEL TRAINING
    /////////////////////////////////////////////////////////////////////
    int gpu_idx         = (int) std::atoi(argv[4]);

    torch::Device device = torch::Device(
        torch::cuda::is_available() ?              // Use GPU for training
            /*torch::DeviceType=*/  torch::kCUDA :
            /*torch::DeviceType=*/  torch::kCPU, 
        /*int device_idx=*/ gpu_idx
    );


    /////////////////////////////////////////////////////////////////////  
    // INITIALIZE MODEL OF FDNet
    /////////////////////////////////////////////////////////////////////
    
    // Initialize FDNet model
    FDNet fdn_net;
    fdn_net->to(device);

    // Check and load pre-trained model (if available)
    std::string file_weight_name = std::string(argv[1]) + "proc_" 
                                    + "batch" + std::string(argv[3])
                                    + "_thres" + std::string(argv[2]) 
                                    + ".pt";
    std::string path_to_weight = std::string("pretrained/") + "batch" + std::string(argv[3]) + "/" + file_weight_name;
    std::cout << "Search the pretrain weight: " << path_to_weight << "\n";

    // if pre-trained path exists
    struct stat st = {0};
    if (stat("pretrained", &st) != -1){  

        // if pre-trained weight exists
        if (stat(path_to_weight.c_str(), &st) != -1){     
            // Load the model
            torch::load(fdn_net, path_to_weight);

            // Warning a message
            std::cout << "FDNet is loaded with pretrained weight from " << path_to_weight << "\n";
        }
    }


    /////////////////////////////////////////////////////////////////////
    // CONFIGS FOR MODEL EVALUATING
    /////////////////////////////////////////////////////////////////////
    
    // A counter for the number of training steps
    int pass_num = 0;

    // Set training metrics: Loss and Accuracy 
    float loss_val  = 0.0;
    float acc_val   = 0.0;


    /////////////////////////////////////////////////////////////////////
    // START EVALUATING LOOP
    /////////////////////////////////////////////////////////////////////

    // Toggle evaluation mode on trained model of FDNet
    std::cout << "Starting model evaluating on " << data_for_PE.size() << " sequences ..." << "\n";
    fdn_net->eval();

    // Loop over multiple sequences
    for(int sequence_idx=0;sequence_idx<data_for_PE.size();sequence_idx++){
        
        // Start data loop via batches
        for (auto& batch : *data_loader) {

            // count the training batch
            pass_num++;

            auto data = batch.data.to(torch::kFloat32).to(device) / 255.0;
            auto target = batch.target.to(torch::kFloat32).to(device) / 255.0;

            fdn_net->zero_grad();

            // Feed forward
            torch::Tensor pred_output = fdn_net->forward(data);

            // Calculate loss
            torch::Tensor loss = model_loss(pred_output, target);

            // Estimate accuracy
            torch::Tensor acc = model_accuracy(pred_output, target);

            // Calculate loss & acc
            loss_val += loss.item<float>();
            acc_val += acc.item<float>();
            
            if( (pass_num % (batch_size*2)  == 0) )
                std::cout << "step = "          << (pass_num) << "\t"
                          << std::fixed         << std::setprecision(7) << "\t"
                          << "Loss = "          << loss_val/pass_num << "\t"
                          << "Acc = "           << acc_val/pass_num << "\n";

        }   // End data loop via batches
    
    } // End loop of sequences          


    /////////////////////////////////////////////////////////////////////
    // EXPORT COLLECTED INFO
    /////////////////////////////////////////////////////////////////////

    // Open file stream to record training log of loss and accuracy
    std::ofstream logging_file;
    std::string file_log_name = "evaluation_benchmark.csv";

    logging_file.open (file_log_name, std::ofstream::app);
    logging_file << std::string(argv[3]) << ","         // training batch_size
                 << std::string(argv[1]) << ","         // training num_proc
                 << std::string(argv[2]) << ","         // training threshold value
                 << acc_val/(1.0*pass_num)  << ","      // evaluating accuracy
                 << loss_val/(1.0*pass_num) << "\n";    // evaluating loss value
    logging_file.close();
    
    return 0;
}

std::vector<std::pair<std::string,std::string>> scatter_data_for_eval(){
    
    // Split data for current PE
    std::vector<std::pair<std::string,std::string>> data_for_pe;
    
    for(auto data_pair : cdnet_data){
        std::string data_scenario = data_pair.first;
        std::vector<std::string> data_sequences = data_pair.second;

        for(auto data_sequence : data_sequences){            
            data_for_pe.push_back(
                std::make_pair(data_scenario, data_sequence)
            );                
        }
    }

    return data_for_pe;
}