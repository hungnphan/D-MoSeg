//
// @file    : main.cpp
// @purpose : An implementation class for FDN training with EventGrad paradigm using UPC++ and LibTorch
// @author  : Hung Ngoc Phan
// @project : Distributed training of Foreground Detection Network with Event-Triggered Data Parallelism
// @licensed: N/A
// @created : 10/05/2021
// @modified: 17/05/2021
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <upcxx/upcxx.hpp>

#include <iostream>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include <sstream>
#include <iomanip>
#include <tuple>
#include <stdint.h>

#include "custom_dataset.h"     // data loader for custom dataset
#include "util.h"               // data conversion: torch::Tensor and cv::Mat
#include "model.h"              // implementation of FDN on TorchLib
#include "config.h"             // configuration of data and training

// Dictionary of dataset CDnet2014
std::map <std::string, std::vector<std::string> > cdnet_data;
cdnet_data["badWeather"]               = std::vector<std::string>{"blizzard","skating","snowFall","wetSnow"};
cdnet_data["baseline"]                 = std::vector<std::string>{"highway","office","pedestrians","PETS2006"};
cdnet_data["cameraJitter"]             = std::vector<std::string>{"badminton","boulevard","sidewalk","traffic"};
cdnet_data["dynamicBackground"]        = std::vector<std::string>{"boats","canoe","fall","fountain01","fountain02","overpass"};
cdnet_data["intermittentObjectMotion"] = std::vector<std::string>{"abandonedBox","parking","sofa","tramstop","winterDriveway"};
cdnet_data["lowFramerate"]             = std::vector<std::string>{"port_0_17fps","tramCrossroad_1fps","tunnelExit_0_35fps","turnpike_0_5fps"};
cdnet_data["nightVideos"]              = std::vector<std::string>{"bridgeEntry","busyBoulvard","fluidHighway","streetCornerAtNight","tramStation","winterStreet"};
cdnet_data["PTZ"]                      = std::vector<std::string>{"continuousPan","intermittentPan","twoPositionPTZCam","zoomInZoomOut"};
cdnet_data["shadow"]                   = std::vector<std::string>{"backdoor","bungalows","busStation","copyMachine","cubicle","peopleInShade"};
cdnet_data["thermal"]                  = std::vector<std::string>{"corridor","diningRoom","lakeSide","library","park"};
cdnet_data["turbulence"]               = std::vector<std::string>{"turbulence0","turbulence1","turbulence2","turbulence3"};

std::vector<std::string>badWeather{"blizzard","skating","snowFall","wetSnow"};

int main(int argc, char* argv[]){

    // Setup UPC++ runtime
    upcxx::init();
    int rank_me    = upcxx::rank_me();    // 0
    int rank_n     = upcxx::rank_n();     // 3
    int rank_left = (rank_me-1+rank_n) % rank_n;
    int rank_right = (rank_me+1) % rank_n;

    // Create config for the model
    Config config;
    config.scenario_name = "badWeather";
    config.sequence_name = badWeather[rank_me];

    // Initialize custom dataset for foreground training
    auto dataset = CustomDataset(
        0,                           // proc_id
        1,                           // num_proc   
        config.FG_TRAINING_FRAME,    // data_dir
        config.scenario_name,        // scenario_name
        config.sequence_name         // sequence_name
    ).map(torch::data::transforms::Stack<>());

    // Generate a data loader with fixed batch size
    int batch_size = 1;
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        dataset,
        batch_size
    );

    // Toogle dataset size and iter/epoch:
    int num_train_samples_per_pe = dataset.size().value();
    int dataset_size = dataset.size().value();
    std::cout << "Proc_id #" << rank_me 
              << ": dataset_size = " << dataset_size 
              << " --- " << std::round(1.0*dataset_size/batch_size) << " steps/epoch"
              << std::endl;

    // Set device CPU or GPU (if available)
    torch::Device device(torch::cuda::is_available() ? 
                            /*torch::DeviceType=*/ torch::kCUDA :
                            /*torch::DeviceType=*/ torch::kCPU, 
                         /*int device_idx=*/ 0);

    // Init model and put model to device
    Model fdn_net;
    fdn_net.to(device);

    // Init optimizer
    torch::optim::Adam optimizer(
        fdn_net.parameters(), 
        torch::optim::AdamOptions(5e-3)
    );


    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // Control variables for event-triggered communication
    //////////////////////////////////////////////////////////////////////////////////////////////////////

    // Collect #layers, #params, data-type size
    auto param = fdn_net.named_parameters();                  // model param <key,value>: torch::OrderedDict< std::string, torch::Tensor >
    size_t sz = param.size();                                 // number of weights' elements
    size_t param_elem_size = param[0].value().element_size(); // sizeof(dtype)

    // count #elements in model
    int num_elem_param = 0;                     
    for(int layer_idx=0;layer_idx<sz;layer_idx++)
        num_elem_param += param[layer_idx].value().numel();
    
    // Init memory buffer to simulate send-get with two neighbors
    upcxx::dist_object<upcxx::global_ptr<float>> win_mem_left  = upcxx::new_array<float>(num_elem_param);
    upcxx::dist_object<upcxx::global_ptr<float>> win_mem_right = upcxx::new_array<float>(num_elem_param);

    // Set Threshold for communication: 0 for non-adaptive, 1 for adaptive
    int thres_type = (int) std::atoi(argv[1]); 
    float horizon, constant;
    if (thres_type == 1)  horizon = (float)std::atof(argv[2]);      // adaptive threshold
    else                  constant = (float)std::atof(argv[2]);     // fixed constant threshold

    // Set history size
    int sent_history = 2;

    // Threshold
    float thres[sz]                              = {0.0};

    // variables at the sender
    float last_sent_values_norm[sz]              = {0.0};
    float last_sent_iters[sz]                    = {0.0};
    float sent_slopes_norm[sz][sent_history]     = {0.0};

    // variables at the receiver
    float left_last_recv_values[num_elem_param]  = {0.0};
    float left_last_recv_values_norm[sz]         = {0.0};
    float left_last_recv_iters[sz]               = {0.0};

    float right_last_recv_values[num_elem_param] = {0.0};
    float right_last_recv_values_norm[sz]        = {0.0};
    float right_last_recv_iters[sz]              = {0.0};

    float left_recv_norm[sz]                     = {0.0};
    float right_recv_norm[sz]                    = {0.0};


    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // Model training
    //////////////////////////////////////////////////////////////////////////////////////////////////////

    // Set training iterations and logging frequency
    int num_epochs          = 5;        // number of epochs (#times loops over dataset)
    int pass_num            = 0;        // number of batch that was iterated
    int initial_comm_passes = 30;       // threshold to avoid stale value
    int num_events          = 0;        // count of #communications: send & receive
    int logging_frequency   = 5;        // how frequent to print train-logging
    
    // Set training metrics: Loss and Accuracy 
    float loss_val  = 0.0;
    float acc_val   = 0.0;

    std::cout << "Starting model training ..." << "\n";

    // Start primary training loop
    for(int epoch=0;epoch<num_epochs;epoch++){

        // Start data loop via batches
        for (auto& batch : *data_loader) {

            // count the training batch
            pass_num++;

            auto data = batch.data.to(torch::kFloat32).to(device) / 255.0;
            auto target = batch.target.to(torch::kFloat32).to(device) / 255.0;

            fdn_net.zero_grad();

            // Feed forward
            torch::Tensor pred_output = fdn_net.forward(data);

            // Calculate loss
            torch::Tensor loss = model_loss(pred_output, target);

            // Estimate accuracy
            torch::Tensor acc = model_accuracy(pred_output, target);

            // Perform backpropagation
            loss.backward();

            // Init offset for param in win_mem_left and win_mem_right
            //              |<---------param[i]--------->|
            // |************||||||||||||||||||||||||||||||****************|
            // |<---disp--->|<------param[i].numel()---->|<---dont-care-->|
            int disp = 0;

            // Begin parameter loop
            for(int i = 0 ; i < sz ; i++){

                // Get dimensions of tensor
                std::vector<int64_t> dim_array;
                for (int j = 0; j < param[i].value().dim(); j++) 
                    dim_array.push_back(param[i].value().size(j));
                
                // Flatten the tensor and copying it to a 1-D vector
                torch::Tensor flat = torch::flatten(param[i].value());
                float * temp = (float *) calloc(flat.numel(), flat.numel() * param_elem_size);
                for (int j = 0; j < flat.numel(); j++)
                    *(temp + j) = flat[j].item<float>();
                
                // Consider norm of current parameter
                float curr_norm = torch::norm(flat).item<float>();
                float value_diff = std::fabs(curr_norm - last_sent_values_norm[i]);
                int iter_diff = pass_num - last_sent_iters[i];

                // Set threshold for param[i]
                if (thres_type == 1) thres[i] = thres[i] * horizon;
                else                 thres[i] = constant;

                /////////////////////////////////////////////////////////////////////
                // Communication part of event-triggered policy
                /////////////////////////////////////////////////////////////////////

                // Sending part:
                //     |--- deliver param[i] to neighbor left and right
                if (value_diff >= thres[i] || pass_num < initial_comm_passes) {

                    // Count messages sent to left and right neighbors
                    num_events += 2;
                    upcxx::global_ptr<float> win_mem_neighbor;

                    // Send param[i] to left using win_mem_right
                    win_mem_neighbor = win_mem_left.fetch(rank_right).wait();
                    upcxx::rput(temp, win_mem_neighbor + disp, flat.numel());

                    // Send param[i] to right using win_mem_left
                    win_mem_neighbor = win_mem_right.fetch(rank_left).wait();
                    upcxx::rput(temp, win_mem_neighbor + disp, flat.numel());

                    // Shift the slope slope[i][j] = slope[i][j+1]
                    // and to assign the new slope value to slope[i][sent_history-1]: 
                    float slope_avg = 0.0;
                    int j = 0;
                    for (j = 0; j < sent_history - 1; j++) {
                        sent_slopes_norm[i][j] = sent_slopes_norm[i][j + 1];
                        slope_avg += sent_slopes_norm[i][j];
                    }

                    // Calculating new slope value
                    sent_slopes_norm[i][j] = value_diff / iter_diff;
                    slope_avg += sent_slopes_norm[i][j];
                    slope_avg = slope_avg / sent_history;

                    // Calculating new threshold if adaptive
                    if (thres_type == 1)  thres[i] = slope_avg;

                    // update last communicated parameters
                    last_sent_values_norm[i] = curr_norm;
                    last_sent_iters[i] = pass_num;
                    
                } 

                // Receiving part:
                //     |--- retrieve tensor value from win_mem_left and win_mem_right
                float* win_mem_local;

                // ---- Part 01: Get value from left neighbor ----
                float* left_recv = (float*) calloc(flat.numel(), flat.numel() * param_elem_size);
                float left_temp = 0.0;

                win_mem_local = win_mem_left->local();
                for (int j = 0; j < flat.numel(); j++) {
                    *(left_recv + j) = *(win_mem_local + disp + j);
                    left_temp += std::pow(*(left_recv + j), 2);
                }

                left_temp = std::sqrt(left_temp / flat.numel());
                left_recv_norm[i] = left_temp;
                float left_recv_diff = std::fabs(left_recv_norm[i] - left_last_recv_values_norm[i]);

                // Update last_rcv_value_norm and last_rcv_iters
                if (left_recv_diff > 0) {
                    left_last_recv_values_norm[i] = left_recv_norm[i];
                    left_last_recv_iters[i] = pass_num;
                }

                // Create tensor for left's param[i] using value got from win_mem_left->local()
                torch::Tensor left_tensor = torch::from_blob(
                    left_recv, 
                    dim_array, 
                    torch::TensorOptions().dtype(torch::kFloat)
                ).to(device).clone();

                // ---- Part 02: Get value from right neighbor ----
                float* right_recv = (float*) calloc(flat.numel(), flat.numel() * param_elem_size);
                float right_temp = 0.0;

                win_mem_local = win_mem_right->local();
                for (int j = 0; j < flat.numel(); j++) {
                    *(right_recv + j) = *(win_mem_local + disp + j);
                    right_temp += std::pow(*(right_recv + j), 2);
                }

                right_temp = std::sqrt(right_temp / flat.numel());
                right_recv_norm[i] = right_temp;
                float right_recv_diff = std::fabs(right_recv_norm[i] - right_last_recv_values_norm[i]);

                // Update last_rcv_value_norm and last_rcv_iters
                if (right_recv_diff > 0) {
                    right_last_recv_values_norm[i] = right_recv_norm[i];
                    right_last_recv_iters[i] = pass_num;
                }

                // Create tensor for left's param[i] using value got from win_mem_right->local()
                torch::Tensor right_tensor = torch::from_blob(
                    right_recv, 
                    dim_array, 
                    torch::TensorOptions().dtype(torch::kFloat)
                ).to(device).clone();

                // Average value of param[i] at: rank_me, rank_left, rank_right
                param[i].value().data().add_(left_tensor.data());
                param[i].value().data().add_(right_tensor.data());
                param[i].value().data().div_(3);

                // Update offset of param[i+1] in win_mem
                disp = disp + flat.numel();

                // Free calloc memory
                free(temp);
                free(left_recv);
                free(right_recv);

            }   // End parameter loop

            // Update weight
            optimizer.step();

            // Calculate loss & acc
            loss_val += loss.item<float>();
            acc_val += acc.item<float>();
            
            if((pass_num) % logging_frequency == 0)
                std::cout << "step = " << (pass_num) << "\t"
                          << "proc-id = " << rank_me << "\t"
                          << "Loss = " << loss_val/pass_num << "\t"
                          << "Acc = " << acc_val/pass_num << "\t"
                          << "num_events = " << num_events << std::endl;

        }   // End data loop via batches        
    }   // End primary training loop

    // Get ready for param synchronization after model training
    upcxx::barrier();

    // Averaging learned params at rank 0
    for (int i = 0; i < sz; i++) {

        torch::Tensor param_i_cpu = param[i].value().cpu();

        upcxx::reduce_all(
            (float*) param_i_cpu.data_ptr(), 
            (float*) param_i_cpu.data_ptr(), 
            param[i].value().numel(), 
            upcxx::op_fast_add
        ).wait();

        if (rank_me == 0){
            param_i_cpu = param_i_cpu.to(device);
            param[i].value().data() = param_i_cpu.data() / rank_n;
        }
    }

    // Collect number of communicating message in work group    
    num_events = upcxx::reduce_all(
        num_events, 
        upcxx::op_fast_add
    ).wait();
    
    if (rank_me == 0) std::cout << "Total number of events - " << num_events << std::endl;
    
    // Close down UPC++ runtime
    upcxx::finalize();

    return 0;
}



