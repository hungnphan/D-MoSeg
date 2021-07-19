//
// @file    : main.cpp
// @purpose : An implementation class for FDN training with EventGrad paradigm using UPC++ and LibTorch
// @author  : Hung Ngoc Phan
// @project : Distributed training of Foreground Detection Network with Event-Triggered Data Parallelism
// @licensed: N/A
// @created : 10/05/2021
// @modified: 03/07/2021
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <upcxx/upcxx.hpp>

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

// Limit the max number of processes by GPU memory
// std::map<int,int> nproc_per_gpu_by_batch_size {
//     { 2, 14 },
//     { 4, 10 },
//     { 8, 6 },
//     { 12, 4 },
// };

std::pair<std::string,std::string> query_data(int proc_id);
std::vector<std::pair<std::string,std::string>> distribute_data_for_pe(int proc_id, int num_proc);


////////////////////////////////////////////////////////////////////////////////////////////////
// ------- ARGUMENT DESCRIPTION -------
// ------------------------------------
// <thres_type>     : int   describes the type of threshold (0 for const, 1 for adaptive)
// <thres_value>    : int   is threshold value
// <batch_size>     : int   is batch size for training
// <num_epoch>      : int   is the number of epoch to loop through the dataset
// <logging>        : int   is flag for log printing (0 for hidden, 1 for display), default is 1
// <log_freq>       : int   is the frequency of logging printing, default 5
// <use_gpu>        : int   is use the GPU for training or not, default 0
////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]){

    /////////////////////////////////////////////////////////////////////
    // ARGUMENT ASSESSMENT
    /////////////////////////////////////////////////////////////////////
    if(argc < 5){
        std::cout << "Please input all necessary arguments:\n";
        std::cout << "------------------------------------\n";
        std::cout << "./run <thres_type> <thres_value> <batch_size> <num_epoch>\n";
        std::cout << "------------------------------------\n";
        std::cout << " <thres_type> : int : the type of threshold (0 for const, 1 for adaptive)\n";
        std::cout << " <thres_value>: int : threshold value\n";
        std::cout << " <batch_size> : int : batch size for training\n";
        std::cout << " <num_epoch>  : int : the number of epoch to loop through the dataset\n";
        std::cout << "------------------------------------\n";

        return 0;
    }


    /////////////////////////////////////////////////////////////////////
    // SET UP UPCXX RUN-TIME
    /////////////////////////////////////////////////////////////////////
    upcxx::init();
    int rank_me     = upcxx::rank_me();    // 0
    int rank_n      = upcxx::rank_n();     // 3
    int rank_left   = (rank_me-1+rank_n) % rank_n;
    int rank_right  = (rank_me+1) % rank_n;

    /////////////////////////////////////////////////////////////////////
    // CONFIGURE MODEL PARAMS
    /////////////////////////////////////////////////////////////////////
    Config config;
    std::vector<std::pair<std::string,std::string>> data_for_PE = distribute_data_for_pe(rank_me, rank_n);
    // std::pair<std::string,std::string> data_of_me = query_data(rank_me);
    // config.scenario_name = data_of_me.first;
    // config.sequence_name = data_of_me.second;


    for(int p=0;p<rank_n;p++){
        if(rank_me == p){
            std::cout << "Process " << p << ": " << data_for_PE.size() << "\n";
            // for(auto data : data_for_PE){
            //     std::cout << "\t"
            //               << data.first << ", "
            //               << data.second << std::endl;
            // }
        }

        upcxx::barrier();
    }


    /////////////////////////////////////////////////////////////////////
    // INITIALIZE DATASET AND LOADER FOR TRAINING
    /////////////////////////////////////////////////////////////////////

    // Custom dataset for foreground: 200 samples/sequence from FgSegNet
    int batch_size = (int) std::atoi(argv[3]);
    auto dataset = CustomDataset(
        0,                           // proc_id
        1,                           // num_proc   
        batch_size,                  // batch_sz
        config.FG_TRAINING_FRAME,    // data_dir
        // config.scenario_name,        // scenario_name
        // config.sequence_name         // sequence_name
        data_for_PE                  // data_sequences
    ).map(torch::data::transforms::Stack<>());

    // Data sampler to load batch of samples    
    auto data_loader_options =  torch::data::DataLoaderOptions(batch_size).workers(1).max_jobs(1);

    // auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        dataset,
        data_loader_options
        // batch_size
    );

    // Toogle dataset size and iter/epoch:
    int num_train_samples_per_pe = dataset.size().value();
    int dataset_size = dataset.size().value();
    std::cout << "Proc_id #" << rank_me << ": "
            //   << "Scenario = " << config.scenario_name << ", Sequence = " << config.sequence_name
            //   << ", dataset_size = " << dataset_size 
              << " --- " << std::round(1.0*dataset_size*data_for_PE.size()/batch_size) << " steps/epoch"
              << std::endl;


    /////////////////////////////////////////////////////////////////////
    // SPECIFY DEVICE FOR MODEL TRAINING
    /////////////////////////////////////////////////////////////////////
    // int nproc_per_gpu   = nproc_per_gpu_by_batch_size[batch_size];
    // int gpu_idx         = (int) ( 1.0*rank_me / (1.0*nproc_per_gpu) );

    int n_gpus          = torch::cuda::device_count();
    int nproc_per_gpu   = (1.0*rank_n)/(1.0*n_gpus);
    if (rank_n % n_gpus != 0) nproc_per_gpu += 1;
    int gpu_idx         = (int) ( 1.0*rank_me / (1.0*nproc_per_gpu) );
    
    // Default is CPU
    torch::Device device(/*torch::DeviceType=*/     torch::kCPU);   // Use CPU for training
    
    // Use GPU if CUDA is available
    device = torch::Device(torch::cuda::is_available() ?            // Use GPU for training
                                /*torch::DeviceType=*/  torch::kCUDA :
                                /*torch::DeviceType=*/  torch::kCPU, 
                            /*int device_idx=*/ gpu_idx);


    /////////////////////////////////////////////////////////////////////  
    // INITIALIZE MODEL OF FDNet
    /////////////////////////////////////////////////////////////////////
    
    // Initialize FDNet model
    FDNet fdn_net;
    fdn_net->to(device);

    // Check and load pre-trained model (if available)
    std::string file_weight_name = std::to_string(rank_n) + "proc_" 
                                    + "batch" + std::to_string(batch_size) 
                                    + "_thres" + std::string(argv[2]) 
                                    + ".pt";
    std::string path_to_weight = std::string("pretrained/") + file_weight_name;

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
    // INITIALIZE OPTIMIZER
    /////////////////////////////////////////////////////////////////////
    torch::optim::Adam optimizer(
        fdn_net->parameters(), 
        torch::optim::AdamOptions(5e-3)
    );


    /////////////////////////////////////////////////////////////////////
    // CONTROL VARIABLES FOR EVENT-TRIGGERED COMMUNICATION
    /////////////////////////////////////////////////////////////////////

    // Collect #layers, #params, data-type size
    auto param = fdn_net->named_parameters();                   // model param <key,value>: torch::OrderedDict< std::string, torch::Tensor >
    size_t sz = param.size();                                   // number of weights' elements
    size_t param_elem_size = param[0].value().element_size();   // sizeof(dtype)

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


    /////////////////////////////////////////////////////////////////////
    // CONFIGS FOR MODEL TRAINING
    /////////////////////////////////////////////////////////////////////

    // Set training iterations and logging frequency
    int num_epochs          = (int) std::atoi(argv[4]);                      // number of epochs (#times loops over dataset)
    int pass_num            = 0;                                             // number of batch that was iterated
    int initial_comm_passes = 30;                                            // threshold to avoid stale value
    int num_events          = 0;                                             // count of #communications: send & receive
    int is_logging          = 1;                                             // flag to control logging: 0 for hidden, 1 for display
    int logging_frequency   = 10;                                            // how frequent/steps to print train-logging
    
    // Set training metrics: Loss and Accuracy 
    float loss_val  = 0.0;
    float acc_val   = 0.0;


    /////////////////////////////////////////////////////////////////////
    // START TRAINING LOOP
    /////////////////////////////////////////////////////////////////////

    // Wait for all processes to complete initialization
    upcxx::barrier();
    std::cout << "Starting model training ..." << "\n";

    // Start the timer
    clock_t time_start = clock();

    // Open file stream to record training log of loss and accuracy
    std::ofstream logging_file;
    std::string file_log_name = "loss-acc_" + std::to_string(rank_n) + "proc_" 
                                    + "batch" + std::to_string(batch_size) 
                                    + "_thres" + std::string(argv[2]) 
                                    + ".csv";
    logging_file.open ("log/"+file_log_name, std::ofstream::app);
    logging_file << "Epoch" << ","
                 << "Loss" << ","
                 << "Acc" << "\n";
    logging_file.close();

    // Start primary training loop
    for(int epoch=0;epoch<num_epochs;epoch++){        

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

                // Perform backpropagation
                loss.backward();

                // Init offset for param in win_mem_left and win_mem_right
                //              |<---------param[i]--------->|
                // |************||||||||||||||||||||||||||||||****************|
                // |<---disp--->|<------param[i].numel()---->|<---dont-care-->|
                int disp = 0;

                // Begin parameter loop
                for(int i = 0 ; i < sz ; i++){
                    
                    // upcxx::barrier();

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

                        // Send param[i] to left using win_mem_right
                        // auto future_of_left = win_mem_left.fetch(rank_right);
                        // upcxx::global_ptr<float> win_mem_neighbor_left = future_of_left.wait();
                        // std::cout << "pass_num #" << pass_num << ", i=" << i << " --- " << "Rank #" << rank_me << ": Start to fetch rank_right" << "\n";
                        upcxx::global_ptr<float> win_mem_neighbor_left = win_mem_left.fetch(rank_right).wait();
                        // std::cout << "pass_num #" << pass_num << ", i=" << i << " --- " << "Rank #" << rank_me << ": Finish fetch rank_right" << "\n";
                        upcxx::rput(temp, win_mem_neighbor_left + disp, flat.numel());

                        // Send param[i] to right using win_mem_left
                        // auto future_of_right = win_mem_right.fetch(rank_left);
                        // upcxx::global_ptr<float> win_mem_neighbor_right = future_of_right.wait();
                        // std::cout << "pass_num #" << pass_num << ", i=" << i << " --- " << "\tRank #" << rank_me << ": Start to fetch rank_left" << "\n";
                        upcxx::global_ptr<float> win_mem_neighbor_right = win_mem_right.fetch(rank_left).wait();
                        // std::cout << "pass_num #" << pass_num << ", i=" << i << " --- " << "\tRank #" << rank_me << ": Finish fetch rank_left" << "\n";
                        upcxx::rput(temp, win_mem_neighbor_right + disp, flat.numel());

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

                    // // Create tensor for left's param[i] using value got from win_mem_left->local()
                    // torch::Tensor left_tensor = torch::from_blob(
                    //     left_recv, 
                    //     dim_array, 
                    //     torch::TensorOptions().dtype(torch::kFloat)
                    // ).to(device).clone();

                    // Beta: Create tensor for left's param[i] using value got from win_mem_left->local()
                    torch::Tensor left_tensor = torch::zeros(
                        torch::ArrayRef<int64_t>(dim_array),
                        torch::TensorOptions().dtype(torch::kFloat)
                    );
                    std::memcpy(
                        left_tensor.data_ptr(), 
                        left_recv, 
                        sizeof(float)*left_tensor.numel()
                    );   


                    // Free left_recv pointer
                    free(left_recv);

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

                    // // Create tensor for left's param[i] using value got from win_mem_right->local()
                    // torch::Tensor right_tensor = torch::from_blob(
                    //     right_recv, 
                    //     dim_array, 
                    //     torch::TensorOptions().dtype(torch::kFloat)
                    // ).to(device).clone();

                    // Beta: Create tensor for right's param[i] using value got from win_mem_right->local()
                    torch::Tensor right_tensor = torch::zeros(
                        torch::ArrayRef<int64_t>(dim_array),
                        torch::TensorOptions().dtype(torch::kFloat)
                    );
                    std::memcpy(
                        right_tensor.data_ptr(), 
                        right_recv, 
                        sizeof(float)*right_tensor.numel()
                    ); 

                    // Free left_recv pointer
                    free(right_recv);

                    // Average value of param[i] at: rank_me, rank_left, rank_right
                    left_tensor = left_tensor.to(device);
                    right_tensor = right_tensor.to(device);                
                    param[i].value().data().add_(left_tensor.data());
                    param[i].value().data().add_(right_tensor.data());
                    param[i].value().data().div_(3);

                    // Update offset of param[i+1] in win_mem
                    disp = disp + flat.numel();

                    // Free calloc memory
                    free(temp);

                }   // End parameter loop

                // Update weight
                optimizer.step();

                // Calculate loss & acc
                loss_val += loss.item<float>();
                acc_val += acc.item<float>();
                
                if( (is_logging == 1) && ((pass_num) % logging_frequency == 0) )
                    std::cout << "step = "          << (pass_num) << "\t"
                            << "proc-id = "       << rank_me << "\t"
                            << "num_events = "    << num_events << "\t"
                            << std::fixed         << std::setprecision(7) << "\t"
                            << "Loss = "          << loss_val/pass_num << "\t"
                            << "Acc = "           << acc_val/pass_num << "\n";

            }   // End data loop via batches

            if (rank_me == 0){
                logging_file.open ("log/"+file_log_name, std::ofstream::app);
                logging_file << pass_num               << ","
                                << loss_val/pass_num   << ","
                                << acc_val/pass_num    << "\n";
                logging_file.close();
            }  
        }

              

    }   // End primary training loop


    /////////////////////////////////////////////////////////////////////
    // AVERAGING MODEL PARAMETERS AT RANK 0
    /////////////////////////////////////////////////////////////////////

    upcxx::barrier();

    for (int i = 0; i < sz; i++) {

        torch::Tensor param_i_cpu = param[i].value().cpu();

        // Get ready !!
        upcxx::barrier();

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

    /////////////////////////////////////////////////////////////////////
    // SAVE MODEL (AVERAGING PARAMS) TO FILE AT RANK 0
    /////////////////////////////////////////////////////////////////////
    if (rank_me == 0){
        torch::save(fdn_net, path_to_weight);
    }

    /////////////////////////////////////////////////////////////////////
    // COUNT THE NUMBER OF MESSAGES AND END TIMER
    /////////////////////////////////////////////////////////////////////
    
    // Get ready for calculating number of messages
    // Collect number of communicating message in work group    
    upcxx::barrier();
    num_events = upcxx::reduce_all(
        num_events, 
        upcxx::op_fast_add
    ).wait();    
    upcxx::barrier();

    // End timer
    clock_t time_end = clock();
    double elapsed_secs = double(time_end - time_start) / CLOCKS_PER_SEC;


    /////////////////////////////////////////////////////////////////////
    // EXPORT COLLECTED INFO
    /////////////////////////////////////////////////////////////////////
    if (rank_me == 0){
        std::cout << "Total number of events - " << num_events << std::endl;

        std::ofstream result_file;
        std::string result_file_name = "log/log_num_messages_" + std::to_string(rank_n) + "proc" + ".csv";
        result_file.open (result_file_name, std::ofstream::app);
        result_file << batch_size << "," 
                    << rank_n << "," 
                    << constant << ","
                    << num_events <<","
                    << elapsed_secs << "\n";
        result_file.close();
    }
    
    /////////////////////////////////////////////////////////////////////
    // TERMINATE UPCXX RUN-TIME
    /////////////////////////////////////////////////////////////////////
    upcxx::finalize();

    return 0;
}

std::pair<std::string,std::string> query_data(int proc_id){
    std::string scenario_name;
    std::string sequence_name;

    int accum_sum = 0;
    for(auto data_pair : cdnet_data){
        std::string data_scenario = data_pair.first;
        std::vector<std::string> data_sequences = data_pair.second;

        if(proc_id < (accum_sum+data_sequences.size())){
            int data_idx = proc_id - accum_sum;

            scenario_name = data_scenario;
            sequence_name = data_sequences[data_idx];

            break;
        }
        else accum_sum += data_sequences.size();
    }

    return std::make_pair(scenario_name,sequence_name);
}

std::vector<std::pair<std::string,std::string>> distribute_data_for_pe(int proc_id, int num_proc){
    
    // Estimate the number of sequences for current PE
    int num_sequence_for_pe = (1.0*kNumSequence) / (1.0*num_proc);
    if(proc_id < (kNumSequence % num_proc)) num_sequence_for_pe++;

    // Estimate the offset of sequences for current PE
    int offset_sequence = (1.0*kNumSequence) / (1.0*num_proc);
    offset_sequence *= proc_id;
    offset_sequence += std::min(proc_id, (kNumSequence % num_proc));

    // Split data for current PE
    std::vector<std::pair<std::string,std::string>> data_for_pe;
    int accum_sum = 0;
    bool isDone = false;

    for(auto data_pair : cdnet_data){
        std::string data_scenario = data_pair.first;
        std::vector<std::string> data_sequences = data_pair.second;

        for(auto data_sequence : data_sequences){
            if((accum_sum+1) <= offset_sequence) accum_sum++;
            else{
                if(data_for_pe.size() < num_sequence_for_pe){
                    data_for_pe.push_back(
                        std::make_pair(data_scenario, data_sequence)
                    );
                }else{
                    isDone = true;
                    break;
                }
            }
        }
        if(isDone == true) break;
    }

    return data_for_pe;
}