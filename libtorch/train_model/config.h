//
// @file    : config.h
// @purpose : A header file for training and model configuration
// @author  : Hung Ngoc Phan
// @project : Distributed training of Foreground Detection Network with Event-Triggered Data Parallelism
// @licensed: N/A
// @created : 10/05/2021
// @modified: 15/05/2021
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <string>

///////////////////////////////////////////////////////////////////////
// Training configuration
///////////////////////////////////////////////////////////////////////
struct Config {
    bool            use_multi_gpus                  = true;
    std::string     sequence_dir                    = "data/cdnet";
    std::string     foreground_training_data_dir    = "data/fg200";
    std::string     scenario_name                   = "badWeather";
    std::string     sequence_name                   = "skating";
    
    int             kFPS							= 240;
    int             kMixture						= 4;
    int             kPixelBatch						= 128;
    
    float           kLearningRate					= 1e-4;
    int             kEpochs							= 1;
    int             kSlidingStep					= 2;
    
    std::string		CKPT_dir						= "training_ckpt";
    std::string		FG_TRAINING_DATA				= "data/fg_train_data";    
    std::string		FG_TRAINING_FRAME				= "data/fg_train_frame";    
};

#endif