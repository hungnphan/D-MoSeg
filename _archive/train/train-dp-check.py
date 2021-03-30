import numpy as np
import cupy as cp
import cv2 as cv
import torch
import os
from arg_parser import parse_config_from_json
from cdn import CDN
from likelihood_loss_function import TrainingCriterion
from data_io import DataLoader

def model_summary(model):
    print("Layer_name"+"\t"*7+"Number of Parameters")
    print("="*100)
    model_parameters = [layer for layer in model.parameters() if layer.requires_grad]
    layer_name = [child for child in model.children()]
    j = 0
    total_params = 0
    print("\t"*10)
    for i in layer_name:
        print()
        param = 0
        try:
            bias = (i.bias is not None)
        except:
            bias = False  
        if not bias:
            param =model_parameters[j].numel()+model_parameters[j+1].numel()
            j = j+2
        else:
            param =model_parameters[j].numel()
            j = j+1
        print(str(i)+"\t"*3+str(param))
        total_params+=param
    print("="*100)
    print(f"Total Params:{total_params}")       

if __name__ == '__main__':
    # Specify device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Read method params from json config file
    config = parse_config_from_json(config_file='config.json')

    # Initialize data_loader
    data_loader = DataLoader(config)

    # Define Convolutional Density Network
    cdn_net = CDN(config.KMIXTURE, \
                  data_loader.img_heigh, data_loader.img_width, data_loader.img_channel)

    if config.USE_MULTI_GPUS == True and torch.cuda.device_count() >= 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        cdn_net = torch.nn.DataParallel(cdn_net)
    cdn_net = cdn_net.to(device)

    # Define loss function
    train_criterion = TrainingCriterion(config)

    # Define network optimizer
    optimizer = torch.optim.Adam(cdn_net.parameters(), lr=config.LEARNING_RATE)

    # Calculate the number of pixel in 2D image space
    N_PIXEL = data_loader.img_heigh * data_loader.img_width

    # Print model summary
    # model_summary(cdn_net) 


    # input shape = [num_pixels, C, 1, FPS]
    num_pixels, C, FPS = config.PIXEL_BATCH, 3, config.FPS
    example_input = np.random.randint(0, 256,[num_pixels,C,1,FPS])
    example_input = (torch.from_numpy(example_input).float() / 255.0).cuda()
    print(f"example_input.shape = {example_input.shape}")

    example_output = cdn_net(example_input)
    print(f"example_output.shape = {example_output.shape}")


