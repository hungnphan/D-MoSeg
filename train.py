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
    print("model_summary")
    print()
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

    # Define Convolutional Density Network
    cdn_net = CDN(config.KMIXTURE).to(device)

    # Define loss function
    train_criterion = TrainingCriterion(config)

    # Initialize data_loader
    data_loader = DataLoader(config)

    # Define network optimizer
    optimizer = torch.optim.Adam(cdn_net.parameters(), lr=config.LEARNING_RATE)

    # print(data_loader.data_frame.shape)

    N_PIXEL = data_loader.img_heigh * data_loader.img_width

    model_summary(cdn_net) 

    # loop over the dataset multiple times
    for sliding_step in range(config.SLIDING_STEP):
        train_data_frame = data_loader.data_frame

        for epoch in range(config.EPOCHS):
            epoch_loss = 0.0
            step_loss = 0.0

            # print(f"there are {N_PIXEL/config.PIXEL_BATCH} batches")
            for train_step in range(round(N_PIXEL/config.PIXEL_BATCH)):
            # for train_step in range(5):
                # zero the parameter gradients
                optimizer.zero_grad()

                batch_start = train_step * config.PIXEL_BATCH                   
                batch_end = min(N_PIXEL, (train_step+1) * config.PIXEL_BATCH)

                # input shape = [N, C, 1, FPS]
                # output shape = [N, 5 * KMIX]
                output = cdn_net(data_loader.data_frame[batch_start:batch_end, ...]) 

                # x.shape = [N, 1, FPS, 3]  y.shape = [N, K_MIXTURES*5]
                loss = train_criterion.likelihood_loss(data_loader.data_frame[batch_start:batch_end, ...].permute(0, 2, 3, 1), output).to(device)
                
                # print(f"loss.item() = {loss.item()}")

                # forward + backward + optimize
                loss.backward()
                optimizer.step()

                # print statistics
                step_loss += loss.item()
                epoch_loss += loss.item()
                if train_step % 675 == 674:    # print every 500 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, train_step + 1, step_loss / 675,))
                    running_loss = 0.0
            print('---> Epoch loss = %.5f' %(epoch_loss / round(N_PIXEL/config.PIXEL_BATCH)))

        data_loader.load_next_k_frame(2)



