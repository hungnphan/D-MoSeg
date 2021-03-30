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

def train(config, camera="camera"):

    # Calculate the number of pixel in 2D image space
    N_PIXEL = data_loader.img_heigh * data_loader.img_width

    print("\nStart the model training ...")
    print(f"There are {round(N_PIXEL/config.PIXEL_BATCH)} batches for training\n")

    # Train on each of sliding window's move
    for sliding_step in range(config.SLIDING_STEP):
        print(f"---Training with the position #{sliding_step} of sliding windows---")

        # Train on a position of sliding window [config.EPOCHS] times
        for epoch in range(config.EPOCHS):
            epoch_loss = 0.0
            step_loss = 0.0

            # Train on batches with a size of [config.PIXEL_BATCH] pixels
            for train_step in range(round(N_PIXEL/config.PIXEL_BATCH)):
                # zero the parameter gradients
                optimizer.zero_grad()

                # Split the training batch
                batch_start = train_step * config.PIXEL_BATCH                   
                batch_end = min(N_PIXEL, (train_step+1) * config.PIXEL_BATCH)                
                training_batch = data_loader.data_frame[batch_start:batch_end, ...]     # input shape = [num_pixels, C, 1, FPS]
               
                # Feed forward on CDN
                output = cdn_net(training_batch)    # output shape = [N, 5 * KMIX]

                # x.shape = [N, 1, FPS, 3]  y.shape = [N, K_MIXTURES*5]
                loss = train_criterion.likelihood_loss(training_batch.permute(0, 2, 3, 1), output).to(device)
                
                # Backward + optimize
                loss.backward()
                optimizer.step()

                # Accumulated loss values
                step_loss += loss.item()        # Accumulated loss for each train_step (train on batches)
                epoch_loss += loss.item()       # Accumulated loss for each
                
                torch.save({
                    'sliding_step': sliding_step,
                    'epoch': epoch,
                    'model_state_dict': cdn_net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss}, 
                    os.path.join(config.TRAINING_DIR, camera)
                )
            
                # Report the average loss every 200 mini-batches
                if (train_step+1) % 200 == 0:
                    print('[epoch=%d, train_step=%5d]\tloss: %.3f' %
                        (epoch + 1, train_step + 1, step_loss / 200,))   
                    step_loss = 0.0         
            
            # Report the average loss at each position of sliding window
            print('---> Everage loss (at each position of sliding window) = %.5f\n' %(epoch_loss / round(N_PIXEL/config.PIXEL_BATCH)))

        data_loader.load_next_k_frame(2)


if __name__ == '__main__':
    # Specify device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Read method params from json config file
    config = parse_config_from_json(config_file='config.json')

    # Initialize data_loader
    data_loader = DataLoader(config)

    # Define Convolutional Density Network
    cdn_net = CDN(config.KMIXTURE, \
                  data_loader.img_heigh, data_loader.img_width, data_loader.img_channel).to(device)

    # Define loss function
    train_criterion = TrainingCriterion(config)

    # Define network optimizer
    optimizer = torch.optim.Adam(cdn_net.parameters(), lr=config.LEARNING_RATE)

    # Print model summary
    model_summary(cdn_net) 

    # Restore model from last checkpoint
    camera = "camera"
    checkpoint = torch.load(os.path.join(config.TRAINING_DIR, camera))
    cdn_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    sliding_step = checkpoint['sliding_step']
    epoch = checkpoint['epoch']

    # Set model mode to continue training
    cdn_net.train() # cdn_net.eval()

    # Perform training on CDN model
    train(config=config, camera=camera)


    # model = TheModelClass(*args, **kwargs)
    # optimizer = TheOptimizerClass(*args, **kwargs)

    # checkpoint = torch.load(PATH)

    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']

    # loss = checkpoint['loss']

    # model.eval()
    # # - or -
    # model.train()





    # Initialize data_loader for output result for demo
    data_loader_2 = DataLoader(config)

    for frame_idx in range (3500):
        print("Frame #%4d" % frame_idx)

        # Calculate background image
        bg_img = cdn_net.calculate_background(data_loader_2.data_frame, batch_size = 16)

        input_img = (data_loader_2.data_frame[..., (data_loader_2.current_frame_idx % data_loader_2.FPS)] * 255.0).type(torch.uint8).cpu().data.numpy()
        input_img = input_img.reshape([data_loader_2.img_heigh, data_loader_2.img_width, data_loader_2.img_channel])

        # Display background image
        cv.imshow('Input image',input_img)
        cv.imshow('Background image',bg_img)
        cv.waitKey(0)


        # Shift the sliding windows for the next iteration
        data_loader_2.load_next_k_frame(1)

    cv.destroyAllWindows()



