import numpy as np
import cupy as cp
import cv2 as cv
import torch
import os
from arg_parser import parse_config_from_json
from cdn import CDN
from likelihood_loss_function import TrainingCriterion
from data_io import DataLoader


class BackgroundTrainer:

    def __init__(self, config_file):
        
        # Specify device GPU if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Read method params from json config file
        self.config = parse_config_from_json(config_file=config_file)

        # Initialize data_loader
        self.data_loader = DataLoader(self.config)

        # Define Convolutional Density Network
        self.cdn_net = CDN(self.config.KMIXTURE, \
                           self.data_loader.img_heigh, self.data_loader.img_width, self.data_loader.img_channel).to(self.device)

        # Define loss function for background training
        self.train_criterion = TrainingCriterion(self.config)

        # Define network optimizer
        self.optimizer = torch.optim.Adam(self.cdn_net.parameters(), lr=self.config.LEARNING_RATE)

        # # Print model summary
        # model_summary(cdn_net) 
        
        # Restore model from last checkpoint
        camera_name = "camera"
        self.path_to_checkpoint = os.path.join(self.config.TRAINING_DIR, camera_name + '.pth')
        if os.path.exists(self.path_to_checkpoint):
            print(f"Loading checkpoint from {self.path_to_checkpoint} ..")
            checkpoint = torch.load(self.path_to_checkpoint)
            self.cdn_net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.last_sliding_step   = checkpoint['sliding_step']
            # self.last_epoch          = checkpoint['epoch']
        else:
            print('No checkpoint is found ! Start to train from scratch.')
            self.last_sliding_step   = 0
            # self.last_epoch          = 0

        # Set model mode to continue training
        self.cdn_net.train() # cdn_net.eval()

    def train(self, camera_name="camera"):
        # Calculate the number of pixel in 2D image space
        N_PIXEL = self.data_loader.img_heigh * self.data_loader.img_width

        print("\nStart the model training ...")
        print(f"There are {round(N_PIXEL/self.config.PIXEL_BATCH)} batches for training\n")

        # Sliding dataframe to the last training step
        for sliding_step in range(self.last_sliding_step+1):
            self.data_loader.load_next_k_frame(k=self.config.WINDOW_STRIDE)
        
        # Train on each of sliding window's move
        for sliding_step in range(self.last_sliding_step+1, self.config.SLIDING_STEP):
            print(f"---Training with the position #{sliding_step} of sliding windows---")

            # Train on a position of sliding window [config.EPOCHS] times
            for epoch in range(self.config.EPOCHS):
                epoch_loss = 0.0
                step_loss = 0.0

                # Train on batches with a size of [config.PIXEL_BATCH] pixels
                for train_step in range(round(N_PIXEL/self.config.PIXEL_BATCH)):
                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # Split the training batch
                    batch_start = train_step * self.config.PIXEL_BATCH                   
                    batch_end = min(N_PIXEL, (train_step+1) * self.config.PIXEL_BATCH)                
                    training_batch = self.data_loader.data_frame[batch_start:batch_end, ...]     # input shape = [num_pixels, C, 1, FPS]
                
                    # Feed forward on CDN
                    output = self.cdn_net(training_batch)    # output shape = [N, 5 * KMIX]

                    # x.shape = [N, 1, FPS, 3]  y.shape = [N, K_MIXTURES*5]
                    loss = self.train_criterion.likelihood_loss(training_batch.permute(0, 2, 3, 1), output).to(self.device)
                    
                    # Backward + optimize
                    loss.backward()
                    self.optimizer.step()

                    # Accumulated loss values
                    step_loss += loss.item()        # Accumulated loss for each train_step (train on batches)
                    epoch_loss += loss.item()       # Accumulated loss for each
                         
                    # Report the average loss every 200 mini-batches
                    if (train_step+1) % 200 == 0:
                        print('[sliding_step=%d, epoch=%d, train_step=%5d]\tloss: %.3f' %
                            (sliding_step, epoch, train_step, step_loss / 200,))   
                        step_loss = 0.0         
                
                # Report the average loss at each position of sliding window
                print('---> Everage loss (at each position of sliding window) = %.5f\n' %(epoch_loss / round(N_PIXEL/self.config.PIXEL_BATCH)))


            # Save training checkpoint: sliding_step, model_state_dict, optimizer_state_dict
            torch.save({
                'sliding_step': sliding_step,
                # 'epoch': epoch,
                'model_state_dict'      :   self.cdn_net.state_dict(),
                'optimizer_state_dict'  :   self.optimizer.state_dict()}, 
                self.path_to_checkpoint
            )

            self.data_loader.load_next_k_frame(k = self.config.WINDOW_STRIDE)

    def demo_background(self):
        # Initialize data_loader for output result for demo
        data_loader = DataLoader(self.config)

        for frame_idx in range (self.config.FPS, data_loader.data_len):
            print("Frame #%4d" % frame_idx)

            # Calculate background image
            bg_img = self.cdn_net.calculate_background(data_loader.data_frame, batch_size = 16)

            input_img = (data_loader.data_frame[..., (data_loader.current_frame_idx % data_loader.FPS)] * 255.0).type(torch.uint8).cpu().data.numpy()
            input_img = input_img.reshape([data_loader.img_heigh, data_loader.img_width, data_loader.img_channel])

            # Display background image
            cv.imshow('Input image',input_img)
            cv.imshow('Background image',bg_img)
            cv.waitKey(1)

            # Shift the sliding windows for the next iteration
            data_loader.load_next_k_frame(1)

        cv.destroyAllWindows()

# Execute the background trainer
bg_trainer = BackgroundTrainer(config_file='config.json')
# bg_trainer.train(camera_name="camera")
bg_trainer.demo_background()

