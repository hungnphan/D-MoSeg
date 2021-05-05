import numpy as np
import cupy as cp
import cv2 as cv
import torch
import h5py
import glob
import os
from args.arg_parser import parse_config_from_json
from model.cdn import CDN
from model.likelihood_loss_function import TrainingCriterion
from data_io.bg_data_io import BgDataLoader

class BackgroundTrainer:

    def __init__(self, config_file, scenario_name, sequence_name):      

        # Specify device GPU if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Read method params from json config file
        self.config = parse_config_from_json(config_file=config_file)  

        # Set scenario + sequence name
        self.scenario_name = scenario_name # self.config.scenario_name
        self.sequence_name = sequence_name # self.config.sequence_name

        # Initialize data_loader
        self.data_loader = BgDataLoader(self.config, scenario_name=self.scenario_name, sequence_name=self.sequence_name)

        # Define Convolutional Density Network
        self.cdn_net = CDN(self.config.KMIXTURE, \
                           self.data_loader.img_heigh, self.data_loader.img_width, self.data_loader.img_channel).to(self.device)

        # Define loss function for background training
        self.train_criterion = TrainingCriterion(self.config)

        # Define network optimizer
        self.optimizer = torch.optim.Adam(self.cdn_net.parameters(), lr=self.config.LEARNING_RATE)
        
        # Restore model from last checkpoint
        self.ckpt_dir = os.path.join(self.config.CKPT_DIR, self.scenario_name, self.sequence_name)
        self.path_to_checkpoint = os.path.join(self.ckpt_dir,'cdn.pth')
        if os.path.exists(self.path_to_checkpoint):
            print(f"Loading checkpoint from {self.path_to_checkpoint} ..")
            checkpoint = torch.load(self.path_to_checkpoint)
            
            # Restore checkpoint variable: model_state_dict, optimizer_state_dict, last_sliding_step
            self.cdn_net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.last_sliding_step   = checkpoint['sliding_step']

        else:
            os.makedirs(self.ckpt_dir)
            print('No checkpoint is found ! Start to train from scratch.')
            self.last_sliding_step   = 0

    def train(self):
        # Set model mode to continue training
        self.cdn_net.train()
        
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
        # Set model mode to continue training
        self.cdn_net.eval()

        # Initialize data_loader for output result for demo
        data_loader = BgDataLoader(self.config, scenario_name=self.scenario_name, sequence_name=self.sequence_name)

        for frame_idx in range (self.config.FPS, data_loader.data_len):
            # print("Frame #%4d" % frame_idx)

            # Calculate background image
            bg_img = self.cdn_net.calculate_background(data_loader.data_frame, batch_size = 256)

            input_img = (data_loader.data_frame[..., (data_loader.current_frame_idx % data_loader.FPS)] * 255.0).type(torch.uint8).cpu().data.numpy()
            input_img = input_img.reshape([data_loader.img_heigh, data_loader.img_width, data_loader.img_channel])

            # Display background image
            cv.imshow('Input image',input_img)
            cv.imshow('Background image',bg_img)
            cv.waitKey(1)

            # Shift the sliding windows for the next iteration
            data_loader.load_next_k_frame(1)

        cv.destroyAllWindows()

    # def prepare_fg_training_data(self):        
    #     # Set model mode to continue training
    #     self.cdn_net.eval()

    #     # Initialize data_loader for output result for demo
    #     data_loader = DataLoader(self.config, scenario_name=self.scenario_name, sequence_name=self.sequence_name)

    #     # Iterate all file in fg groundtruth file
    #     fg_gt_path = os.path.join(self.config.foreground_training_data_dir, self.scenario_name, self.sequence_name+"200")

    #     # Extract frame_idx of labelled foreground
    #     file_names = sorted(os.listdir(fg_gt_path))
    #     labelled_indices = []
    #     for file_name in file_names:
    #         if file_name.startswith('gt') and file_name.endswith('.png'):      
    #             labelled_indices.append(int(file_name[2:8]))
    #     # print(len(labelled_indices))

    #     # Create a h5py file stream 
    #     # print(data_loader.img_heigh, data_loader.img_width)
    #     fg_data_file = os.path.join(self.config.FG_TRAINING_DATA, self.scenario_name + "_" + self.sequence_name + ".hdf5")
    #     h5py_writer = h5py.File(fg_data_file, "w")

    #     # Extract a pair of [input_frame, bg, fg] corresponding with idx in labelled_indices
    #     data_idx = -1
    #     for frame_idx in range (self.config.FPS, data_loader.data_len):
    #         # Break the loop if data is out of labelled range
    #         if (frame_idx+1) > labelled_indices[-1]:
    #             break

    #         # If the frame_idx has labelled foreground
    #         if (frame_idx + 1) in labelled_indices:
    #             # Set data_index
    #             data_idx = data_idx + 1
    #             print("Sampling %03d" % data_idx)
                
    #             # Get input image
    #             input_img = (data_loader.data_frame[..., (data_loader.current_frame_idx % data_loader.FPS)] * 255.0).type(torch.uint8).cpu().data.numpy()
    #             input_img = input_img.reshape([data_loader.img_heigh, data_loader.img_width, data_loader.img_channel])

    #             # Get background image
    #             bg_img = self.cdn_net.calculate_background(data_loader.data_frame, batch_size = 256)

    #             # Get foreground image 
    #             fg_img = cv.imread(os.path.join(fg_gt_path,"gt%06d.png" % (frame_idx + 1)), cv.IMREAD_GRAYSCALE)
    #             fg_img = np.expand_dims(fg_img, axis=-1)

    #             output_tensor = np.concatenate([input_img, bg_img, fg_img], axis=-1)
    #             # print(output_tensor.shape)

    #             # DEBUG: Display background image
    #             cv.imshow('Input image',output_tensor[...,:3])
    #             cv.imshow('Background image',output_tensor[...,3:6])
    #             cv.imshow('Foreground image',output_tensor[...,6])
    #             cv.waitKey(1)

    #             h5py_writer.create_dataset(str(data_idx), shape=output_tensor.shape, dtype=np.uint8, data=output_tensor,
    #                                        compression='gzip', compression_opts=9)

    #         # Shift the sliding windows for the next iteration
    #         data_loader.load_next_k_frame(1)

    #     cv.destroyAllWindows()
    
    # def check_fg_training_data(self):        
    #     # Set model mode to continue training
    #     self.cdn_net.eval()

    #     # Initialize data_loader for output result for demo
    #     data_loader = DataLoader(self.config, scenario_name=self.scenario_name, sequence_name=self.sequence_name)

    #     # Iterate all file in fg groundtruth file
    #     fg_gt_path = os.path.join(self.config.foreground_training_data_dir, self.scenario_name, self.sequence_name+"200")

    #     # Extract frame_idx of labelled foreground
    #     file_names = sorted(os.listdir(fg_gt_path))
    #     labelled_indices = []
    #     for file_name in file_names:
    #         if file_name.startswith('gt') and file_name.endswith('.png'):      
    #             labelled_indices.append(int(file_name[2:8]))
    #     # print(len(labelled_indices))

    #     # Create a h5py file stream 
    #     # print(data_loader.img_heigh, data_loader.img_width)
    #     fg_data_file = os.path.join(self.config.FG_TRAINING_DATA, self.scenario_name + "_" + self.sequence_name + ".hdf5")
    #     h5py_reader = h5py.File(fg_data_file, "r")

    #     # # Check data in h5py file
    #     # data_idx = -1
    #     # # print(type(h5py_reader))
    #     # indices = list(h5py_reader.keys())
    #     # # print(len(indices))
    #     # # print(indices)
    #     # for idx in indices:
    #     #     # Set data_index
    #     #     data_idx = data_idx + 1
    #     #     print("Sampling %03d" % int(idx))
            
    #     #     # output_tensor = np.concatenate([input_img, bg_img, fg_img], axis=-1)
    #     #     # print(output_tensor.shape)
    #     #     output_tensor = h5py_reader[idx]

    #     #     # DEBUG: Display background image
    #     #     cv.imshow('Input image',output_tensor[...,:3])
    #     #     cv.imshow('Background image',output_tensor[...,3:6])
    #     #     cv.imshow('Foreground image',output_tensor[...,6])
    #     #     cv.waitKey(1)

    #     # cv.destroyAllWindows()
    


##########################################
# Execute the background trainer
# bg_trainer = BackgroundTrainer(config_file='config/config.json')

# bg_trainer.train()
# bg_trainer.demo_background()

# bg_trainer.train()
# bg_trainer.demo_background()
# bg_trainer.prepare_fg_training_data()
# bg_trainer.check_fg_training_data()

