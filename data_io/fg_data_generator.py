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


class FgDataGenerator:
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
        
        # Restore model from last checkpoint
        self.ckpt_dir = os.path.join(self.config.CKPT_DIR, self.scenario_name, self.sequence_name)
        self.path_to_checkpoint = os.path.join(self.ckpt_dir,'cdn.pth')
        if os.path.exists(self.path_to_checkpoint):
            print(f"Loading checkpoint from {self.path_to_checkpoint} ..")
            checkpoint = torch.load(self.path_to_checkpoint)
            
            # Restore checkpoint variable: model_state_dict
            self.cdn_net.load_state_dict(checkpoint['model_state_dict'])

    def prepare_fg_training_data(self):
        # Set model mode to continue training
        self.cdn_net.eval()

        # Iterate all file in fg groundtruth file
        fg_gt_path = os.path.join(self.config.foreground_training_data_dir, self.scenario_name, self.sequence_name+"200")

        # Extract frame_idx of labelled foreground
        file_names = sorted(os.listdir(fg_gt_path))
        labelled_indices = []
        for file_name in file_names:
            if file_name.startswith('gt') and file_name.endswith('.png'):      
                labelled_indices.append(int(file_name[2:8]))
        # print(len(labelled_indices))

        # Create a h5py file stream 
        # print(data_loader.img_heigh, data_loader.img_width)
        fg_data_file = os.path.join(self.config.FG_TRAINING_DATA, self.scenario_name + "_" + self.sequence_name + ".hdf5")
        h5py_writer = h5py.File(fg_data_file, "w")

        # Extract a pair of [input_frame, bg, fg] corresponding with idx in labelled_indices
        data_idx = -1
        for frame_idx in range (self.config.FPS, self.data_loader.data_len):
            # Break the loop if data is out of labelled range
            if (frame_idx+1) > labelled_indices[-1]:
                break

            # If the frame_idx has labelled foreground
            if (frame_idx + 1) in labelled_indices:
                # Set data_index
                data_idx = data_idx + 1
                
                if (data_idx + 1) % 50 == 0:
                    print("Sampling %03d" % data_idx)
                
                # Get input image
                input_img = (self.data_loader.data_frame[..., (self.data_loader.current_frame_idx % self.data_loader.FPS)] * 255.0).type(torch.uint8).cpu().data.numpy()
                input_img = input_img.reshape([self.data_loader.img_heigh, self.data_loader.img_width, self.data_loader.img_channel])

                # Get background image
                bg_img = self.cdn_net.calculate_background(self.data_loader.data_frame, batch_size = 256)

                # Get foreground image 
                fg_img = cv.imread(os.path.join(fg_gt_path,"gt%06d.png" % (frame_idx + 1)), cv.IMREAD_GRAYSCALE)

                # Resize image to the dimension that is divisible vy 4
                out_height = 4*(self.data_loader.img_heigh//4)
                out_width = 4*(self.data_loader.img_width//4)
                input_img = cv.resize(input_img, (out_width, out_height))
                bg_img = cv.resize(bg_img, (out_width, out_height))
                fg_img = cv.resize(fg_img, (out_width, out_height))

                # Preprocess fg image:
                #   Set boundary + shadow -> 255
                #   Set ignore region     -> 0
                raw_fg = fg_img
                fg_img = np.zeros_like(raw_fg)
                fg_binary_mask = np.logical_or(raw_fg>=170,raw_fg==50)
                fg_img[fg_binary_mask] = 255

                # Create data tensor
                fg_img = np.expand_dims(fg_img, axis=-1)
                output_tensor = np.concatenate([input_img, bg_img, fg_img], axis=-1)

                # DEBUG: Display background image
                # cv.imshow('Input image',output_tensor[...,:3])
                # cv.imshow('Background image',output_tensor[...,3:6])
                # cv.imshow('Foreground image',output_tensor[...,6])
                # cv.imshow('Raw Foreground image',raw_fg)
                # cv.waitKey(1)

                h5py_writer.create_dataset("%03d" % data_idx, 
                                           shape=output_tensor.shape, dtype=np.uint8, data=output_tensor,
                                           compression='gzip', compression_opts=9)

            # Shift the sliding windows for the next iteration
            self.data_loader.load_next_k_frame(1)

        # DEBUG: 
        # cv.destroyAllWindows()
    
    def check_fg_training_data(self):   
        # Read file h5py
        # print(data_loader.img_heigh, data_loader.img_width)
        fg_data_file = os.path.join(self.config.FG_TRAINING_DATA, self.scenario_name + "_" + self.sequence_name + ".hdf5")
        h5py_reader = h5py.File(fg_data_file, "r")

        # Check data in h5py file
        indices = sorted(list(h5py_reader.keys()))

        for data_idx in indices:
            # Get data from h5py using key 'data_idx'
            output_tensor = h5py_reader[data_idx]

            # print(output_tensor.shape)

            # DEBUG: Display background image
            cv.imshow('Input image',output_tensor[...,:3])
            cv.imshow('Background image',output_tensor[...,3:6])
            cv.imshow('Foreground image',output_tensor[...,6])
            
            # Toogle pause video process
            key = cv.waitKey(1)
            if key == ord('p'):
                while(cv.waitKey(1) != ord('p')):
                    continue

        cv.destroyAllWindows()

    def export_fg_training_data(self): 
        # Read file h5py
        # print(data_loader.img_heigh, data_loader.img_width)
        fg_data_file = os.path.join(self.config.FG_TRAINING_DATA, self.scenario_name + "_" + self.sequence_name + ".hdf5")
        h5py_reader = h5py.File(fg_data_file, "r")

        # Check data in h5py file
        indices = sorted(list(h5py_reader.keys()))

        # Export data from h5py file to a sequence of frames
        export_dir = os.path.join("data","fg_train_frame",self.scenario_name, self.sequence_name)

        if not os.path.exists(export_dir):
            os.makedirs(export_dir)

        frame_idx = 0
        for data_idx in indices:
            # Get data from h5py using key 'data_idx'
            output_tensor = h5py_reader[data_idx]

            in_img = output_tensor[...,:3]
            bg_img = output_tensor[...,3:6]
            fg_img = output_tensor[...,6]

            # Export data to image files
            frame_idx += 1
            cv.imwrite(os.path.join(export_dir,"%06d_in.png" % frame_idx), in_img)
            cv.imwrite(os.path.join(export_dir,"%06d_bg.png" % frame_idx), bg_img)
            cv.imwrite(os.path.join(export_dir,"%06d_fg.png" % frame_idx), fg_img)



# ##########################################
# # # Execute the background trainer
# fg_generator = FgDataGenerator(config_file='config/config.json', scenario_name='badWeather', sequence_name='skating')

# fg_generator.prepare_fg_training_data()
# fg_generator.check_fg_training_data()

