import numpy as np
import cupy as cp
import cv2 as cv
import torch
from model.fdn import FDN
from model.fg_metric import foreground_accuracy, foreground_loss
from fg_data_io import ForegroundIterableDataset
from args.arg_parser import parse_config_from_json

class ForegroundTrainer:

    def __init__(self):
        
        # Specify device GPU if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Config
        self.config = parse_config_from_json(config_file='config/config.json')

        # Initialize data_loader
        fg_dataset = ForegroundIterableDataset(
            rank=0, 
            world_size=1, 
            config=self.config,
            n_replica=4
        )
        self.train_loader = torch.utils.data.DataLoader(
            dataset=fg_dataset,
            batch_size=8,
            shuffle=False,            
            num_workers=1,
            pin_memory=True
        )  

        # Define Foreground Detection Network (FDN)
        self.fdn_net = FDN().to(self.device)
        

        # Define loss function for foreground training
        self.train_criterion = foreground_loss

        # Define network optimizer
        self.optimizer = torch.optim.Adam(self.fdn_net.parameters(), 
                                          lr=self.config.LEARNING_RATE)
        
        # Restore model from last checkpoint


    def train(self):
        # Set model mode to continue training
        self.fdn_net.train()

        losses = 0
        batch_idx = 0

        for train_x, train_y in self.train_loader:
            # for item in range(8):
            #     cv.imshow('Input image',train_x[item,:, :,:3].cpu().detach().numpy())
            #     cv.imshow('Background image',train_x[item,:, :,3:6].cpu().detach().numpy())
            #     cv.imshow('Foreground image',train_y[item,:, :].cpu().detach().numpy())

            #     # Toogle pause video process
            #     key = cv.waitKey(50)
            #     if key == ord('p'):
            #         while(cv.waitKey(1) != ord('p')):
            #             continue

            

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # preprocess data
            train_x = train_x.permute(0,3,1,2).float().cuda() / 255.0
            train_y = train_y.unsqueeze(1).float().cuda() / 255.0

            # Feed forward on CDN
            output = self.fdn_net(train_x)    # output shape = [N, ]
            # print(train_x.shape,train_y.shape,output.shape)

            # x.shape = [N, 1, FPS, 3]  y.shape = [N, K_MIXTURES*5]
            loss = self.train_criterion(output, train_y)
            
            # Backward + optimize
            loss.backward()
            self.optimizer.step()

            losses += loss
            batch_idx += 1

            if (batch_idx) % 5 == 0:
                # Report the average loss at each position of sliding window
                acc = foreground_accuracy(output, train_y)
                print('---> Everage loss = %.5f\tAccuracy = %.5f\n' %(losses/batch_idx, acc))

    def test(self):
        # Set model mode to continue training
        self.fdn_net.eval()

        # Initialize data_loader
        fg_dataset = ForegroundIterableDataset(
            rank=0, 
            world_size=1, 
            config=self.config,
            n_replica=4
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=fg_dataset,
            batch_size=8,
            shuffle=False,            
            num_workers=1,
            pin_memory=True
        )  

        for train_x, train_y in self.train_loader:
            # preprocess data
            processed_train_x = train_x.permute(0,3,1,2).float().cuda() / 255.0
            processed_train_y = train_y.unsqueeze(1).float().cuda() / 255.0

            # Feed forward on CDN
            output = self.fdn_net(processed_train_x)    # output shape = [N, ]
            output = (torch.round(output) * 255).type(torch.uint8).cpu().data.numpy()

            for item in range(8):
                cv.imshow('Input image',train_x[item,:, :,:3].cpu().detach().numpy())
                cv.imshow('Background image',train_x[item,:, :,3:6].cpu().detach().numpy())
                cv.imshow('Foreground image',train_y[item,:, :].cpu().detach().numpy())
                cv.imshow('Predicted Foreground image',output[item].squeeze(0))

                # Toogle pause video process
                key = cv.waitKey(50)
                if key == ord('p'):
                    while(cv.waitKey(1) != ord('p')):
                        continue

        cv.destroyAllWindows()                    


fg_trainer = ForegroundTrainer()
fg_trainer.train()
fg_trainer.test()







