//
// @file    : model.h
// @purpose : A header file for the architecture of Foreground Detection Network (FDN)
// @author  : Hung Ngoc Phan
// @project : Distributed training of Foreground Detection Network with Event-Triggered Data Parallelism
// @licensed: N/A
// @created : 10/05/2021
// @modified: 16/05/2021
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _MODEL_H_
#define _MODEL_H_

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

///////////////////////////////////////////////////////////////////////
// Model definition: FDNnet
///////////////////////////////////////////////////////////////////////
struct Model : torch::nn::Module {
  Model()
    : dconv1       (torch::nn::Conv2dOptions(6,6,3).stride(1).padding(1).groups(6).bias(false)),
      pconv1       (torch::nn::Conv2dOptions(6,16,1).stride(1).groups(1).bias(false)),
      dconv2       (torch::nn::Conv2dOptions(16,16,3).stride(1).padding(1).groups(16).bias(false)),
      pconv2       (torch::nn::Conv2dOptions(16,16,1).stride(1).groups(1).bias(false)),
      dconv3       (torch::nn::Conv2dOptions(16,16,3).stride(1).padding(1).groups(16).bias(false)),
      pconv3       (torch::nn::Conv2dOptions(16,16,1).stride(1).groups(1).bias(false)),    
      dconv4       (torch::nn::Conv2dOptions(16,16,3).stride(1).padding(1).groups(16).bias(false)),
      pconv4       (torch::nn::Conv2dOptions(16,16,1).stride(1).groups(1).bias(false)),    
      dconv5       (torch::nn::Conv2dOptions(16,16,3).stride(1).padding(1).groups(16).bias(false)),
      pconv5       (torch::nn::Conv2dOptions(16,16,1).stride(1).groups(1).bias(false)), 
      dconv6       (torch::nn::Conv2dOptions(16,16,3).stride(1).padding(1).groups(16).bias(false)),
      pconv6       (torch::nn::Conv2dOptions(16,16,1).stride(1).groups(1).bias(false)),    
      dconv7       (torch::nn::Conv2dOptions(16,16,3).stride(1).padding(1).groups(16).bias(false)),
      pconv7       (torch::nn::Conv2dOptions(16,16,1).stride(1).groups(1).bias(false)),    
      dconv8       (torch::nn::Conv2dOptions(16,16,3).stride(1).padding(1).groups(16).bias(false)),
      pconv8       (torch::nn::Conv2dOptions(16,1,1).stride(1).groups(1).bias(false)) {
    
    register_module("dconv1", dconv1);
    register_module("pconv1", pconv1);
    register_module("dconv2", dconv2);
    register_module("pconv2", pconv2);
    register_module("dconv3", dconv3);
    register_module("pconv3", pconv3);
    register_module("dconv4", dconv4);
    register_module("pconv4", pconv4);
    register_module("dconv5", dconv5);
    register_module("pconv5", pconv5);
    register_module("dconv6", dconv6);
    register_module("pconv6", pconv6);
    register_module("dconv7", dconv7);
    register_module("pconv7", pconv7);
    register_module("dconv8", dconv8);
    register_module("pconv8", pconv8);
  }

  torch::Tensor forward(torch::Tensor x) {
    // Group 1
    x = torch::nn::functional::relu(             // depthwise separable 1
        pconv1->forward(dconv1->forward(x))
    );     
    x = torch::nn::functional::relu(             // depthwise separable 2
        pconv2->forward(dconv2->forward(x))
    );    
    
    // Group 2
    x = torch::max_pool2d(x, 2);                 // Sampling -> shape/2
    x = torch::nn::functional::relu(             // depthwise separable 3
        pconv3->forward(dconv3->forward(x))
    );     
    x = torch::nn::functional::relu(             // depthwise separable 4
        pconv4->forward(dconv4->forward(x))
    );     
    
    // Group 3
    x = torch::max_pool2d(x, 2);                 // Sampling -> shape/4
    x = pconv5->forward(dconv5->forward(x));     // depthwise separable 5
    x = torch::nn::functional::relu(
        torch::nn::functional::instance_norm(x)
    ); 
    
    // Group 4
    x = torch::nn::functional::interpolate(      // UpSampling -> shape/2
        x, 
        torch::nn::functional::InterpolateFuncOptions()
            .scale_factor(std::vector<double>({2.0, 2.0}))
            .mode(torch::kNearest)
    );
    x = pconv6->forward(dconv6->forward(x));     // depthwise separable 6
    x = torch::nn::functional::relu(             // Instance norm 1 + ReLU
        torch::nn::functional::instance_norm(x)
    ); 

    // Group 5
    x = torch::nn::functional::interpolate(      // UpSampling -> shape
        x, 
        torch::nn::functional::InterpolateFuncOptions()
            .scale_factor(std::vector<double>({2.0, 2.0}))
            .mode(torch::kNearest)
    );
    x = pconv7->forward(dconv7->forward(x));     // depthwise separable 7
    x = torch::nn::functional::relu(             // Instance norm 2 + ReLU
        torch::nn::functional::instance_norm(x)
    ); 
    
    // Group 6
    x = torch::sigmoid(
        pconv8->forward(dconv8->forward(x))
    );
      
    return x;
  }
    
  // Pre-defined layers
  torch::nn::Conv2d dconv1;  // The depthwise separable layer #1
  torch::nn::Conv2d pconv1;
  torch::nn::Conv2d dconv2;  // The depthwise separable layer #2
  torch::nn::Conv2d pconv2;
  torch::nn::Conv2d dconv3;  // The depthwise separable layer #3
  torch::nn::Conv2d pconv3;
  torch::nn::Conv2d dconv4;  // The depthwise separable layer #4
  torch::nn::Conv2d pconv4;
  torch::nn::Conv2d dconv5;  // The depthwise separable layer #5
  torch::nn::Conv2d pconv5;
  torch::nn::Conv2d dconv6;  // The depthwise separable layer #6
  torch::nn::Conv2d pconv6;
  torch::nn::Conv2d dconv7;  // The depthwise separable layer #7
  torch::nn::Conv2d pconv7;
  torch::nn::Conv2d dconv8;  // The depthwise separable layer #8
  torch::nn::Conv2d pconv8;
};

///////////////////////////////////////////////////////////////////////
// Learning metrics: Loss function and Accuracy
///////////////////////////////////////////////////////////////////////
torch::Tensor model_loss(torch::Tensor& y_pred, torch::Tensor& y_true){
    torch::nn::BCELoss loss_criterion;
    return loss_criterion(y_pred, y_true);
}

torch::Tensor model_accuracy(torch::Tensor& y_pred, torch::Tensor& y_true){
    torch::Tensor round_pred = torch::round(y_pred);
    return torch::mean(torch::eq(round_pred, y_true).to(torch::kFloat32));
}


#endif