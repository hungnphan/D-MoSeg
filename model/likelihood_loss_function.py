import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import params
import math

class TrainingCriterion:
    def __init__(self, args):
        self.K_MIXTURES = args.KMIXTURE

    def normal_distribution_func(self, y, mu, variance):
        """
            Compute the value of each Gaussian components with the corresponding means and variances
            on the given data points (<y>).
            This code calculate the function given here:
                https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Non-degenerate_case
                (The formula is given in <Non-degenerate case> section)

            Input:
                y: A TF tensor of the input data points, shape = (N, FPS, 3)
                mu: A TF tensor of the means of the Gaussian components, shape = (N, K_MIXTURES, 3)
                variance: A TF tensor of the variances of the Gaussian components, shape = (N, K_MIXTURES)

            Output:
                The calculated value of each of the K_MIXTURES Gaussian function for each data point. (N, FPS, K_MIXTURES)
        """

        y_tmp = y.unsqueeze(2)          # (N, FPS, 1, 3)
        mu_tmp = mu.unsqueeze(1)        # (N, 1, K_MIXTURES, 3)

        # Calculate determinant of the <diagonal> covariance matrix (has formula)
        covariance_det = torch.pow(variance, 3).unsqueeze(1)    # (N, 1, K_MIXTURES)
        
        # Inverse of the covariance matrix generated with the appropriate shape
        # Basically, this line do this:
        #   1. Generate an identity matrix of shape (1, K_MIXTURES, 3, 3) (3 is the image depth).
        #   2. Invert the variances, i.e. turn x into 1/x.
        #   3. Expand the dimension of the tensor in step 2 to multiple with the identity matrix
        #   4. Expand the dimension of the resulting tensor to get the tensor of shape (1, 1, K_MIXTURES, 3, 3)
        # The resulting tensor is basically an inverse covariance matrix for each Gaussian component.    
        # (1, K_MIXTURES, 3, 3) * (N, K_MIXTURES, 1, 1) -> [N, 1, K_MIXTURES, 3, 3]
        
        inverse_covariance = (torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(1, self.K_MIXTURES, 1, 1).cuda() \
            * torch.reciprocal(variance).unsqueeze(-1).unsqueeze(-1)).unsqueeze(1)   # shape = [N, 1, K_MIXTURES, 3, 3]
 
        # Pairwise difference between the means and the data points,
        # i.e. each data point is subtracted with each component.
        diff = (y_tmp - mu_tmp).unsqueeze(-1)   #shape = (N, FPS, K_MIXTURES, 3, 1)

        # Tranposed version of the <diff> tensor to calculate the numerator of the
        # multivariate normal distribution formula.
        diff_transpose = diff.permute(0, 1, 2, 4, 3) #shape = (N, FPS, K_MIXTURES, 1, 3)
        
        # The numerator of the formula on the Wikipedia page
        # This line utilizes the broadcasting rule of TF tensors,
        # i.e. the dimensions of the tensors in the calculation should match each other
        # or one of the axis with different dimension size should be 1, e.g. the 3rd axis
        # of tensor x and y are 1 and 9 in order is valid.
        # 
        # After calculating the denominator, the dimensions of size 1 is omitted by applying tf.squeeze 
        numerator = torch.squeeze(torch.exp(-0.5 * torch.matmul(torch.matmul(diff_transpose, inverse_covariance), 
                                                                diff))) # shape = (N, FPS, K_MIXTURES)

        # The denominator
        denominator = torch.sqrt(math.pow(2.0*math.pi, 3) * covariance_det) # shape = (N, 1, K_MIXTURES)

        return torch.divide(numerator, denominator) # shape = (N, FPS, K_MIXTURES)


    def likelihood_loss(self, x, y_pred):
        """
            Compute log-likelihood loss. This function hypothesize that the mean of the Gaussian components is
            normalized into the range [0, 1] from [0, 255]. The variances of the components also limited to
            [16, 32] to prevent the loss goes NaN and also to prevent the component spanning the whole
            space of the given data points.

            Input:
                x: A TF tensor of the input data points of the current iteration, shape = (N, 1, FPS, 3)
                y_pred: A TF tensor of the corresponding prediction of Gaussian components on these data points,
                    shape = (N, K_MIXTURES*5)
            
            Ouput:
                Average likelihood loss of one batch.
        """

        x = torch.squeeze(x)
        out_pi, out_variance, out_mu = torch.split(y_pred, (self.K_MIXTURES, self.K_MIXTURES, 3*self.K_MIXTURES), dim=-1)

        # make the variance has range [16, 32], prevent the variance from getting to small or too big
        out_variance = (16.0 + 16.0*out_variance) / 255.0

        out_mu = out_mu.view(-1, self.K_MIXTURES, 3) #shape = (N, K_MIXTURES, 3)

        out_pi = out_pi.unsqueeze(1) #shape = (N, 1, K_MIXTURES)

        result = self.normal_distribution_func(x, out_mu, out_variance) #shape = (N, FPS, K_MIXTURES)
        
        # Multiply the Gaussian components output with its corresponding weights
        # (N, FPS, K_MIXTURES) * (N, 1, K_MIXTURES) -> shape = (N, FPS, K_MIXTURES)
        result = result * out_pi 

        # Sum the result over the K_MIXTURES dimension
        result = torch.sum(result, dim=-1, keepdim=True) #shape = (N, FPS)

        # Average over FPS (temporal frames) dimension
        result = -torch.sum(torch.log(result), dim=-1)  #shape = (N,)

        # print(f"loss shape = {result.shape}")

        return torch.mean(result)   # average over batch dimension






# from arg_parser import parse_config_from_json

# # x: A TF tensor of the input data points of the current iteration, shape = (N, 1, FPS, 3)
# # y_pred: A TF tensor of the corresponding prediction of Gaussian components on these data points,
# # shape = (N, K_MIXTURES*5)
# N = 23
# FPS = 240
# KMIX = 4

# x       = torch.from_numpy(np.random.randint(0, 256, size=[N, 1, FPS, 3])).type(torch.FloatTensor).cuda()
# y_pred  = torch.from_numpy(np.random.randint(0, 256, size=[N, KMIX*5])).type(torch.FloatTensor).cuda()

# model_pi = torch.
# [N, K_MIXTURES]
# model_sigma =   [N, K_MIXTURES]
# model_mu =      [N, K_MIXTURES * 3]    





# config = parse_config_from_json(config_file='config.json')
# train_criterion = TrainingCriterion(config)

# loss = train_criterion.likelihood_loss(x, y_pred)
# print(f"loss.shape = {loss.shape}")
# print(f"loss = {loss.item()}")

# # Test normal_distribution_func(y, mu, variance)
# # y: A TF tensor of the input data points, shape = (N, FPS, 3)
# # mu: A TF tensor of the means of the Gaussian components, shape = (N, K_MIXTURES, 3)
# # variance: A TF tensor of the variances of the Gaussian components, shape = (N, K_MIXTURES)
# N = 23
# FPS = 240
# KMIX = 4

# y = torch.from_numpy(np.random.randint(0, 256, size=[N, FPS, 3])).type(torch.FloatTensor)
# mu = torch.from_numpy(np.random.randint(0, 256, size=[N, KMIX, 3])).type(torch.FloatTensor)
# variance = torch.from_numpy(np.random.randint(0, 256, size=[N, KMIX])).type(torch.FloatTensor)

