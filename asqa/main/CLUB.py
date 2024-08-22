"""
Adapted from https://github.com/Linear95/CLUB/blob/master/mi_estimators.py
"""

import numpy as np
import math

import torch 
import torch.nn as nn




class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(self, x_dim=4096, y_dim=4096, hidden_size=4096, ):
        super(CLUB, self).__init__()
        # self.is_sampled_version = is_sampled_version
        # p_mu outputs mean of q(Y|X)
        #print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())
        
    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    

    def forward(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)

        # if self.is_sampled_version:
        sample_size = x_samples.shape[0]
        #random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()
        
        positive = - (mu - y_samples)**2 / logvar.exp()
        negative = - (mu - y_samples[random_index])**2 / logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

        mi_upper = upper_bound / 2
        # else:
            
        #     # log of conditional probability of positive sample pairs
        #     positive = - (mu - y_samples)**2 /2./logvar.exp()  
            
        #     prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        #     y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        #     # log of conditional probability of negative sample pairs
        #     negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

        #     mi_upper = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return mi_upper 

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)
