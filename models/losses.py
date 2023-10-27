"""
Loss functions for the reconstruction term of the ELBO.
"""
import torch
import torch.nn.functional as F


def loss_reconstruction_binary(x, x_decoded_mean, weights):
    x = torch.flatten(x, start_dim=1)
    x_decoded_mean = [torch.flatten(decoded_leaf, start_dim=1) for decoded_leaf in x_decoded_mean]
    loss = torch.sum(
        torch.stack([weights[i] *
                        F.binary_cross_entropy(input = x_decoded_mean[i], target = x, reduction='none').sum(dim=-1)
                        for i in range(len(x_decoded_mean))], dim=-1), dim=-1)
    return loss

def loss_reconstruction_mse(x, x_decoded_mean, weights):
    x = torch.flatten(x, start_dim=1)
    x_decoded_mean = [torch.flatten(decoded_leaf, start_dim=1) for decoded_leaf in x_decoded_mean]
    loss = torch.sum(
        torch.stack([weights[i] *
                        F.mse_loss(input = x_decoded_mean[i], target = x, reduction='none').sum(dim=-1)
                        for i in range(len(x_decoded_mean))], dim=-1), dim=-1)
    return loss

def loss_reconstruction_cov_mse_eval(x, x_decoded_mean, weights):
    # NOTE Only use for evaluation purposes, as the clamping stops gradient flow
    # NOTE WE ASSUME IDENTITY MATRIX BECAUSE WE ASSUME THIS IMPLICITLY WHEN ONLY OPTIMIZING MSE
    scale = torch.diag(torch.ones_like(x_decoded_mean[0])) 
    logpx = torch.zeros_like(weights[0])
    for i in range(len(x_decoded_mean)):
        x_dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=torch.clamp(x_decoded_mean[i],0,1), covariance_matrix=scale)   
        logpx = logpx + weights[i] * x_dist.log_prob(x)
    return logpx
