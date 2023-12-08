import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.distributions.MultivariateNormal
import torch.optim as optim
import numpy as np
import datetime
import sys
import gym
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt


def Gaussian_kl_divergence(mu1, sigma1, mu2, sigma2):
    """
    Calculate KL divergence between two diagonal multivariate Gaussian distributions.

    Parameters:
        mu1 (torch.Tensor): Mean of the first distribution.
        sigma1 (torch.Tensor): Diagonal standard deviations of the first distribution.
        mu2 (torch.Tensor): Mean of the second distribution.
        sigma2 (torch.Tensor): Diagonal standard deviations of the second distribution.

    Returns:
        torch.Tensor: KL divergence.
    """
    k = mu1.size(-1)  # Dimensionality of the distributions

    trace_term = torch.trace(
    	torch.diag_embed(
    		sigma2.reciprocal()
    	) @ torch.diag_embed(sigma1)
    )

    Mahalanobis_term = torch.t(mu2 - mu1) @ torch.diag_embed(
    	sigma2.reciprocal()
    ) @ (mu2 - mu1)

    log_term = -k + torch.sum(torch.log(sigma2)) - torch.sum(torch.log(sigma1))

    kl = 0.5 * (trace_term + Mahalanobis_term + log_term)

    return kl

def Gaussian_entropy(mu, sigma):
    """
    Calculate entropy of the parameterised multivariate Gaussian distribution.

    Parameters:
        mu (torch.Tensor): Mean of the distribution.
        sigma (torch.Tensor): Diagonal standard deviations of the distribution.

    Returns:
        torch.Tensor: Gaussian Entropy.
    """
    k = mu1.size(-1)  # Dimensionality of the distributions

    log_term = torch.sum(torch.log(sigma))

    entropy = 0.5 * (k + k*torch.log(2*torch.tensor(np.pi)) + log_term)

    return entropy

def monte_carlo_EFE(mu1, sigma1, mu2, sigma2, observation_model, N = 1):
	"""
    Calculates the 'N' Monte Carlo approximation of the Expected Free Energy (EFE).
    sigma1, mu2 parameterise the distribution W.R.T to which the samples are drawn. 

    Parameters:
        mu1 (torch.Tensor): Mean of the first distribution.
        sigma1 (torch.Tensor): Diagonal standard deviations of the first distribution.
        mu2 (torch.Tensor): Mean of the second distribution.
        sigma2 (torch.Tensor): Diagonal standard deviations of the second distribution.
        N: The number of Monte Carlo samples to use when approximating the expected entropy. 

    Returns:
        torch.Tensor: Monte Carlo EFE approximation.
    """

    # calculate the KL divergenced:
    kl_div = Gaussian_kl_divergence(mu1, sigma1, mu2, sigma2)

    # draw N samples from the first distribution:
    epsilon = torch.randn_like(mu1)
    
    # Reparameterization trick: transform the sample to match the target distribution
    sampled_latent_states = torch.tensor([(mu + epsilon * sigma) for i in range(N)])

    # get the predicted/reconstructed observation distribution
    mu_xi, sigma_xi = observation_model(sampled_latent_states) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # compute the observation entropy:
    H_obs = Gaussian_entropy(sigma_xi)

    # Compute the Monte Carlo observation entropy:
    mc_obs_ent = (1/N)*torch.sum(H_obs)

    # compute the final approx EFE:
    mc_efe = kl_div - mc_obs_ent

    return mc_efe





if __name__ == '__main__':

	# Example usage
	mu1 = torch.tensor([1.0])
	sigma1 = torch.tensor([0.5])

	mu2 = torch.tensor([0.0])
	sigma2 = torch.tensor([0.2])

	kl_div = Gaussian_kl_divergence(mu1, sigma1, mu2, sigma2)
	entropy = Gaussian_entropy(mu1, sigma1)

	# print("KL Divergence:", kl_div.item())
	print("entropy:", entropy.item())
