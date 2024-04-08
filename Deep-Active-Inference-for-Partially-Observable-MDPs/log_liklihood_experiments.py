import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import datetime
import sys
import gym
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt


# Define function to compute log likelihood of sampled observation o given s
def mc_log_likelihood_obs_model(state_sample_phi):

    # Generate the batch of predicted observation beliefs
    mu_xi, log_var_xi = generative_observation_net_xi(state_sample_phi)
    # var_xi = torch.diag_embed(torch.exp(log_var_xi))
    var_xi = torch.exp(log_var_xi)

    # Reparameterize Observation Samples
    obs_sample_xi = generative_observation_net_xi.rsample(mu_xi, log_var_xi)

    log_likelihood = -0.5 * ((obs_sample_xi - mu_xi) ** 2 / var_xi + log_var_xi + torch.log(2 * torch.tensor(torch.pi)))

    # print(f"\n\nlog_likelihood: {log_likelihood}")
    # print(f"torch.sum(log_likelihood): {torch.sum(log_likelihood)}\n\n")

    return torch.sum(log_likelihood)

# Function to compute negative expected log likelihood
def mc_neg_expected_log_evidence(samples_phi):

    neg_log_likelihoods = []

    for i in range(len(samples_phi)):

        # Compute log likelihood of observation o given sampled s
        log_likelihood_i = self.mc_log_likelihood_obs_model(samples_phi[i])
        neg_log_likelihoods.append(log_likelihood_i)

    # Average over all samples
    neg_expected_log_likelihood = -torch.mean(torch.stack(neg_log_likelihoods))

    print(f"neg_expected_log_likelihood: {neg_expected_log_likelihood}")

    return neg_expected_log_likelihood

def univ_gaussian(input, mu, var):
    prob = (1. / np.sqrt(2 * np.pi * var)) * \
        np.exp(-(input - mu) ** 2 / (2 * var))

    return prob

def multiv_gaussian(input, mu, var):
    """
    Compute the probability density function of a univariate Gaussian distribution using PyTorch.

    Parameters:
        input: torch.Tensor
            The input data point or tensor of data points.
        mu: torch.Tensor
            The mean of the Gaussian distribution.
        var: torch.Tensor
            The variance of the Gaussian distribution.

    Returns:
        prob: torch.Tensor
            The probability density function evaluated at the input point(s).
    """
    coefficient = 1. / torch.sqrt(2 * torch.tensor(np.pi) * var)
    exponent = -((input - mu) ** 2) / (2 * var)
    prob = coefficient * torch.exp(exponent)
    return prob

# def multiv_gaussian_log_likelihood(input, mu, var):
#     """
#     Compute the log-likelihood of a multivariate Gaussian distribution using PyTorch.

#     Parameters:
#         input: torch.Tensor
#             The input data point or tensor of data points.
#         mu: torch.Tensor
#             The mean of the Gaussian distribution.
#         var: torch.Tensor
#             The variance of the Gaussian distribution.

#     Returns:
#         log_likelihood: torch.Tensor
#             The log-likelihood of the input point(s).
#     """
#     log_coefficient = -0.5 * (torch.log(2 * torch.tensor(np.pi) * var))
#     exponent = -0.5 * ((input - mu) ** 2) / var
#     log_likelihood = log_coefficient + exponent
#     return log_likelihood

def compute_VFE(
    expected_log_ev, 
    state_mu_batch_t1, state_logvar_batch_t1,
    pred_error_batch_t0t1
):

    # Determine the action distribution for time t1:
    state_batch_t1 = torch.cat((state_mu_batch_t1, state_logvar_batch_t1), dim = 1)
    policy_batch_t1 = policy_net_nu(state_batch_t1)
    
    # Determine the EFEs for time t1:
    EFEs_batch_t1 = value_net_psi(state_batch_t1) # ONLY REPLACE THIS WITH ACTS PLANNING

    # Take a gamma-weighted Boltzmann distribution over the EFEs:
    boltzmann_EFEs_batch_t1 = torch.softmax(-gamma * EFEs_batch_t1, dim=1).clamp(min=1e-9, max=1-1e-9)
    
    # Weigh them according to the action distribution:
    energy_term_batch = -(policy_batch_t1 * torch.log(boltzmann_EFEs_batch_t1)).sum(-1).unsqueeze(1)
    
    # Determine the entropy of the action distribution
    entropy_batch = -(policy_batch_t1 * torch.log(policy_batch_t1)).sum(-1).unsqueeze(1)
    
    # Determine the VFE, then take the mean over all batch samples:
    VFE_batch = expected_log_ev + pred_error_batch_t0t1 + (energy_term_batch - entropy_batch) # OG VERSION
    # VFE_batch = - expected_log_ev + pred_error_batch_t0t1 + (energy_term_batch - entropy_batch)

    VFE = torch.mean(VFE_batch)

    # print(f"\ntorch.mean(expected_log_ev): {torch.mean(expected_log_ev)}\
    #     \ntorch.mean(pred_error_batch_t0t1): {torch.mean(pred_error_batch_t0t1)}\
    #     \ntorch.mean(energy_term_batch): {torch.mean(energy_term_batch)}\
    #     \ntorch.mean(entropy_batch): {torch.mean(entropy_batch)}\
    #     \nVFE: {VFE}\n"
    # )
    
    return VFE

def multiv_gaussian_log_likelihood_S(input_val, mu, covariance_matrix):
    """
    Compute the log-likelihood of a multivariate Gaussian distribution using PyTorch.

    Parameters:
        input_val: torch.Tensor
            The input data point or tensor of data points.
        mu: torch.Tensor
            The mean of the Gaussian distribution.
        covariance_matrix: torch.Tensor
            The covariance matrix of the Gaussian distribution.

    Returns:
        log_likelihood: torch.Tensor
            The log-likelihood of the input point(s).
    """
    D = input_val.size(-1)  # Dimensionality of the multivariate Gaussian
    constant_term = -0.5 * D * torch.log(2 * torch.tensor(np.pi))
    cov_inv = torch.inverse(covariance_matrix)
    determinant = torch.det(covariance_matrix)
    mahalanobis_term = -0.5 * torch.matmul(
    	torch.matmul(
    		(input_val - mu).unsqueeze(-2), 
    		cov_inv
    	), (input_val - mu).unsqueeze(-1)
    ).squeeze()
    log_likelihood = constant_term - 0.5 * torch.log(determinant) + mahalanobis_term

    return log_likelihood


def multiv_gaussian_log_likelihood_T(input_val, mu, covariance_matrix):
	"""
    Compute the log-likelihood of a multivariate Gaussian distribution using PyTorch native functions.

    Parameters:
        input_val: torch.Tensor
            The input data point or tensor of data points. 
        mu: torch.Tensor
            The mean of the Gaussian distribution.
        covariance_matrix: torch.Tensor
            The covariance matrix of the Gaussian distribution.

    Returns:
        log_likelihood: torch.Tensor
            The log-likelihood of the input point(s).
    """

	# Create a MultivariateNormal distribution object
	multivariate_normal = torch.distributions.MultivariateNormal(mu, covariance_matrix)

	# Calculate the log likelihood of the observation given the distribution
	log_likelihood = multivariate_normal.log_prob(input_val)

	return log_likelihood


if __name__ == '__main__':

	state_sample = torch.tensor([0.5, 0.5, 0.5, 0.5])
	mu = torch.tensor([0.0, 0.0, 0.0, 0.0])
	covariance_matrix = torch.eye(4)

	log_li_S = multiv_gaussian_log_likelihood_S(state_sample, mu, covariance_matrix)
	log_li_T = multiv_gaussian_log_likelihood_T(state_sample, mu, covariance_matrix)

	print(f"log_li_S: {log_li_S}")
	print(f"log_li_T: {log_li_T}")



