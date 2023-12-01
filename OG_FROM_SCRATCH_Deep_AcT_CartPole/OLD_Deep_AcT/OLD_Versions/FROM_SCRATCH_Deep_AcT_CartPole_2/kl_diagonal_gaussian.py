import numpy as np
import torch


def kl_divergence_diag_covariance_torch(mu1, sigma1_sq, mu2, sigma2_sq):
    """
    Calculate KL-divergence between two multivariate Gaussian distributions with diagonal covariance matrices.
    
    Parameters:
        mu1 (torch.Tensor): Mean of the first Gaussian distribution.
        sigma1_sq (torch.Tensor): Variance of the first Gaussian distribution.
        mu2 (torch.Tensor): Mean of the second Gaussian distribution.
        sigma2_sq (torch.Tensor): Variance of the second Gaussian distribution.
    
    Returns:
        kl_div (torch.Tensor): KL-divergence between the two Gaussian distributions.
    """
    d = mu1.shape[0]
    
    kl_div = 0.5 * torch.sum(
        (sigma2_sq / sigma1_sq) + (mu1 - mu2)**2 / sigma2_sq - 1 + torch.log(sigma2_sq / sigma1_sq)
    )
    
    return kl_div

if __name__ == '__main__':

    # Example usage
    mu1 = torch.tensor([0.0])
    sigma1_sq = torch.tensor([1.0])
    mu2 = torch.tensor([1.0])
    sigma2_sq = torch.tensor([1.0])

    kl_divergence = kl_divergence_diag_covariance_torch(mu1, sigma1_sq, mu2, sigma2_sq)
    print("KL-divergence:", kl_divergence.item())