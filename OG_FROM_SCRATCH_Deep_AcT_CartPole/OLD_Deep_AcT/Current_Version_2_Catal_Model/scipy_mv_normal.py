import numpy as np
import scipy.stats

import torch
import torch.distributions.multivariate_normal as mvn


def diagonal_multivariate_gaussian_pdf(sample, mean, diag_vars):
    """
    Compute the PDF of a diagonal multivariate Gaussian distribution.

    Parameters:
    - sample (numpy.ndarray): The sample vector.
    - mean (numpy.ndarray): The mean vector.
    - diag_vars (list or numpy.ndarray): The diagonal elements of the covariance matrix.

    Returns:
    - pdf (float): The PDF value at the given sample.
    """
    # Ensure that diag_vars is a NumPy array
    diag_vars = np.array(diag_vars)
    
    # Compute the dimension of the distribution
    dim = len(mean)
    
    # Check if the dimensions of sample and mean match
    if sample.shape != mean.shape or len(diag_vars) != dim:
        raise ValueError("Sample, mean, and diag_vars must have the same dimension.")
    
    # Compute the PDF using scipy's multivariate_normal
    pdf = scipy.stats.multivariate_normal.pdf(
        sample, mean = mean, cov = np.diag(diag_vars)
    )
    
    return pdf

def diagonal_multivariate_gaussian_pdf_torch(sample, mean, diag_vars):
    """
    Compute the PDF of a diagonal multivariate Gaussian distribution using PyTorch.

    Parameters:
    - sample (torch.Tensor): The sample tensor.
    - mean (torch.Tensor): The mean tensor.
    - diag_vars (list or torch.Tensor): The diagonal elements of the covariance matrix.

    Returns:
    - pdf (torch.Tensor): The PDF value at the given sample.
    """
    # Ensure that diag_vars is a PyTorch tensor
    if not isinstance(diag_vars, torch.Tensor):
        diag_vars = torch.tensor(diag_vars, dtype=torch.float32)
    
    # Compute the PDF using PyTorch's multivariate_normal
    mvn_dist = mvn.MultivariateNormal(
        loc = mean, 
        covariance_matrix = torch.diag(diag_vars)
    )
    pdf = torch.exp(mvn_dist.log_prob(sample))
    
    return pdf

if __name__ == '__main__':

    # # Example usage:
    # sample = np.array([1.0, 2.0, 3.0, 4.0])  # Replace with your sample
    # mean = np.array([0.0, 0.0, 0.0, 0.0])     # Replace with your mean vector
    # diag_vars = [0.1, 0.2, 0.3, 0.4]              # Replace with your diagonal covariance elements

    # pdf_value = diagonal_multivariate_gaussian_pdf(sample, mean, diag_vars)
    # print(f"PDF at sample {sample}: {pdf_value}")

    # Example usage with torch.Tensor:
    sample = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)  # Replace with your sample
    mean = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)     # Replace with your mean tensor
    diag_vars = [0.1, 0.2, 0.3, 0.4]                                       # Replace with your diagonal covariance elements

    pdf_value = diagonal_multivariate_gaussian_pdf_torch(sample, mean, diag_vars)
    print(f"PDF at sample {sample}: {pdf_value.item()}")