import torch
import torch.distributions as dist
import numpy as np

# def negative_log_likelihood(x, μ, diagonal_elements):
#     # Convert inputs to PyTorch tensors
#     x = torch.tensor(x, dtype=torch.float32)
#     μ = torch.tensor(μ, dtype=torch.float32)
#     diagonal_elements = torch.tensor(diagonal_elements, dtype=torch.float32)

#     # Calculate dimensionality (d) from the length of diagonal_elements
#     d = len(diagonal_elements)

#     # Create a diagonal covariance matrix Σ
#     Σ = torch.diag(diagonal_elements)

#     # Calculate the inverse of Σ
#     Σ_inv = torch.inverse(Σ)

#     # Calculate the quadratic term
#     quad_term = 0.5 * torch.matmul(torch.matmul((x - μ).T, Σ_inv), (x - μ))

#     # Calculate the log determinant term
#     log_det_term = 0.5 * torch.log(torch.det(Σ))

#     # Calculate the constant term
#     constant_term = 0.5 * d * torch.log(2 * torch.tensor([3.141592653589793], dtype=torch.float32))

#     # Calculate the negative log likelihood (surprisal)
#     S_x = quad_term + log_det_term + constant_term

#     return S_x.item()  # Convert the result to a Python float

# # Example usage:
# x = [1.0, 2.0]  # Input data
# μ = [0.0, 0.0]  # Mean vector
# diagonal_elements = [1.0, 1.0]  # Diagonal elements of the covariance matrix

# S = negative_log_likelihood(x, μ, diagonal_elements)
# print("Negative Log Likelihood (Surprisal):", S)

def negative_log_likelihood(x, μ, diagonal_elements):
    # Convert inputs to PyTorch tensors
    x = torch.tensor(x, dtype=torch.float32)
    μ = torch.tensor(μ, dtype=torch.float32)
    diagonal_elements = torch.tensor(diagonal_elements, dtype=torch.float32)

    # Calculate dimensionality (d) from the length of diagonal_elements
    d = len(diagonal_elements)

    # Create a diagonal covariance matrix Σ
    Σ = torch.diag(diagonal_elements)

    # Calculate the inverse of Σ
    Σ_inv = torch.inverse(Σ)

    # Transpose x to make the matrix multiplication compatible
    x_t = x.unsqueeze(1)  # Transpose by adding an extra dimension
    x_minus_μ = x_t - μ

    # Calculate the quadratic term
    quad_term = 0.5 * torch.matmul(torch.matmul(x_minus_μ.transpose(0, 1), Σ_inv), x_minus_μ)

    # Calculate the log determinant term
    log_det_term = 0.5 * torch.log(torch.det(Σ))

    # Calculate the constant term
    constant_term = 0.5 * d * torch.log(2 * torch.tensor([np.pi], dtype=torch.float32))

    # Calculate the negative log likelihood (surprisal)
    S_x = quad_term + log_det_term + constant_term

    # Sum all the elements to get a scalar result
    S_x_scalar = S_x.sum()

    return S_x_scalar.item()  # Convert the result to a Python float

# Example usage:
x = [1.0, 2.0]  # Input data
μ = [0.0, 0.0]  # Mean vector
diagonal_elements = [1.0, 1.0]  # Diagonal elements of the covariance matrix

S = negative_log_likelihood(x, μ, diagonal_elements)
print("Negative Log Likelihood (Surprisal):", S)