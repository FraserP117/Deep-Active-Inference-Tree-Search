# import torch

# # Define the mean vector (D-dimensional)
# mean_vector = torch.tensor([0.0, 1.0, -1.0])  # Example mean vector with 3 dimensions (D=3)

# # Define the diagonal covariance matrix (DxD)
# # Each diagonal element represents the variance of the corresponding dimension
# # Here, we use diagonal variances of 1.0, 2.0, and 3.0 for illustration
# variances = torch.tensor([1.0, 2.0, 3.0])

# # Create a diagonal multivariate Gaussian distribution
# diagonal_gaussian = torch.distributions.MultivariateNormal(
# 	mean_vector, 
# 	covariance_matrix = torch.diag(variances)
# )

# # Sample from the distribution
# samples = diagonal_gaussian.sample()  # Generates a random sample

# # Calculate the log probability of a sample
# log_prob = diagonal_gaussian.log_prob(samples)

# print("Sample:", samples)
# print("Log Probability:", log_prob)

import torch

# Define the mean vector (D-dimensional)
mean = torch.tensor([0.0, 1.0, -1.0])  # Example mean vector with 3 dimensions (D=3)

# Define the diagonal covariance matrix (DxD)
# Each diagonal element represents the variance of the corresponding dimension
variances = torch.tensor([1.0, 2.0, 3.0])

# Create a diagonal multivariate Gaussian distribution
diagonal_gaussian = torch.distributions.MultivariateNormal(mean, covariance_matrix=torch.diag(variances))

# Use rsample() to sample from the distribution with reparameterization
samples = diagonal_gaussian.rsample()  # Generates a reparameterized sample

print("Reparameterized Sample:", samples)