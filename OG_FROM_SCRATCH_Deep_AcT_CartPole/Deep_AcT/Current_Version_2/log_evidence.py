import torch

# Define the mean vector (D-dimensional)
mean = torch.tensor([0.0, 0.0])  # Example mean vector with 3 dimensions (D=3)

# Define the diagonal covariance matrix (DxD)
# Each diagonal element represents the variance of the corresponding dimension
variances = torch.tensor([1.0, 1.0])

# Create a diagonal multivariate Gaussian distribution
diagonal_gaussian = torch.distributions.MultivariateNormal(mean, covariance_matrix=torch.diag(variances))

# Assuming you have observed data points (replace this with your actual data)
observed_data = torch.tensor([1.0, 2.0])  # Example observed data

# Calculate the log likelihood for each data point
log_likelihoods = diagonal_gaussian.log_prob(observed_data)

# Compute the log evidence by summing the log likelihoods
log_evidence = log_likelihoods.sum()

print(f"\nlog_likelihoods: {log_likelihoods}")
print(f"Log Evidence: {log_evidence.item()}\n")  # Convert to a scalar value