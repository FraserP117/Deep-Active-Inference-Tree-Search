import numpy as np

# Define the means for each state
mean_x = 0.0
mean_x_dot = 0.0
mean_theta = 0.0
mean_theta_dot = 0.0

# Define the variances for each state
var_x = 0.5  # Adjust as needed for x
var_x_dot = 1.0  # Adjust as needed for x_dot
var_theta = 0.001  # Small variance for theta to make it highly peaked at 0
var_theta_dot = 0.001  # Small variance for theta_dot to make it highly peaked at 0

# Create the covariance matrix
cov_matrix = np.diag([var_x, var_x_dot, var_theta, var_theta_dot])

# Sample from the multivariate Gaussian distribution
num_samples = 1000  # Number of samples to generate
samples = np.random.multivariate_normal(
    [mean_x, mean_x_dot, mean_theta, mean_theta_dot],
    cov_matrix,
    num_samples
)

# Print the first few samples
print(samples[:5, :])