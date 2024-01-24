import torch
import torch.nn as nn
import torch.optim as optim

# DEFINATELY USE LOG VARIANCES INSTEAD !!!! DOES HIMST AND LANILLOS DO THIS??

class MVGaussianModel(nn.Module):

    def __init__(self, n_inputs, n_outputs, n_hidden, lr=1e-3, device='cpu'):

        super(MVGaussianModel, self).__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden

        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)

        self.mean_fc = nn.Linear(n_hidden, n_outputs)
        self.log_var = nn.Linear(n_hidden, n_outputs)

        self.optimizer = optim.Adam(self.parameters(), lr)
        self.device = device
        self.to(self.device)

    def forward(self, x):

        x_1 = torch.relu(self.fc1(x))
        x_2 = torch.relu(self.fc2(x_1))

        mean = self.mean_fc(x_2)
        log_var = self.log_var(x_2)

        return mean, log_var

    def rsample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(mu)

        return mu + (eps * std)


if __name__ == '__main__':

    # Create instances of Q_phi and P_xi
    input_size_q = 4  # Define input size for Q_phi
    output_size_q = 8  # Define output size for Q_phi
    q_network = MVGaussianModel(input_size_q, output_size_q, 64)

    input_size_p = 16  # Adjust input size for P_xi based on concatenation
    output_size_p = 4  # Define output size for P_xi
    p_network = MVGaussianModel(input_size_p, output_size_p, 64)

    # Define the number of Monte Carlo samples
    N = 30

    # Step 1: Generate State Samples from Q_phi
    inputs_q = torch.randn(N, input_size_q)  # Input to Q_phi (random)

    mu_q, log_var_q = q_network(inputs_q)
    samples_q = q_network.rsample(mu_q, log_var_q)

    # Step 2: Feed State Samples into P_xi
    inputs_p = torch.cat(
        (samples_q, log_var_q), 
        dim = 1
    )

    mu_p, log_var_p = p_network(inputs_p)

    # Step 3: Reparameterize Observation Samples
    samples_p = p_network.rsample(mu_p, log_var_p)

    # Step 4: generate P_xi and calculate the log likelihood values
    multivariate_normal_p = torch.distributions.MultivariateNormal(
        mu_p, 
        covariance_matrix = torch.diag_embed(torch.exp(log_var_p))
    )
    log_likelihood_values_p = multivariate_normal_p.log_prob(samples_p)

    # Step 5: Compute Monte Carlo Estimate
    monte_carlo_estimate = torch.mean(log_likelihood_values_p)

    print("Monte Carlo Estimate:", monte_carlo_estimate.item())



    # # Step 4: Evaluate Log Likelihoods
    # log_likelihood_values_p = torch.distributions.Normal(mu_p, torch.exp(log_var_p)).log_prob(samples_p)










# class GaussianModel(nn.Module):
#     def __init__(self, input_size, output_size, hidden_size, lr=1e-3, device='cpu'):
#         super(GaussianModel, self).__init__()
#         self.fc = nn.Linear(input_size, hidden_size)
#         self.mu = nn.Linear(hidden_size, output_size)
#         self.log_var = nn.Linear(hidden_size, output_size)

#         self.optimizer = optim.Adam(self.parameters(), lr)
#         self.device = device
#         self.to(self.device)

#     def forward(self, input_arg):
#         x = torch.relu(self.fc(input_arg))
#         mu = self.mu(x)
#         log_var = self.log_var(x)
#         return mu, log_var

#     def rsample(self, mu, log_var):
#         std = torch.exp(0.5 * log_var)
#         eps = torch.randn_like(std)

#         return mu + eps * std
