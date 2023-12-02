def calculate_EFE(self, mean_state_theta, var_state_theta, mean_obs_xi, var_obs_xi):
    with torch.no_grad():
        # 1. Calculate the KL divergence between Q(s_t | pi) and the state prior P(s_t | C):
        state_divergence = self.kl_divergence_diag_cov_gaussian(
            mean_state_theta, var_state_theta,
            mean_state_prior, var_state_prior
        )

        # 2. Calculate the expected entropy of the observation likelihood, under Q(s_t | pi):
        num_samples = 100  # Number of samples to estimate the expectation
        entropy_sum = 0.0

        for _ in range(num_samples):
            
            # Sample s_t from Q(s_t | pi)
            sampled_state = torch.normal(mean_state_theta, torch.sqrt(var_state_theta))

            # Calculate predicted observation mean and variance using generative observation network
            predicted_obs_mean, predicted_obs_var = generative_observation_net(sampled_state)

            # Calculate entropy of p_xi(o_t | s_t) using diagonal_gaussian_entropy
            entropy = diagonal_gaussian_entropy(predicted_obs_var, D)  # D is the dimensionality of observation

            entropy_sum += entropy

        # Average the calculated entropies
        expected_entropy = entropy_sum / num_samples

        # Calculate the expected free energy
        expected_free_energy = state_divergence + expected_entropy

        return expected_free_energy
