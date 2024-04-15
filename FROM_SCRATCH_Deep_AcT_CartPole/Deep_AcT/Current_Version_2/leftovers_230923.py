 # def calculate_approximate_EFE(self, mean_state_phi, var_state_phi, mean_obs_xi, var_obs_xi):

        # with torch.no_grad():

        #     # 1. Construct the state prior - this is: [0.0, 0.0, 0.0, 0.0] 

        #     # 1.1 Define the mean vector over preffered hidden states:
        #     mean_x = 0.0
        #     mean_x_dot = 0.0
        #     mean_theta = 0.0
        #     mean_theta_dot = 0.0
        #     mean_state_prior = torch.tensor([mean_x, mean_x_dot, mean_theta, mean_theta_dot])

        #     # 1.2 Define the (diagonal) covariance matrix over preffered hidden states:
        #     var_x = 1.0  # Adjust as needed for x
        #     var_x_dot = 1.0  # Adjust as needed for x_dot
        #     var_theta = 0.001  # Small variance for theta to make it highly peaked at 0
        #     var_theta_dot = 0.001  # Small variance for theta_dot to make it highly peaked at 0

        #     # 1.3 Create the "covariance matrix" - array of diagonal entries of the covariance matrix: 
        #     var_state_prior = torch.tensor([var_x, var_x_dot, var_theta, var_theta_dot])

        #     # 2. Calculate the KL divergence between Q(s_t | pi) and the state prior P(s_t | C):
        #     state_divergence = self.kl_divergence_diag_cov_gaussian(
        #         mean_state_phi, var_state_phi,
        #         mean_state_prior, var_state_prior
        #     )

        #     # 3. calculate the approximate expectatied unconditional observation entropy, under Q(s_t) - using Monte Carlo sampling
        #     N = 10
        #     entropy = self.diagonal_gaussian_entropy(var_obs_xi, 4) # D = 4 for CartPole-v1
        #     ambiguity_term = 0
        #     for i in range(N):
        #         ambiguity_term += self.diagonal_multivariate_gaussian_probs(
        #             # self.sample_diagonal_multivariate_gaussian(mean_state_phi, var_state_phi),
        #             self.variational_transition_net.reparameterize(mean_state_phi, var_state_phi),
        #             mean_state_phi,
        #             var_state_phi
        #         ) * entropy # the probability of sample from sample_diagonal_multivariate_gaussian
        #     expected_entropy = ambiguity_term/N


        #     # 4. Calculate the approximate expected free energy (upper bound on the actual EFE)
        #     expected_free_energy = state_divergence + (1/self.rho) * expected_entropy

        # return expected_free_energy

'''
TESTING TESTING
'''

    # def diagonal_multivariate_gaussian_probs(self, sample, mean, std_devs):

    #     exponent = -0.5 * torch.sum(((sample - mean) / std_devs)**2)
    #     normalization = torch.prod(1.0 / (torch.sqrt(2 * torch.tensor(np.pi)) * std_devs))
    #     pdf = normalization * torch.exp(exponent)

    #     print(f"\n\ntype(sample): {type(sample)}")
    #     print(f"type(mean): {type(mean)}")
    #     print(f"type(std_devs): {type(std_devs)}\n") # I don't think these are stdevs any more...
    #     print(f"exponent: {exponent}")
    #     print(f"normalization: {normalization}")
    #     print(f"pdf: {pdf}\n\n")

    #     # exponent: -225627.21875
    #     # normalization: inf
    #     # pdf: nan

    #     breakpoint()

    #     return pdf

    # def diagonal_multivariate_gaussian_probs(self, sample, mean, diag_vars):
    #     """
    #     Compute the PDF of a diagonal multivariate Gaussian distribution using PyTorch.

    #     Parameters:
    #     - sample (torch.Tensor): The sample tensor.
    #     - mean (torch.Tensor): The mean tensor.
    #     - diag_vars (list or torch.Tensor): The diagonal elements of the covariance matrix.

    #     Returns:
    #     - pdf (torch.Tensor): The PDF value at the given sample.
    #     """
    #     # Ensure that diag_vars is a PyTorch tensor
    #     if not isinstance(diag_vars, torch.Tensor):
    #         diag_vars = torch.tensor(diag_vars, dtype=torch.float32)

    #     print(f"\nsample: {sample}")
    #     print(f"mean: {mean}")
    #     print(f"diag_vars: {diag_vars}\n")
        
    #     # Compute the PDF using PyTorch's multivariate_normal
    #     mvn_dist = mvn.MultivariateNormal(
    #         loc = mean, 
    #         covariance_matrix = torch.diag(diag_vars)
    #     )
    #     pdf = torch.exp(mvn_dist.log_prob(sample))

    #     print(f"\n\nsample: {sample}")
    #     print(f"mean: {mean}")
    #     print(f"diag_vars: {diag_vars}\n")
    #     print(f"mvn_dist: {mvn_dist}")
    #     print(f"pdf: {pdf}\n\n")

    #     breakpoint()
        
    #     return pdf

    # def diagonal_multivariate_gaussian_probs(self, sample, mean, diag_vars):
    #     """
    #     Compute the PDF of a diagonal multivariate Gaussian distribution using PyTorch.

    #     Parameters:
    #     - sample (torch.Tensor): The sample tensor.
    #     - mean (torch.Tensor): The mean tensor.
    #     - diag_vars (list or torch.Tensor): The diagonal elements of the covariance matrix.

    #     Returns:
    #     - pdf (torch.Tensor): The PDF value at the given sample.
    #     """
    #     # Ensure that diag_vars is a PyTorch tensor
    #     if not isinstance(diag_vars, torch.Tensor):
    #         diag_vars = torch.tensor(diag_vars, dtype = torch.float32)

    #     print(f"\nmean: {mean}")
    #     print(f"diag_vars: {diag_vars}\n")

    #     breakpoint()

    #     # Expand the dimensions of diag_vars to create a valid covariance matrix
    #     cov_matrix = torch.diag(diag_vars)
    #     # Add batch dimensions if needed
    #     if len(cov_matrix.shape) == 2:
    #         # cov_matrix = cov_matrix.unsqueeze(0).expand(sample.shape[0], -1, -1)
    #         print(f"AHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH!!!!!!!!!!!!!!!!!!!!!!!!")

    #     print(f"\ncov_matrix: {cov_matrix}\n")

    #     breakpoint()

    #     # Compute the PDF using PyTorch's multivariate_normal
    #     mvn_dist = mvn.MultivariateNormal(
    #         loc = mean,
    #         covariance_matrix = cov_matrix
    #     )
    #     pdf = torch.exp(mvn_dist.log_prob(sample))

    #     return pdf

    # def diagonal_multivariate_gaussian_probs(self, samples, means, diag_vars):
    #     """
    #     Compute the PDF of a diagonal multivariate Gaussian distribution using PyTorch.

    #     Parameters:
    #     - samples (torch.Tensor): The samples tensor (batch_size x num_features).
    #     - means (torch.Tensor): The means tensor (batch_size x num_features).
    #     - diag_vars (torch.Tensor): The diagonal elements of the covariance matrix (batch_size x num_features).

    #     Returns:
    #     - pdf (torch.Tensor): The average PDF value over the batch.
    #     """

    #     # Compute the PDF for each row in the batch
    #     mvn_dists = mvn.MultivariateNormal(
    #         loc = means,
    #         covariance_matrix = torch.diag_embed(diag_vars)  # Convert diag_vars to a diagonal matrix
    #     )
    #     pdfs = torch.exp(mvn_dists.log_prob(samples))

    #     # Compute the average PDF over the batch
    #     pdf = torch.mean(pdfs)

    #     print(f"\n\nsamples: {samples}")
    #     print(f"means: {means}")
    #     print(f"diag_vars: {diag_vars}\n")
    #     print(f"mvn_dists: {mvn_dists}")
    #     print(f"pdfs: {pdfs}")
    #     print(f"pdf: {pdf}\n\n")

    #     breakpoint()

    #     return pdf
















    # def diagonal_multivariate_gaussian_probs(self, samples, means, diag_vars):
    #     """
    #     Compute the PDF of a diagonal multivariate Gaussian distribution using PyTorch.

    #     Parameters:
    #     - samples (torch.Tensor): The samples tensor (batch_size x num_features).
    #     - means (torch.Tensor): The means tensor (batch_size x num_features).
    #     - diag_vars (torch.Tensor): The diagonal elements of the covariance matrix (batch_size x num_features).

    #     Returns:
    #     - pdf (torch.Tensor): The average PDF value over the batch.
    #     """

    #     batch_size = len(samples)

    #     final_pdf = 0

    #     for i in range(batch_size):

    #         # Compute the PDF for each row in the batch
    #         mvn_dist_i = mvn.MultivariateNormal(
    #             loc = means[i],
    #             covariance_matrix = torch.diag_embed(diag_vars[i])  # Convert diag_vars to a diagonal matrix
    #         )

    #         print(f"\ntorch.exp(mvn_dist_{i}.log_prob(samples[{i}])): {torch.exp(mvn_dist_i.log_prob(samples[i]))}\n")

    #         pdf_i = torch.exp(mvn_dist_i.log_prob(samples[i]))

    #         final_pdf += pdf_i

    #         # print(f"\npdf_i: {pdf_i}\n")

    #     # Compute the average PDF over the batch
    #     # pdf = torch.mean(pdfs)
    #     pdf = final_pdf / batch_size

    #     # print(f"\n\nsamples: {samples}")
    #     # print(f"means: {means}")
    #     # print(f"diag_vars: {diag_vars}\n")
    #     # print(f"pdf: {pdf}\n\n")

    #     # breakpoint()

    #     return pdf



    # def calculate_approximate_EFE(self, mean_state_phi, var_state_phi, mean_obs_xi, var_obs_xi):

    #     with torch.no_grad():

    #         # 1. Construct the state prior - this is: [0.0, 0.0, 0.0, 0.0] 

    #         # 1.1 Define the mean vector over preffered hidden states:
    #         mean_x = 0.0
    #         mean_x_dot = 0.0
    #         mean_theta = 0.0
    #         mean_theta_dot = 0.0
    #         mean_state_prior = torch.tensor([mean_x, mean_x_dot, mean_theta, mean_theta_dot])

    #         # 1.2 Define the (diagonal) covariance matrix over preffered hidden states:
    #         var_x = 1.0  # Adjust as needed for x
    #         var_x_dot = 1.0  # Adjust as needed for x_dot
    #         var_theta = 0.001  # Small variance for theta to make it highly peaked at 0
    #         var_theta_dot = 0.001  # Small variance for theta_dot to make it highly peaked at 0

    #         # 1.3 Create the "covariance matrix" - array of diagonal entries of the covariance matrix: 
    #         var_state_prior = torch.tensor([var_x, var_x_dot, var_theta, var_theta_dot])

    #         # 2. Calculate the KL divergence between Q(s_t | pi) and the state prior P(s_t | C):
    #         state_divergence = self.kl_divergence_diag_cov_gaussian(
    #             mean_state_phi, var_state_phi,
    #             mean_state_prior, var_state_prior
    #         )

    #         # Generate the raparamed state samples:
    #         hidden_state_samples = self.variational_transition_net.reparameterize(mean_state_phi, var_state_phi)

    #         # get the associated predicted observations:
    #         mean_observations, var_observations = self.generative_observation_net(hidden_state_samples)

    #         # Generate the reparamed observation samples:
    #         observation_samples = self.generative_observation_net.reparameterize(mean_observations, var_observations)

    #         # print(f"\nhidden_state_samples.shape: {hidden_state_samples.shape}")
    #         # print(f"mean_observations.shape: {mean_observations.shape}")
    #         # print(f"var_observations.shape: {var_observations.shape}")
    #         # print(f"observation_samples.shape: {observation_samples.shape}\n")




    #         # # 3. calculate the approximate expectatied unconditional observation entropy, under Q(s_t) - using Monte Carlo sampling
    #         # N = 10
    #         # entropy = self.diagonal_gaussian_entropy(var_obs_xi, 4) # D = 4 for CartPole-v1
    #         # ambiguity_term = 0
    #         # for i in range(N):
    #         #     ambiguity_term += self.diagonal_multivariate_gaussian_probs(
    #         #         # self.sample_diagonal_multivariate_gaussian(mean_state_phi, var_state_phi),
    #         #         self.variational_transition_net.reparameterize(mean_state_phi, var_state_phi),
    #         #         mean_state_phi,
    #         #         var_state_phi
    #         #     ) * entropy # the probability of sample from sample_diagonal_multivariate_gaussian
    #         # expected_entropy = ambiguity_term/N


    #         # # 4. Calculate the approximate expected free energy (upper bound on the actual EFE)
    #         # expected_free_energy = state_divergence + (1/self.rho) * expected_entropy

    #     return expected_free_energy


        # def kl_divergence_diag_cov_gaussian(self, mu1, sigma1_sq, mu2, sigma2_sq):

    #     kl_div = 0.5 * torch.sum(
    #         (torch.sqrt(sigma2_sq) / torch.sqrt(sigma1_sq)) + ((mu1 - mu2)**2 / sigma1_sq) - 1 + torch.log(torch.sqrt(sigma1_sq) / torch.sqrt(sigma2_sq))
    #     )

    #     # print(f"\nkl_div: {kl_div}")
    #     # print(f"mu1.shape: {mu1.shape}")
    #     # print(f"sigma1_sq.shape: {sigma1_sq.shape}")
    #     # print(f"mu2.shape: {mu2.shape}")
    #     # print(f"sigma2_sq.shape: {sigma2_sq.shape}\n")

    #     # breakpoint()

    #     return kl_div

    # def diag_cov_gaussian_log_li(self, predicted_cur_obs_mean, predicted_cur_obs_var, observation_samples, D):

    #     nll = 0.5 * torch.sum(
    #         torch.log(predicted_cur_obs_var) + (((observation_samples - predicted_cur_obs_mean) ** 2) / predicted_cur_obs_var) + D * torch.log(torch.tensor(2 * np.pi)),  # Constant term due to the Gaussian distribution
    #         dim = 1  # Sum along the state dimensionality
    #     )

    #     # print(f"\nnll: {nll}")
    #     # print(f"predicted_cur_obs_mean.shape: {predicted_cur_obs_mean.shape}")
    #     # print(f"predicted_cur_obs_var.shape: {predicted_cur_obs_var.shape}")
    #     # print(f"observation_samples.shape: {observation_samples.shape}\n")

    #     # breakpoint()

    #     return torch.mean(nll)


        # def calculate_free_energy_loss(self, pred_mean_state_batch_t1, action_batch_t1):

    #     # generative transition net inputs:
    #     pred_mean_cur_state_and_cur_action = torch.cat((pred_mean_state_batch_t1.detach(), action_batch_t1.float()), dim=1)

    #     # variational transition net inputs:
    #     current_action = action_batch_t1.float() # must cast to float

    #     # generative observation net inputs:
    #     pred_mean_cur_state = pred_mean_state_batch_t1

    #     # Forward pass through the networks:
    #     generative_next_state_distribution_mean, generative_next_state_distribution_var = self.generative_transition_net(pred_mean_cur_state_and_cur_action)  # predicted next state prob (gen)
    #     # generative_next_state_distribution_var = torch.exp(generative_next_state_distribution_var)
    #     # print(f"\n\ngenerative_next_state_distribution_mean: {generative_next_state_distribution_mean}")
    #     # print(f"generative_next_state_distribution_var: {generative_next_state_distribution_var}\n")

    #     variational_next_state_distribution_mean, variational_next_state_distribution_var = self.variational_transition_net(current_action) # predicted next state prob (var)
    #     # variational_next_state_distribution_var = torch.exp(variational_next_state_distribution_var)
    #     # print(f"variational_next_state_distribution_mean: {variational_next_state_distribution_mean}")
    #     # print(f"variational_next_state_distribution_var: {variational_next_state_distribution_var}")

    #     generative_cur_observation_distribution_mean, generative_cur_observation_distribution_var = self.generative_observation_net(pred_mean_cur_state) # predicted current obs (gen)
    #     # generative_cur_observation_distribution_var = torch.exp(generative_cur_observation_distribution_var)
    #     # print(f"generative_cur_observation_distribution_mean: {generative_cur_observation_distribution_mean}")
    #     # print(f"generative_cur_observation_distribution_var: {generative_cur_observation_distribution_var}\n\n")

    #     # Calculate the predicted state, KL divergence between the generative and variational models
    #     kl_divergence = self.kl_divergence_diag_cov_gaussian(
    #         variational_next_state_distribution_mean, variational_next_state_distribution_var, # q_\phi
    #         generative_next_state_distribution_mean, generative_next_state_distribution_var # p_\theta
    #     )

    #     # Use the reparameterization trick to sample from the variational state distribution
    #     # reparamed_hidden_state_samples = self.generative_transition_net.reparameterize(generative_next_state_distribution_mean, generative_next_state_distribution_log_var) # keept as logvar 
    #     reparamed_hidden_state_samples = self.generative_transition_net.reparameterize(generative_next_state_distribution_mean, generative_next_state_distribution_var) # keept as logvar 

    #     # predict the corresponding observations for the 'reparamed_hidden_state_samples':
    #     # generative_observation_mean, generative_observation_log_var = self.generative_observation_net(reparamed_hidden_state_samples)
    #     generative_observation_mean, generative_observation_var = self.generative_observation_net(reparamed_hidden_state_samples)
    #     # generative_observation_var = torch.exp(generative_observation_log_var)

    #     # Use the reparameterization trick to sample from the generative observation model:
    #     # reparamed_observation_samples = self.generative_observation_net.reparameterize(generative_observation_mean, generative_observation_log_var)
    #     reparamed_observation_samples = self.generative_observation_net.reparameterize(generative_observation_mean, generative_observation_var)

    #     # Calculate the negative log likelihood of the observation
    #     log_likelihood = self.diag_cov_gaussian_log_li(
    #         predicted_cur_obs_mean = generative_cur_observation_distribution_mean, 
    #         predicted_cur_obs_var = generative_cur_observation_distribution_var, 
    #         observation_samples = reparamed_observation_samples,
    #         D = 4 # check the dimensionality of the other inputs, since I think they're already batched.
    #     )

    #     # calculate the probability of the generative transition samples:
    #     gen_trans_probs = self.diagonal_multivariate_gaussian_probs(
    #         reparamed_hidden_state_samples,
    #         generative_next_state_distribution_mean,
    #         generative_next_state_distribution_var
    #     )

    #     evidence_term = gen_trans_probs * log_likelihood

    #     # Compute the final loss
    #     loss = kl_divergence - evidence_term

    #     # print(f"\n\ngenerative_next_state_distribution_mean: {generative_next_state_distribution_mean}")
    #     # print(f"generative_next_state_distribution_var: {generative_next_state_distribution_var}")
    #     # print(f"reparamed_hidden_state_samples: {reparamed_hidden_state_samples}")
    #     # print(f"generative_observation_mean: {generative_observation_mean}")
    #     # print(f"generative_observation_var: {generative_observation_var}")
    #     # print(f"reparamed_observation_samples: {reparamed_observation_samples}")
    #     # print(f"log_likelihood: {log_likelihood}")
    #     # print(f"gen_trans_probs: {gen_trans_probs}")
    #     # print(f"evidence_term: {evidence_term}")
    #     # print(f"loss: {loss}\n\n")

    #     # breakpoint()

    #     # gen_trans_probs: nan
    #     # evidence_term: nan
    #     # loss: nan

    #     return loss


        # def get_actions_AcT(self, node):

    #     unused_actions = []

    #     for action in self.all_actions:

    #         # print(f"action: {action}") # only ever "0""

    #         if action not in node.used_actions: # NOT CORRECT !!!!!
    #             unused_actions.append(action)

    #             return unused_actions

    #         else:
    #             # All actions have been used, so we simply return the best known action
    #             sorted_children = sorted(node.children, key = lambda child: child.predictive_EFE)
    #             action_with_minimum_EFE = sorted_children[0].action_at_tau_minus_one

    #             return action_with_minimum_EFE

    # def get_actions_AcT(self, node):

    #     global get_actions_ctr

    #     get_actions_ctr += 1

    #     unused_actions = []

    #     for action in self.all_actions:

    #         if action not in node.used_actions:

    #             # print(f"\naction: {action} un-used for node.name: {node.name}\n")

    #             unused_actions.append(action)

    #         else:

    #             # All actions have been used, so we simply return the best known action
    #             sorted_children = sorted(node.children, key = lambda child: child.predictive_EFE)
    #             action_with_minimum_EFE = sorted_children[0].action_at_tau_minus_one

    #             # print(f"\naction_with_minimum_EFE: {action_with_minimum_EFE} for node.name: {node.name}\n")

    #             return action_with_minimum_EFE

    #     print(f"\nget_actions_AcT - unique_node_id: {unique_node_id}")
    #     print(f"get_actions_AcT - unused_actions: {unused_actions}\n")

    #     return unused_actions

    # THIS IS INCORRECT: NEVER DOES VARIATIONAL INFERENCE
    # def active_inferece_tree_search(self, initial_state_belief_mean, delta, epsilon):

    #     # Initializing the planning tree
    #     root_node = Node(
    #         pred_mean_state = initial_state_belief_mean,
    #         action_at_tau_minus_one = None, 
    #         parent = None
    #     )

    #     # Begin AcT planning - hardcoded for 1-ply "planning" for the bootstrap phase
    #     for t in range(1, self.number_AcT_procedure_execs + 1):

    #         # Perform expansion only for bootstrap phase
    #         focal_node = self.expand_AcT(root_node)

    #         # Evaluate the expanded node
    #         g_delta = self.evaluate_AcT(focal_node, delta)

    #         # Perform path integration
    #         self.path_integrate_AcT(focal_node, g_delta)

    #     return root_node


        # def diag_cov_gaussian_log_li(self, predicted_cur_obs_mean, predicted_cur_obs_var, observation_samples, D):
    #     '''
    #     Returns the average log likelihood for batched inputs. 
    #     In case of scaler inputs, simply returns the log likelihood for the istribution.
    #     '''

    #     avg_ll = 0

    #     for i in range(len(predicted_cur_obs_mean)):

    #         ll_i = 0.5 * torch.sum(
    #             torch.log(predicted_cur_obs_var[i]) + (((observation_samples[i] - predicted_cur_obs_mean[i]) ** 2) / predicted_cur_obs_var[i]) + D * torch.log(torch.tensor(2 * np.pi))
    #         )

    #         avg_ll += ll_i

    #     avg_ll = avg_ll / len(predicted_cur_obs_mean)

    #     return avg_ll





        # def calculate_free_energy_loss(self, obs_batch, action_batch):

    #     # variational transition net inputs:
    #     current_action = action_batch.float() # must cast to float

    #     # generative transition net inputs:
    #     # pred_mean_cur_state_and_cur_action = torch.cat((obs_batch.detach(), action_batch.float()), dim=1)
    #     pred_mean_cur_state_and_cur_action = torch.cat((obs_batch, current_action), dim=1)

    #     # Forward pass through the networks:
    #     generative_next_state_distribution_mean, generative_next_state_distribution_var = self.generative_transition_net(pred_mean_cur_state_and_cur_action)  # predicted next state prob (gen)
    #     # generative_next_state_distribution_var = torch.exp(generative_next_state_distribution_var)

    #     variational_next_state_distribution_mean, variational_next_state_distribution_var = self.variational_transition_net(current_action) # predicted next state prob (var)
    #     # variational_next_state_distribution_var = torch.exp(variational_next_state_distribution_var)

    #     # generative_cur_observation_distribution_mean, generative_cur_observation_distribution_var = self.generative_observation_net(obs_batch) # predicted current obs (gen)
    #     # generative_cur_observation_distribution_var = torch.exp(generative_cur_observation_distribution_var)

    #     # Calculate the predicted state, KL divergence between the generative and variational models
    #     kl_divergence = self.kl_divergence_diag_cov_gaussian(
    #         variational_next_state_distribution_mean, variational_next_state_distribution_var, # q_\phi
    #         generative_next_state_distribution_mean, generative_next_state_distribution_var # p_\theta
    #     )

    #     # Use the reparameterization trick to sample from the variational state distribution
    #     reparamed_hidden_state_samples = self.variational_transition_net.reparameterize(variational_next_state_distribution_mean, variational_next_state_distribution_var)
    #     # reparamed_hidden_state_samples = self.generative_transition_net.reparameterize(generative_next_state_distribution_mean, generative_next_state_distribution_var)
    #     # reparamed_hidden_state_samples = self.generative_transition_net.reparameterize(generative_next_state_distribution_mean, generative_next_state_distribution_log_var) # keept as logvar 

    #     # predict the corresponding observations for the 'reparamed_hidden_state_samples':
    #     generative_observation_mean, generative_observation_var = self.generative_observation_net(reparamed_hidden_state_samples)
    #     # generative_observation_mean, generative_observation_log_var = self.generative_observation_net(reparamed_hidden_state_samples)
    #     # generative_observation_var = torch.exp(generative_observation_log_var)

    #     # Use the reparameterization trick to sample from the generative observation model:
    #     reparamed_observation_samples = self.generative_observation_net.reparameterize(generative_observation_mean, generative_observation_var)
    #     # reparamed_observation_samples = self.generative_observation_net.reparameterize(generative_observation_mean, generative_observation_log_var)

    #     # # Calculate the negative log likelihood of the observation
    #     # avg_log_likelihood = self.diag_cov_gaussian_log_li(
    #     #     predicted_cur_obs_mean = generative_cur_observation_distribution_mean, 
    #     #     predicted_cur_obs_var = generative_cur_observation_distribution_var, 
    #     #     observation_samples = reparamed_observation_samples,
    #     #     D = 4
    #     # )

    #     # Calculate the negative log likelihood of the observation
    #     avg_log_likelihood = self.diag_cov_gaussian_log_li(
    #         predicted_cur_obs_mean = generative_observation_mean, 
    #         predicted_cur_obs_var = generative_observation_var, 
    #         observation_samples = reparamed_observation_samples,
    #         D = 4
    #     )

    #     # Compute the final loss
    #     loss = kl_divergence - avg_log_likelihood

    #     # print(f"\nVFE - kl_divergence: {kl_divergence}")
    #     # print(f"VFE - avg_log_likelihood: {avg_log_likelihood}")
    #     # print(f"VFE - loss: {loss}\n")

    #     # global losses
    #     # global kl_divs
    #     # global avg_log_likelis

    #     # losses.append(loss.cpu().detach().numpy())
    #     # kl_divs.append(kl_divergence.cpu().detach().numpy())
    #     # avg_log_likelis.append(avg_log_likelihood.cpu().detach().numpy())

    #     # breakpoint()

    #     return loss, kl_divergence, avg_log_likelihood

        # def train(self):

    #     filename = f"Deep_AIF_MDP_Cart_Pole_v1"
    #     figure_file = f"plots/{filename}.png"

    #     msg = "Environment is: {}\nTraining started at {}".format(self.env.unwrapped.spec.id, datetime.datetime.now())
    #     print(msg)
    #     print(f"\n----------------- Deep Active Inference Tree Search -----------------\n")
    #     if self.keep_log:
    #         self.record.write(msg+"\n")

    #     results = []

    #     if torch.cuda.is_available():
    #         print("CUDA is available")
    #     else:
    #         print("CUDA is NOT available")
    #     print(f"self.device: {self.device}")
    #     print(f"Playing {self.n_episodes} episodes")

    #     # Define the standard deviation of the Gaussian noise
    #     noise_std = 0.05

    #     for ith_episode in range(self.n_episodes):

    #         total_reward = 0
    #         obs, _ = self.env.reset()
    #         noisy_obs = obs + noise_std * np.random.randn(*obs.shape)

    #         terminated = False
    #         truncated = False
    #         done = terminated or truncated

    #         reward = 0

    #         num_iters = 0

    #         while not done:

    #             # Assign the initial hidden state belief's sufficient statistics:
    #             state_belief_mean = torch.tensor([noisy_obs[0], noisy_obs[1], noisy_obs[2], noisy_obs[3]], dtype=torch.float32, device=self.device)
    #             state_belief_var = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32, device=self.device) # it needs to use the learned value not this !!!!!!!

    #             # ACTIVE INFERENCE TREE SEARCH: Do Active Inference Tree Search:
    #             root_node = self.active_inferece_tree_search(
    #                 state_belief_mean, 
    #                 state_belief_var,
    #                 self.delta, 
    #                 self.epsilon
    #             )

    #             # self.env.render()

    #             # cast the noisy obs to tensor
    #             noisy_obs = torch.tensor(noisy_obs, dtype = torch.float32, device = self.device)

    #             # ACTIVE INFERENCE TREE SEARCH: Select the best action based on the updated tree
    #             # action = self.select_action_AcT(root_node)
    #             action = self.select_action_myopic_AcT(root_node)
    #             # action = self.env.action_space.sample()
    #             action = torch.tensor([action], dtype = torch.int64, device = self.device) # cast action back to tensor

    #             # print(f"\nchosen action: {action}\n")
    #             # draw_tree(root_node)
    #             # breakpoint()

    #             # Push the noisy tuple to self.memory
    #             self.memory.push(noisy_obs, action, reward, terminated, truncated)

    #             # ACTIVE INFERENCE TREE SEARCH: Execute the selected action and transition to a new state - AcT:
    #             # obs, reward, terminated, truncated, _  = self.env.step(action)
    #             obs, reward, terminated, truncated, _  = self.env.step(action[0].item())

    #             # update observation
    #             noisy_obs = obs + noise_std * np.random.randn(*obs.shape)

    #             # increment the per-episode reward:
    #             total_reward += reward

    #             # learn
    #             free_energy_loss, kl_divergence, avg_log_likelihood = self.learn()

    #             # print(f"\nhidden state: {obs}")
    #             # print(f"noisy_obs: {noisy_obs}")
    #             # print(f"action: {action}")
    #             # print(f"total_reward: {total_reward}\n")

    #             num_iters += 1

    #             if terminated or truncated:

    #                 done = True

    #                 noisy_obs = torch.tensor(noisy_obs, dtype = torch.float32, device = self.device)

    #                 self.memory.push(noisy_obs, -99, -99, terminated, truncated)

    #         results.append(total_reward)

    #         # Print and keep a (.txt) record of stuff
    #         if ith_episode > 0 and ith_episode % self.print_timer == 0:
    #             avg_reward = np.mean(results)
    #             last_x = np.mean(results[-self.print_timer:])
    #             msg = "Episodes: {:4d}, avg score: {:3.2f}, over last {:d}: {:3.2f}".format(ith_episode, avg_reward, self.print_timer, last_x)
    #             print(msg)

    #             if self.keep_log:
    #                 self.record.write(msg+"\n")

    #                 if ith_episode % self.log_save_timer == 0:
    #                     self.record.close()
    #                     self.record = open(self.log_path, "a")

    #         # If enabled, save the results and the network (state_dict)
    #         if self.save_results and ith_episode > 0 and ith_episode % self.results_save_timer == 0:
    #             np.savez("results/intermediary/intermediary_results{}_{:d}".format(self.run_id, ith_episode), np.array(results))
    #         if self.save_network and ith_episode > 0 and ith_episode % self.network_save_timer == 0:
    #             torch.save(self.generative_transition_net.state_dict(), "networks/intermediary/intermediary_transnet{}_{:d}.pth".format(self.run_id, ith_episode))
    #             torch.save(self.variational_transition_net.state_dict(), "networks/intermediary/intermediary_vartransnet{}_{:d}.pth".format(self.run_id, ith_episode))

    #             torch.save(self.generative_observation_net.state_dict(), "networks/intermediary/intermediary_obsnet{}_{:d}.pth".format(self.run_id, ith_episode))

    #     self.env.close()

    #     # If enabled, save the results and the network (state_dict)
    #     if self.save_results:
    #         np.savez("results/intermediary/intermediary_results{}_end".format(self.run_id), np.array(results))
    #         np.savez(self.results_path, np.array(results))
    #     if self.save_network:
    #         torch.save(self.generative_transition_net.state_dict(), "networks/intermediary/intermediary_transnet{}_end.pth".format(self.run_id))

    #         torch.save(self.generative_observation_net.state_dict(), "networks/intermediary/intermediary_obsnet{}_end.pth".format(self.run_id))

    #         torch.save(self.variational_transition_net.state_dict(), "networks/intermediary/intermediary_vartransnet{}_end.pth".format(self.run_id))

    #         torch.save(self.generative_transition_net.state_dict(), self.network_save_path.format("trans"))
    #         torch.save(self.variational_transition_net.state_dict(), self.network_save_path.format("vartrans"))

    #         torch.save(self.generative_observation_net.state_dict(), self.network_save_path.format("obs"))

    #     # Print and keep a (.txt) record of stuff
    #     msg = "Training finished at {}".format(datetime.datetime.now())
    #     print(msg)
    #     if self.keep_log:
    #         self.record.write(msg)
    #         self.record.close()

    #     x = [i + 1 for i in range(self.n_episodes)]
    #     plot_learning_curve(x, results, figure_file)

    # WORKING VERSION 1
    # def train(self):

    #     filename = f"Deep_AIF_MDP_Cart_Pole_v1"
    #     figure_file = f"plots/{filename}.png"

    #     msg = "Environment is: {}\nTraining started at {}".format(self.env.unwrapped.spec.id, datetime.datetime.now())
    #     print(msg)
    #     print(f"\n----------------- Deep Active Inference Tree Search -----------------\n")
    #     if self.keep_log:
    #         self.record.write(msg+"\n")

    #     results = []

    #     if torch.cuda.is_available():
    #         print("CUDA is available")
    #     else:
    #         print("CUDA is NOT available")
    #     print(f"self.device: {self.device}")
    #     print(f"Playing {self.n_episodes} episodes")

    #     # Define the standard deviation of the Gaussian noise
    #     noise_std = 0.05

    #     for ith_episode in range(self.n_episodes):

    #         total_reward = 0
    #         obs, _ = self.env.reset()
    #         noisy_obs = obs + noise_std * np.random.randn(*obs.shape)

    #         # initial state belief mean
    #         # state_belief_mean = torch.tensor([noisy_obs[0], noisy_obs[1], noisy_obs[2], noisy_obs[3]], dtype=torch.float32, device=self.device)
    #         state_belief_mean = torch.tensor(noisy_obs, dtype = torch.float32, device = self.device)

    #         # initial state belief variance 
    #         state_belief_var = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32, device=self.device)

    #         terminated = False
    #         truncated = False
    #         done = terminated or truncated

    #         reward = 0

    #         num_iters = 0

    #         while not done:

    #             # # Assign the initial hidden state belief's sufficient statistics:
    #             # state_belief_mean = torch.tensor([noisy_obs[0], noisy_obs[1], noisy_obs[2], noisy_obs[3]], dtype=torch.float32, device=self.device)
    #             # state_belief_var = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32, device=self.device) # it needs to use the learned value not this !!!!!!!

    #             # # ACTIVE INFERENCE TREE SEARCH: Do Active Inference Tree Search:
    #             # root_node = self.active_inferece_tree_search(
    #             #     state_belief_mean, 
    #             #     state_belief_var,
    #             #     self.delta, 
    #             #     self.epsilon
    #             # )

    #             # ACTIVE INFERENCE TREE SEARCH: Do Active Inference Tree Search:

    #             AcT_selected_actions = []

    #             for n in range(5):

    #                 root_node = self.active_inferece_tree_search(
    #                     state_belief_mean, 
    #                     state_belief_var,
    #                     self.delta, 
    #                     self.epsilon
    #                 )

    #                 # self.env.render()

    #                 # ACTIVE INFERENCE TREE SEARCH: Select the best action based on the updated tree
    #                 # action = self.select_action_AcT(root_node)
    #                 action = self.select_action_myopic_AcT(root_node) # need to implement multiple samples
    #                 # action = torch.tensor([action], dtype = torch.int64, device = self.device) # cast action back to tensor

    #                 AcT_selected_actions.append(action.item())

    #             action = stat.mode(AcT_selected_actions)
    #             action = torch.tensor([action], dtype = torch.int64, device = self.device) # cast action back to tensor

    #             # print(f"\n\nAcT_selected_actions: {AcT_selected_actions}")
    #             # print(f"selected action: {action}\n\n")

    #             # cast the noisy obs to tensor
    #             noisy_obs = torch.tensor(noisy_obs, dtype = torch.float32, device = self.device)

    #             # RANDOM ACTION SELECTION 
    #             # action = self.env.action_space.sample()

    #             # Push the noisy tuple to self.memory
    #             self.memory.push(noisy_obs, action, reward, terminated, truncated)

    #             # Execute the selected action and transition to a new state:
    #             # obs, reward, terminated, truncated, _  = self.env.step(action)
    #             obs, reward, terminated, truncated, _  = self.env.step(action[0].item())

    #             # increment the per-episode reward:
    #             total_reward += reward

    #             # learn, and get the new state belief variance 
    #             free_energy_loss, kl_divergence, avg_log_likelihood, state_belief_var = self.learn()

    #             # update observation
    #             noisy_obs = obs + noise_std * np.random.randn(*obs.shape)
    #             # noisy_obs = torch.tensor(noisy_obs, dtype = torch.float32, device = self.device)
    #             # state_belief_mean = torch.tensor([noisy_obs[0], noisy_obs[1], noisy_obs[2], noisy_obs[3]], dtype=torch.float32, device=self.device)
    #             # state_belief_var is the new variance 

    #             # print(f"\nstate_belief_mean:\n{state_belief_mean}")
    #             # print(f"state_belief_var:\n{state_belief_var}\n")
    #             # breakpoint()

    #             num_iters += 1

    #             if terminated or truncated:

    #                 done = True

    #                 noisy_obs = torch.tensor(noisy_obs, dtype = torch.float32, device = self.device)

    #                 self.memory.push(noisy_obs, -99, -99, terminated, truncated)

    #         results.append(total_reward)

    #         # Print and keep a (.txt) record of stuff
    #         if ith_episode > 0 and ith_episode % self.print_timer == 0:
    #             avg_reward = np.mean(results)
    #             last_x = np.mean(results[-self.print_timer:])
    #             msg = "Episodes: {:4d}, avg score: {:3.2f}, over last {:d}: {:3.2f}".format(ith_episode, avg_reward, self.print_timer, last_x)
    #             print(msg)

    #             if self.keep_log:
    #                 self.record.write(msg+"\n")

    #                 if ith_episode % self.log_save_timer == 0:
    #                     self.record.close()
    #                     self.record = open(self.log_path, "a")

    #         # If enabled, save the results and the network (state_dict)
    #         if self.save_results and ith_episode > 0 and ith_episode % self.results_save_timer == 0:
    #             np.savez("results/intermediary/intermediary_results{}_{:d}".format(self.run_id, ith_episode), np.array(results))
    #         if self.save_network and ith_episode > 0 and ith_episode % self.network_save_timer == 0:
    #             torch.save(self.generative_transition_net.state_dict(), "networks/intermediary/intermediary_transnet{}_{:d}.pth".format(self.run_id, ith_episode))
    #             torch.save(self.variational_transition_net.state_dict(), "networks/intermediary/intermediary_vartransnet{}_{:d}.pth".format(self.run_id, ith_episode))

    #             torch.save(self.generative_observation_net.state_dict(), "networks/intermediary/intermediary_obsnet{}_{:d}.pth".format(self.run_id, ith_episode))

    #     self.env.close()

    #     # If enabled, save the results and the network (state_dict)
    #     if self.save_results:
    #         np.savez("results/intermediary/intermediary_results{}_end".format(self.run_id), np.array(results))
    #         np.savez(self.results_path, np.array(results))
    #     if self.save_network:
    #         torch.save(self.generative_transition_net.state_dict(), "networks/intermediary/intermediary_transnet{}_end.pth".format(self.run_id))

    #         torch.save(self.generative_observation_net.state_dict(), "networks/intermediary/intermediary_obsnet{}_end.pth".format(self.run_id))

    #         torch.save(self.variational_transition_net.state_dict(), "networks/intermediary/intermediary_vartransnet{}_end.pth".format(self.run_id))

    #         torch.save(self.generative_transition_net.state_dict(), self.network_save_path.format("trans"))
    #         torch.save(self.variational_transition_net.state_dict(), self.network_save_path.format("vartrans"))

    #         torch.save(self.generative_observation_net.state_dict(), self.network_save_path.format("obs"))

    #     # Print and keep a (.txt) record of stuff
    #     msg = "Training finished at {}".format(datetime.datetime.now())
    #     print(msg)
    #     if self.keep_log:
    #         self.record.write(msg)
    #         self.record.close()

    #     x = [i + 1 for i in range(self.n_episodes)]
    #     plot_learning_curve(x, results, figure_file)




    # WORKING VERSION 2
    # def train(self):

    #     filename = f"Deep_AIF_MDP_Cart_Pole_v1"
    #     figure_file = f"plots/{filename}.png"

    #     msg = "Environment is: {}\nTraining started at {}".format(self.env.unwrapped.spec.id, datetime.datetime.now())
    #     print(msg)
    #     print(f"\n----------------- Deep Active Inference Tree Search -----------------\n")
    #     if self.keep_log:
    #         self.record.write(msg+"\n")

    #     results = []

    #     if torch.cuda.is_available():
    #         print("CUDA is available")
    #     else:
    #         print("CUDA is NOT available")
    #     print(f"self.device: {self.device}")
    #     print(f"Playing {self.n_episodes} episodes")

    #     # Define the standard deviation of the Gaussian noise
    #     noise_std = 0.05

    #     for ith_episode in range(self.n_episodes):

    #         total_reward = 0
    #         obs, _ = self.env.reset()
    #         noisy_obs = obs + noise_std * np.random.randn(*obs.shape)

    #         # initial state belief mean
    #         state_belief_mean = torch.tensor(noisy_obs, dtype = torch.float32, device = self.device)

    #         # initial state belief variance 
    #         state_belief_var = torch.tensor([0.01, 0.01, 0.01, 0.01], dtype=torch.float32, device=self.device)

    #         terminated = False
    #         truncated = False
    #         done = terminated or truncated

    #         reward = 0

    #         num_iters = 0

    #         while not done:

    #             # # Assign the initial hidden state belief's sufficient statistics:
    #             # state_belief_mean = torch.tensor([noisy_obs[0], noisy_obs[1], noisy_obs[2], noisy_obs[3]], dtype=torch.float32, device=self.device)
    #             # state_belief_var = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32, device=self.device) # it needs to use the learned value not this !!!!!!!

    #             # # ACTIVE INFERENCE TREE SEARCH: Do Active Inference Tree Search:
    #             # root_node = self.active_inferece_tree_search(
    #             #     state_belief_mean, 
    #             #     state_belief_var,
    #             #     self.delta, 
    #             #     self.epsilon
    #             # )




    #             # ACTIVE INFERENCE TREE SEARCH: Do Active Inference Tree Search:
    #             AcT_selected_actions = []

    #             for n in range(5):

    #                 root_node = self.active_inferece_tree_search(
    #                     state_belief_mean, 
    #                     state_belief_var,
    #                     self.delta, 
    #                     self.epsilon
    #                 )

    #                 # self.env.render()

    #                 # ACTIVE INFERENCE TREE SEARCH: Select the best action based on the updated tree
    #                 # action = self.select_action_AcT(root_node)
    #                 action = self.select_action_myopic_AcT(root_node) # need to implement multiple samples
    #                 # action = torch.tensor([action], dtype = torch.int64, device = self.device) # cast action back to tensor

    #                 AcT_selected_actions.append(action.item())

    #             action = stat.mode(AcT_selected_actions)
    #             action = torch.tensor([action], dtype = torch.int64, device = self.device) # cast action back to tensor




    #             # print(f"\n\nAcT_selected_actions: {AcT_selected_actions}")
    #             # print(f"selected action: {action}\n\n")

    #             # cast the noisy obs to tensor
    #             # state_belief_mean = torch.tensor(noisy_obs, dtype = torch.float32, device = self.device)

    #             # # RANDOM ACTION SELECTION 
    #             # action = self.env.action_space.sample()
    #             # action = torch.tensor([action], dtype = torch.int64, device = self.device) # cast action back to tensor

    #             # # Push the noisy tuple to self.memory
    #             # self.memory.push(state_belief_mean, action, reward, terminated, truncated)

    #             # Execute the selected action and transition to a new state:
    #             # obs, reward, terminated, truncated, _  = self.env.step(action)
    #             obs, reward, terminated, truncated, _  = self.env.step(action[0].item())

    #             state_belief_mean = obs + noise_std * np.random.randn(*obs.shape)
    #             state_belief_mean = torch.tensor(state_belief_mean, dtype = torch.float32, device = self.device) 

    #             # Push the noisy tuple to self.memory
    #             self.memory.push(state_belief_mean, action, reward, terminated, truncated)

    #             # increment the per-episode reward:
    #             total_reward += reward

    #             # learn, and get the new state belief variance 
    #             free_energy_loss, kl_divergence, avg_log_likelihood, state_belief_var, state_belief_mean = self.learn() # this is probably not the variance that corresponds to the "state_belief_mean"  - get both of these from the replay buffer? YES

    #             # print(f"\n\nstate_belief_var: {state_belief_var}")
    #             # print(f"state_belief_mean: {state_belief_mean}\n\n")

    #             # update observation
    #             # state_belief_mean = obs + noise_std * np.random.randn(*obs.shape)
    #             # state_belief_mean = state_belief_mean.cpu() + noise_std * np.random.randn(*obs.shape)
    #             state_belief_mean = state_belief_mean.cpu().detach().numpy() + noise_std * np.random.randn(*obs.shape)
    #             state_belief_mean = torch.tensor(state_belief_mean, dtype = torch.float32, device = self.device)
    #             # state_belief_mean = torch.tensor([noisy_obs[0], noisy_obs[1], noisy_obs[2], noisy_obs[3]], dtype=torch.float32, device=self.device)
    #             # state_belief_var is the new variance 

    #             num_iters += 1

    #             if terminated or truncated:

    #                 done = True

    #                 # state_belief_mean = torch.tensor(state_belief_mean, dtype = torch.float32, device = self.device)

    #                 self.memory.push(state_belief_mean, -99, -99, terminated, truncated) # may not be best to append -99

    #         results.append(total_reward)

    #         # Print and keep a (.txt) record of stuff
    #         if ith_episode > 0 and ith_episode % self.print_timer == 0:
    #             avg_reward = np.mean(results)
    #             last_x = np.mean(results[-self.print_timer:])
    #             msg = "Episodes: {:4d}, avg score: {:3.2f}, over last {:d}: {:3.2f}".format(ith_episode, avg_reward, self.print_timer, last_x)
    #             print(msg)

    #             if self.keep_log:
    #                 self.record.write(msg+"\n")

    #                 if ith_episode % self.log_save_timer == 0:
    #                     self.record.close()
    #                     self.record = open(self.log_path, "a")

    #         # If enabled, save the results and the network (state_dict)
    #         if self.save_results and ith_episode > 0 and ith_episode % self.results_save_timer == 0:
    #             np.savez("results/intermediary/intermediary_results{}_{:d}".format(self.run_id, ith_episode), np.array(results))
    #         if self.save_network and ith_episode > 0 and ith_episode % self.network_save_timer == 0:
    #             torch.save(self.generative_transition_net.state_dict(), "networks/intermediary/intermediary_transnet{}_{:d}.pth".format(self.run_id, ith_episode))
    #             torch.save(self.variational_transition_net.state_dict(), "networks/intermediary/intermediary_vartransnet{}_{:d}.pth".format(self.run_id, ith_episode))

    #             torch.save(self.generative_observation_net.state_dict(), "networks/intermediary/intermediary_obsnet{}_{:d}.pth".format(self.run_id, ith_episode))

    #     self.env.close()

    #     # If enabled, save the results and the network (state_dict)
    #     if self.save_results:
    #         np.savez("results/intermediary/intermediary_results{}_end".format(self.run_id), np.array(results))
    #         np.savez(self.results_path, np.array(results))
    #     if self.save_network:
    #         torch.save(self.generative_transition_net.state_dict(), "networks/intermediary/intermediary_transnet{}_end.pth".format(self.run_id))

    #         torch.save(self.generative_observation_net.state_dict(), "networks/intermediary/intermediary_obsnet{}_end.pth".format(self.run_id))

    #         torch.save(self.variational_transition_net.state_dict(), "networks/intermediary/intermediary_vartransnet{}_end.pth".format(self.run_id))

    #         torch.save(self.generative_transition_net.state_dict(), self.network_save_path.format("trans"))
    #         torch.save(self.variational_transition_net.state_dict(), self.network_save_path.format("vartrans"))

    #         torch.save(self.generative_observation_net.state_dict(), self.network_save_path.format("obs"))

    #     # Print and keep a (.txt) record of stuff
    #     msg = "Training finished at {}".format(datetime.datetime.now())
    #     print(msg)
    #     if self.keep_log:
    #         self.record.write(msg)
    #         self.record.close()

    #     x = [i + 1 for i in range(self.n_episodes)]
    #     plot_learning_curve(x, results, figure_file, "AcT Action Selection")