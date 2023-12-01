# def calculate_approximate_EFE(self, mean_state_theta, var_state_theta, mean_obs_xi, var_obs_xi):
    #     with torch.no_grad():
    #         # 1. Calculate the KL divergence between Q(s_t | pi) and the state prior P(s_t | C):
    #         state_divergence = self.kl_divergence_diag_cov_gaussian(
    #             mean_state_theta, var_state_theta,
    #             mean_state_prior, var_state_prior
    #         )

    #         # 2. Calculate the expected entropy of the observation likelihood, under Q(s_t | pi):
    #         num_samples = 100  # Number of samples to estimate the expectation
    #         entropy_sum = 0.0

    #         for _ in range(num_samples):
                
    #             # Sample s_t from Q(s_t | pi)
    #             sampled_state = torch.normal(mean_state_theta, torch.sqrt(var_state_theta))

    #             # Calculate predicted observation mean and variance using generative observation network
    #             predicted_obs_mean, predicted_obs_var = generative_observation_net(sampled_state)

    #             # Calculate entropy of p_xi(o_t | s_t) using diagonal_gaussian_entropy
    #             entropy = diagonal_gaussian_entropy(predicted_obs_var, D)  # D is the dimensionality of observation

    #             entropy_sum += entropy

    #         # Average the calculated entropies
    #         expected_entropy = entropy_sum / num_samples

    #         # Calculate the expected free energy
    #         expected_free_energy = state_divergence + expected_entropy

    #         return expected_free_energy

    # def calculate_approximate_EFE(self, mean_state_theta, var_state_theta, mean_obs_xi, var_obs_xi):
    #     with torch.no_grad():
    #         # 1. Calculate the KL divergence between Q(s_t | pi) and the state prior P(s_t | C):
    #         state_divergence = self.kl_divergence_diag_cov_gaussian(
    #             mean_state_theta, var_state_theta,
    #             mean_state_prior, var_state_prior
    #         )

    #         # 2. Calculate the expected entropy of the observation likelihood, under Q(s_t | pi)
    #         # with  using Monte Carlo sampling:
    #         num_samples = 100  # Number of samples to estimate the expectation
    #         entropy_sum = 0.0

    #         for _ in range(num_samples):
                
    #             # Sample s_t from Q(s_t | pi)
    #             sampled_state = torch.normal(mean_state_theta, torch.sqrt(var_state_theta))

    #             # Calculate predicted observation mean and variance using generative observation network
    #             predicted_obs_mean, predicted_obs_var = generative_observation_net(sampled_state)

    #             # Calculate entropy of p_xi(o_t | s_t) using diagonal_gaussian_entropy
    #             entropy = diagonal_gaussian_entropy(predicted_obs_var, D)  # D is the dimensionality of observation

    #             entropy_sum += entropy

    #         # Average the calculated entropies
    #         expected_entropy = entropy_sum / num_samples

    #         # Calculate the expected free energy
    #         expected_free_energy = state_divergence + expected_entropy

    #         return expected_free_energy

    # def calculate_approximate_EFE(self, mean_state_theta, var_state_theta, mean_obs_xi, var_obs_xi):
    #     with torch.no_grad():
    #         # 1. Calculate the KL divergence between Q(s_t | pi) and the state prior P(s_t | C):
    #         state_divergence = self.kl_divergence_diag_cov_gaussian(
    #             mean_state_theta, var_state_theta,
    #             mean_state_prior, var_state_prior
    #         )

    #         # 2. calculate the unconditional entropy of the observation model:
    #         entropy = self.diagonal_gaussian_entropy(var_obs_xi, 4) # D = 4 for CartPole-v1

    #         # 2. Calculate the expected entropy of the observation likelihood, under Q(s_t | pi):
    #         # num_samples = 100  # Number of samples to estimate the expectation
    #         # entropy_sum = 0.0

    #         # # Calculate predicted observation mean and variance using generative observation network
    #         # predicted_obs_mean, predicted_obs_var = generative_observation_net(mean_state_theta)

    #         # for _ in range(num_samples):
    #         #     # Sample s_t from Q(s_t | pi)
    #         #     sampled_state = torch.normal(mean_state_theta, torch.sqrt(var_state_theta))

    #         #     # Calculate entropy of p_xi(o_t | s_t) using diagonal_gaussian_entropy
    #         #     entropy = diagonal_gaussian_entropy(predicted_obs_var, D)  # D is the dimensionality of observation

    #         #     entropy_sum += entropy

    #         # # Average the calculated entropies
    #         # expected_entropy = entropy_sum / num_samples

    #         # Calculate the expected free energy
    #         # expected_free_energy = state_divergence + expected_entropy
    #         expected_free_energy = state_divergence + entropy

    #     return expected_free_energy
















    # def learn(self):

    #     # print("\tINSIDE learn")

    #     # If there are not enough transitions stored in memory, return:
    #     if self.memory.push_count - self.max_n_indices*2 < self.batch_size:
    #         return

    #     # After every freeze_period time steps, update the target network:
    #     if self.freeze_cntr % self.freeze_period == 0:
    #         self.target_net.load_state_dict(self.efe_value_net.state_dict())
    #     self.freeze_cntr += 1

    #     # Retrieve transition data in mini batches:
    #     (
    #         obs_batch_t0, obs_batch_t1, obs_batch_t2, action_batch_t0,
    #         action_batch_t1, reward_batch_t1, terminated_batch_t2, truncated_batch_t2,
    #         pred_error_batch_t0t1
    #     ) = self.get_mini_batches()

    #     # Compute the value network loss:
    #     efe_value_net_loss = self.compute_EFE_efe_value_net_loss(
    #         obs_batch_t1, obs_batch_t2,
    #         action_batch_t1, reward_batch_t1,
    #         terminated_batch_t2, truncated_batch_t2, pred_error_batch_t0t1
    #     )

    #     # Compute the variational free energy:
    #     VFE = self.compute_VFE(obs_batch_t1, pred_error_batch_t0t1)

    #     # Reset the gradients:
    #     self.generative_transition_net.optimizer.zero_grad()
    #     self.policy_net.optimizer.zero_grad()
    #     self.efe_value_net.optimizer.zero_grad()

    #     self.generative_observation_net.optimizer.zero_grad()

    #     # Compute the gradients:
    #     VFE.backward()
    #     efe_value_net_loss.backward()

    #     # Perform gradient descent:
    #     self.generative_transition_net.optimizer.step()
    #     self.policy_net.optimizer.step()
    #     self.efe_value_net.optimizer.step()

    #     self.generative_observation_net.optimizer.step()








    # ##############################################################################################################################
    # # # OG Himst and Lanillos main loop 
    # # action = self.select_action_dAIF(obs)
    
    # # # self.memory.push(obs, action, reward, done)
    # # self.memory.push(obs, action, reward, terminated, truncated)
    
    # # # obs, reward, done, _ = self.env.step(action[0].item())
    # # obs, reward, terminated, truncated, _  = self.env.step(action[0].item())

    # # obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
    # # total_reward += reward
    
    # # self.learn()

    # action = self.select_action_dAIF(obs)
    
    # # self.memory.push(obs, action, reward, done)
    # self.memory.push(obs, action, reward, terminated, truncated)
    
    # # obs, reward, done, _ = self.env.step(action[0].item())
    # obs, reward, terminated, truncated, _  = self.env.step(action[0].item())

    # # Add noise to the observation
    # noise = np.random.normal(0, noise_std, size=obs.shape)  # Generate noise
    # obs_with_noise = obs + noise  # Add noise to observation

    # obs = torch.tensor(obs_with_noise, dtype=torch.float32, device=self.device)
    # total_reward += reward
    
    # self.learn()

    ##############################################################################################################################







    # def compute_VFE(self, obs_batch_t1, pred_error_batch_t0t1):

    #     # print("\tINSIDE compute_VFE")

    #     # Determine the action distribution for time t1:
    #     # policy_batch_t1 = self.policy_net(obs_batch_t1)

    #     # print(f"\n\ncompute_VFE - obs_batch_t1: {obs_batch_t1}\n\n")

    #     # normalised_policy_batch_means_t1, policy_batch_stdvs_t1 = self.policy_net(obs_batch_t1)
    #     normalised_policy_batch_means_t1 = self.policy_net(obs_batch_t1)

    #     # Determine the EFEs for time t1:
    #     EFEs_batch_t1 = self.efe_value_net(obs_batch_t1).detach()

    #     # Take a gamma-weighted Boltzmann distribution over the EFEs:
    #     boltzmann_EFEs_batch_t1 = torch.softmax(-self.gamma * EFEs_batch_t1, dim=1).clamp(min=1e-9, max=1-1e-9).to(self.device) # added to device

    #     # Weigh them according to the action distribution:
    #     # energy_batch = -(policy_batch_t1 * torch.log(boltzmann_EFEs_batch_t1)).sum(-1).view(self.batch_size, 1).to(self.device) # added to device $ OG VERSION
    #     energy_batch = -(normalised_policy_batch_means_t1 * torch.log(boltzmann_EFEs_batch_t1)).sum(-1).view(self.batch_size, 1).to(self.device) # added to device

    #     # Determine the entropy of the action distribution
    #     # entropy_batch = -(policy_batch_t1 * torch.log(policy_batch_t1)).sum(-1).view(self.batch_size, 1).to(self.device) # added to device # OG VERSION
    #     entropy_batch = -(normalised_policy_batch_means_t1 * torch.log(normalised_policy_batch_means_t1)).sum(-1).view(self.batch_size, 1).to(self.device) # added to device

    #     # Determine the VFE, then take the mean over all batch samples:
    #     VFE_batch = pred_error_batch_t0t1 + (energy_batch - entropy_batch)
    #     VFE = torch.mean(VFE_batch).to(self.device) # added to device

    #     return VFE

















        # def compute_EFE_efe_value_net_loss(
    #     self, obs_batch_t1, obs_batch_t2, action_batch_t1, reward_batch_t1,
    #     terminated_batch_t2, truncated_batch_t2, pred_error_batch_t0t1
    # ):
    #     # print("\tINSIDE compute_EFE_efe_value_net_loss")

    #     with torch.no_grad():

    #         # Determine the action distribution for time t2:
    #         # policy_batch_t2 = self.policy_net(obs_batch_t2) # better be normalised # OG VERSION

    #         # print(f"\n\ncompute_EFE_efe_value_net_loss - obs_batch_t2: {obs_batch_t2}\n\n")

    #         # normalised_action_means_t2, action_stdvs_t2 = self.policy_net(obs_batch_t2)
    #         normalised_action_means_t2 = self.policy_net(obs_batch_t2)

    #         # Determine the target EFEs for time t2:
    #         target_EFEs_batch_t2 = self.target_net(obs_batch_t2)

    #         # # OG VERSION
    #         # # Weigh the target EFEs according to the action distribution:
    #         # weighted_EFE_value_targets = ((1-(terminated_batch_t2 | truncated_batch_t2)) * policy_batch_t2 *
    #         #                     target_EFEs_batch_t2).sum(-1).unsqueeze(1) # OG VERSION

    #         # # Weigh the target EFEs according to the action distribution:
    #         # weighted_EFE_value_targets = ((1-(terminated_batch_t2 | truncated_batch_t2)) * normalised_action_means_t2 *
    #         #                     target_EFEs_batch_t2).sum(-1).unsqueeze(1)

    #         ###############################################################################
    #         # Convert boolean tensors to float tensors
    #         weighting_factor = (1.0 - (terminated_batch_t2 | truncated_batch_t2)).float()

    #         # Weigh the target EFEs according to the action distribution:
    #         weighted_EFE_value_targets = (weighting_factor * normalised_action_means_t2 * target_EFEs_batch_t2).sum(-1).unsqueeze(1)
    #         ###############################################################################
 

    #         # # use this if policy_batch_t2 not normalised
    #         # normalized_policy_batch_t2 = policy_batch_t2 / policy_batch_t2.sum(-1, keepdim=True)
    #         # weighted_EFE_value_targets = ((1 - (terminated_batch_t2 | truncated_batch_t2)) * normalized_policy_batch_t2 *
    #         #                  target_EFEs_batch_t2).sum(-1, keepdim=True)

    #         # Determine the batch of bootstrapped estimates of the EFEs:
    #         EFE_estimate_batch = -reward_batch_t1 + pred_error_batch_t0t1 + self.Beta * weighted_EFE_value_targets

    #     # Determine the EFE at time t1 according to the value network:
    #     output = self.efe_value_net(obs_batch_t1)
    #     # print(f"\noutput.size(): {output.size()}")
    #     EFE_batch_t1 = output.gather(1, action_batch_t1)
    #     # print(f"EFE_batch_t1.size(): {EFE_batch_t1.size()}\n")

    #     # print(f"\nEFE_batch_t1.size(): {EFE_batch_t1.size()}")
    #     # print(f"EFE_estimate_batch.size(): {EFE_estimate_batch.size()}")

    #     # Determine the MSE loss between the EFE estimates and the value network output:
    #     EFE_efe_value_net_loss = F.mse_loss(EFE_estimate_batch, EFE_batch_t1)

    #     # print(f"\ncompute_value_net_loss - EFE_batch_t1.size(): {EFE_batch_t1.size()}")
    #     # print(f"compute_value_net_loss - EFE_estimate_batch.size(): {EFE_estimate_batch.size()}")
    #     # print(f"compute_value_net_loss - EFE_efe_value_net_loss: {EFE_efe_value_net_loss}\n")

    #     return EFE_efe_value_net_loss # the MSE between a scaler and a vector ??????????? No because each have a batch dimension













    # # def expand_AcT(self, node, B):
    # def expand_AcT(self, node):
    #     '''
    #     Current Problems:
    #         * always computes: a_prime = 0
    #         * unused_actions always only = 0 
    #     '''

    #     # print("\tINSIDE expand_AcT")

    #     # perform an unused action:
    #     unused_actions = self.get_actions_AcT(node) # always only 0

    #     # a_prime = torch.tensor(random.choice(unused_actions)).float().unsqueeze(0).to(self.device) # always left action: "0"
    #     a_prime = torch.tensor(random.choice(unused_actions)).float().to(self.device) # CHANGED
    #     a_prime = torch.unsqueeze(a_prime, 0) # ADDED

    #     node.used_actions.append(a_prime)
    #     node.policy[node.depth] = a_prime.item() # add this action to the "policy so far"

    #     # At time t0 predict the state at time t1:
    #     state_transition_t0t1 = torch.cat((node.state_belief, a_prime), dim = 0) # For MDP, obs_t0 = node.state
    #     pred_state_t0t1 = self.generative_transition_net(state_transition_t0t1).detach() # tensor type

    #     # instantiate a child node as a consequence of performing action a_prime
    #     # child_node = Node(future_time_horizon = self.planning_horizon) # probably wrong future time horizon
    #     child_node = Node(future_time_horizon = self.planning_horizon - (node.depth + 1)) # correct time horizon?
    #     # child_node.visit_count += 1 # Done in path integration stage only 
    #     child_node.parent = node
    #     child_node.depth = node.depth + 1
    #     child_node.state_belief = pred_state_t0t1 # x_tau = the "predicted state" in the MDP case (it's not a "belief"). 

    #     child_node.action_at_tau_minus_one = a_prime # the action that led to the visitation of the present node
    #     child_node.policy = copy.deepcopy(node.policy) # copy the policy generated from the history
    #     # child_node.policy[child_node.depth] = a_prime # add the current node's generating action to the history # No, this is wrong, I think

    #     ##########################

    #     raw_efe = self.calculate_approximate_EFE(mean_state_theta, var_state_theta, mean_obs_xi, var_obs_xi)

    #     ##########################

    #     # raw_efe = self.efe_value_net(torch.tensor(child_node.policy)).detach() # The EFE computed at time tau: G(*, v_tau).
    #     # raw_efe = self.efe_value_net(pred_state_t0t1)[int(a_prime.item())] # The EFE computed at time tau: G(*, v_tau) of action a_prime.
    #     # raw_efe = self.efe_value_net(pred_state_t0t1) # The EFE computed at time tau: G(*, v_tau) of action a_prime.

    #     raw_efe_action_taken = raw_efe[int(a_prime.item())]

    #     child_node.raw_efe = raw_efe_action_taken # store the raw efe as intermediate stage in comuting predictive efe

    #     # finally, add the child node to the node's children
    #     node.children.append(child_node)

    #     return child_node
    #     # return child_node, raw_efe












    # # WORKING VERSION 1
    # def get_mini_batches(self):

    #     # Retrieve transition data in mini batches
    #     all_obs_batch, all_actions_batch, reward_batch_t1, terminated_batch_t2, truncated_batch_t2 = self.memory.sample(
    #         self.obs_indices, self.action_indices,
    #         self.reward_indices, self.terminated_indicies, self.truncated_indicies,
    #         self.max_n_indices, self.batch_size
    #     )

    #     # 1. Retrieve a batch of observations for 3 consecutive points in time
    #     obs_batch_t0 = all_obs_batch[:, 0].view([self.batch_size] + [dim for dim in self.obs_shape])
    #     obs_batch_t1 = all_obs_batch[:, 1].view([self.batch_size] + [dim for dim in self.obs_shape])
    #     obs_batch_t2 = all_obs_batch[:, 2].view([self.batch_size] + [dim for dim in self.obs_shape])

    #     # 2. Retrieve a batch of distributions over states for 3 consecutive points in time
    #     pred_state_mu_batch_t0_from_obs, pred_state_var_batch_t0_from_obs = self.generative_observation_net(obs_batch_t0)
    #     pred_state_mu_batch_t1_from_obs, pred_state_var_batch_t1_from_obs = self.generative_observation_net(obs_batch_t1)
    #     pred_state_mu_batch_t2_from_obs, pred_state_var_batch_t2_from_obs = self.generative_observation_net(obs_batch_t2)

    #     # # 3. Combine the sufficient statistics (mean and variance) into a single vector for each state batch
    #     # state_batch_t0 = torch.cat((pred_state_mu_batch_t0_from_obs, pred_state_var_batch_t0_from_obs), dim=1)
    #     # state_batch_t1 = torch.cat((pred_state_mu_batch_t1_from_obs, pred_state_var_batch_t1_from_obs), dim=1)
    #     # state_batch_t2 = torch.cat((pred_state_mu_batch_t2_from_obs, pred_state_var_batch_t2_from_obs), dim=1)

    #     # Retrieve the agent's action history for time t0 and time t1
    #     action_batch_t0 = all_actions_batch[:, 0].unsqueeze(1)
    #     action_batch_t1 = all_actions_batch[:, 1].unsqueeze(1)

    #     # At time t0 predict the state at time t1:
    #     X = torch.cat((pred_state_mu_batch_t0_from_obs.detach(), action_batch_t0.float()), dim=1)
    #     pred_mu_batch_t0t1_from_s_a, pred_var_batch_t0t1_from_s_a = self.generative_transition_net(X)

    #     pred_error_batch_t0t1 = torch.mean(
    #         self.kl_divergence_diag_cov_gaussian(pred_state_mu_batch_t1_from_obs, pred_state_var_batch_t1_from_obs, pred_mu_batch_t0t1_from_s_a, pred_var_batch_t0t1_from_s_a )
    #     )

    #     return (
    #         obs_batch_t0, obs_batch_t1, obs_batch_t2, action_batch_t0,
    #         action_batch_t1, reward_batch_t1, terminated_batch_t2, truncated_batch_t2,
    #         pred_error_batch_t0t1
    #     )

    # # WORKING VERSION 2
    # def get_mini_batches_AcT(self):

    #     # Retrieve transition data in mini batches
    #     all_obs_batch, all_actions_batch, reward_batch_t1, terminated_batch_t2, truncated_batch_t2 = self.memory.sample(
    #         self.obs_indices, self.action_indices,
    #         self.reward_indices, self.terminated_indicies, self.truncated_indicies,
    #         self.max_n_indices, self.batch_size
    #     )

    #     # 1. Retrieve a batch of observations for 2 consecutive points in time
    #     obs_batch_t0 = all_obs_batch[:, 0].view([self.batch_size] + [dim for dim in self.obs_shape]) # time t-1
    #     obs_batch_t1 = all_obs_batch[:, 1].view([self.batch_size] + [dim for dim in self.obs_shape]) # time t 
    #     # obs_batch_t2 = all_obs_batch[:, 2].view([self.batch_size] + [dim for dim in self.obs_shape])

    #     # 2. Retrieve a batch of distributions over states for 3 consecutive points in time
    #     # pred_state_mu_batch_t0_from_obs, pred_state_var_batch_t0_from_obs = self.generative_observation_net(obs_batch_t0)
    #     # pred_state_mu_batch_t1_from_obs, pred_state_var_batch_t1_from_obs = self.generative_observation_net(obs_batch_t1)
    #     # pred_state_mu_batch_t2_from_obs, pred_state_var_batch_t2_from_obs = self.generative_observation_net(obs_batch_t2)

    #     # # 3. Combine the sufficient statistics (mean and variance) into a single vector for each state batch
    #     # state_batch_t0 = torch.cat((pred_state_mu_batch_t0_from_obs, pred_state_var_batch_t0_from_obs), dim=1)
    #     # state_batch_t1 = torch.cat((pred_state_mu_batch_t1_from_obs, pred_state_var_batch_t1_from_obs), dim=1)
    #     # state_batch_t2 = torch.cat((pred_state_mu_batch_t2_from_obs, pred_state_var_batch_t2_from_obs), dim=1)

    #     # Retrieve the agent's action history for time t0 and time t1
    #     action_batch_t0 = all_actions_batch[:, 0].unsqueeze(1)
    #     action_batch_t1 = all_actions_batch[:, 1].unsqueeze(1)

    #     # At time t0 predict the state at time t1:
    #     X = torch.cat((pred_state_mu_batch_t0_from_obs.detach(), action_batch_t0.float()), dim=1)
    #     pred_mu_batch_t0t1_from_s_a, pred_var_batch_t0t1_from_s_a = self.generative_transition_net(X)

    #     # pred_error_batch_t0t1 = torch.mean(
    #     #     self.kl_divergence_diag_cov_gaussian(pred_state_mu_batch_t1_from_obs, pred_state_var_batch_t1_from_obs, pred_mu_batch_t0t1_from_s_a, pred_var_batch_t0t1_from_s_a )
    #     # )

    #     # return (
    #     #     obs_batch_t0, obs_batch_t1, obs_batch_t2, action_batch_t0,
    #     #     action_batch_t1, reward_batch_t1, terminated_batch_t2, truncated_batch_t2,
    #     #     pred_error_batch_t0t1
    #     # )
    #     return (
    #         obs_batch_t0, obs_batch_t1, obs_batch_t2, action_batch_t0,
    #         action_batch_t1, reward_batch_t1, terminated_batch_t2, truncated_batch_t2,
    #         pred_error_batch_t0t1
    #     )


# class MVGaussianModel(nn.Module):

#     def __init__(self, n_inputs, n_outputs, n_hidden, lr=1e-3, softmax=False, device='cpu'):
#         super(MVGaussianModel, self).__init__()

#         self.n_inputs = n_inputs
#         self.n_outputs = n_outputs
#         self.n_hidden = n_hidden

#         self.fc1 = nn.Linear(n_inputs, n_hidden)
#         self.fc2 = nn.Linear(n_hidden, n_hidden)
#         self.fc3 = nn.Linear(n_hidden, n_hidden) # added

#         self.mean_fc = nn.Linear(n_hidden, n_outputs)
#         self.var_fc = nn.Linear(n_hidden, n_outputs)

#         self.softmax = softmax # If true apply a softmax function to the output

#         self.optimizer = optim.Adam(self.parameters(), lr) # Adam optimizer

#         self.device = device
#         self.to(self.device)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.relu(self.fc3(x))

#         mean = self.mean_fc(x)

#         if self.softmax: # for action-probability affordance

#             action_probs_mean_vector = F.softmax(mean, dim=-1).clamp(min=1e-9, max=1-1e-9)

#             return action_probs_mean_vector

#         # var = torch.exp(self.var_fc(x)) # exp to ensure positivity
#         var = torch.log(1 + torch.exp(self.var_fc(x))) # softplus to ensure positivity

#         return mean, var




    # # WORKING VERSION 5
    # def active_inferece_tree_search(self, initial_state_belief, delta, epsilon):

    #     # print("INSIDE active_inferece_tree_search")

    #     '''
    #     The focal_node is "focal" in the sense that it has either already been expanded,
    #     and is simply being visited again, or it has just been expanded for the first time. 

    #     The ONLY AcT procedures to be directly called in AcT are: 
    #         1. self.tree_policy_AcT()
    #         2. self.evaluate_AcT()
    #         3. self.path_integrate_AcT
    #     '''

    #     all_nodes = []

    #     # Initializing the planning tree
    #     root_node = Node(state_belief=initial_state_belief, action_at_tau_minus_one=None, parent=None, future_time_horizon = self.planning_horizon)

    #     all_nodes.append(root_node)

    #     # # Initialize the dummy "parent" to the root node
    #     # dummy_parent = Node(future_time_horizon = self.planning_horizon)
    #     # dummy_parent.children = [root_node]
    #     # root_node.parent = dummy_parent

    #     # Begin the planning time horizon
    #     # t = 0
    #     # t = 1 # start from 1 due to delta ** t < epsilon. 

    #     # print(f"\nactive_inferece_tree_search - (self.delta ** t < self.epsilon): {(self.delta ** t < self.epsilon)}")
    #     # print(f"delta: {delta}, epsilon: {epsilon}\n")

    #     # while not self.halting_conditions_satisfied_AcT(t, delta, epsilon): # Evaluating to False
    #     # while self.halting_conditions_not_satisfied_AcT(t, delta, epsilon): # Evaluating to False
    #     # while not (self.delta ** t < self.epsilon):
    #     for t in range(1, 2):
    #     # for t in range(0, 3):

    #         # print(f"\nAcT - not self.halting_conditions_satisfied_AcT(t, delta, epsilon): {not self.halting_conditions_satisfied_AcT(t, delta, epsilon)}")
    #         # print(f"AcT - delta: {delta}, epsilon: {epsilon}")

    #         # print(f"\nactive_inferece_tree_search - exec tree_policy_AcT, t = {t}")

    #         # Perform tree policy - either variational_inference or expand
    #         focal_node = self.tree_policy_AcT(root_node)

    #         all_nodes.append(focal_node)

    #         # print(f"AcT - tree_policy_AcT complete, t = {t}")

    #         # print(f"active_inferece_tree_search - exec evaluate_AcT, t = {t}")

    #         # Evaluate the expanded node
    #         g_delta = self.evaluate_AcT(focal_node, delta)

    #         # print(f"AcT - evaluate_AcT complete, t = {t}")

    #         # print(f"active_inferece_tree_search - exec path_integrate_AcT, t = {t}")

    #         # Perform path integration
    #         self.path_integrate_AcT(focal_node, g_delta)

    #         # print(f"AcT - exec path_integrate_AcT, t = {t}\n")

    #         # print(f"\nactive_inferece_tree_search - focal_node.parent:\n{focal_node.parent}")
    #         # print(f"active_inferece_tree_search - g_delta: {g_delta}\n")

    #         # print(f"active_inferece_tree_search - self.delta ** t: {self.delta ** t}")
    #         # print(f"active_inferece_tree_search - self.epsilon: {self.epsilon}\n")

    #         # t += 1

    #     return root_node, all_nodes


    # def select_action_dAIF(self, obs):

    #     # print("\tINSIDE select_action_dAIF")

    #     with torch.no_grad():

    #         # Determine the action distribution given the current observation:
    #         # policy = self.policy_net(obs) # OG VERSION
    #         # normalised_action_means, action_stdvs = self.policy_net(obs) # THESE ARE SOMETIMES NAN
    #         normalised_action_means = self.policy_net(obs) # THESE ARE SOMETIMES NAN

    #         # print(f"\n\nselect_action_dAIF - obs: {obs}")
    #         # print(f"select_action_dAIF - normalised_action_means: {normalised_action_means}")
    #         # # # print(f"select_action_dAIF - action_stdvs: {action_stdvs}")
    #         # print(f"select_action_dAIF - torch.multinomial(normalised_action_means, 1): {torch.multinomial(normalised_action_means, 1)}\n\n")

    #         # return torch.multinomial(policy, 1) # OG VERSION
    #         return torch.multinomial(normalised_action_means, 1)






















    # def learn(self):

    #     # print("\tINSIDE learn")

    #     # If there are not enough transitions stored in memory, return:
    #     if self.memory.push_count - self.max_n_indices*2 < self.batch_size:
    #         return

    #     # After every freeze_period time steps, update the target network:
    #     if self.freeze_cntr % self.freeze_period == 0:
    #         self.target_net.load_state_dict(self.efe_value_net.state_dict())
    #     self.freeze_cntr += 1

    #     # # Retrieve transition data in mini batches:
    #     # (
    #     #     obs_batch_t0, obs_batch_t1, obs_batch_t2, action_batch_t0,
    #     #     action_batch_t1, reward_batch_t1, terminated_batch_t2, truncated_batch_t2,
    #     #     pred_error_batch_t0t1
    #     # ) = self.get_mini_batches()

    #     # Retrieve transition data in mini batches:
    #     (
    #         obs_batch_t0, obs_batch_t1, action_batch_t0,
    #         action_batch_t1, reward_batch_t1, terminated_batch_t2, truncated_batch_t2,
    #         pred_mean_state_batch_t1, pred_var_state_batch_t1
    #     ) = self.get_mini_batches_AcT()

    #     # # Compute the value network loss:
    #     # efe_value_net_loss = self.compute_EFE_efe_value_net_loss(
    #     #     obs_batch_t1, obs_batch_t2,
    #     #     action_batch_t1, reward_batch_t1,
    #     #     terminated_batch_t2, truncated_batch_t2, pred_error_batch_t0t1
    #     # )

    #     # # Compute the variational free energy:
    #     # VFE = self.compute_VFE(obs_batch_t1, pred_error_batch_t0t1)
    #     free_energy_loss = self.calculate_free_energy_loss(pred_mean_state_batch_t1, action_batch_t1)

    #     # Reset the gradients:
    #     self.generative_transition_net.optimizer.zero_grad()
    #     self.variational_transition_net.zero_grad()
    #     # self.policy_net.optimizer.zero_grad()
    #     # self.efe_value_net.optimizer.zero_grad()
    #     self.generative_observation_net.optimizer.zero_grad()

    #     # Compute the gradients:
    #     # VFE.backward()
    #     # efe_value_net_loss.backward()

    #     free_energy_loss.backward()

    #     # Perform gradient descent:
    #     self.generative_transition_net.optimizer.step()
    #     self.variational_transition_net.optimizer.step()
    #     # self.policy_net.optimizer.step()
    #     # self.efe_value_net.optimizer.step()
    #     self.generative_observation_net.optimizer.step()




    # def calculate_entropy(probabilities):
    #     # Calculate entropy given probabilities
    #     log_probabilities = torch.log(probabilities)
    #     entropy = -torch.sum(probabilities * log_probabilities, dim=-1)
    #     return entropy




    # class Model(nn.Module):

#     def __init__(self, n_inputs, n_outputs, n_hidden=64, lr=1e-3, softmax=False, device='cpu'):
#         super(Model, self).__init__()

#         self.n_inputs = n_inputs # Number of inputs
#         self.n_hidden = n_hidden # Number of hidden units
#         self.n_outputs = n_outputs # Number of outputs
#         self.softmax = softmax # If true apply a softmax function to the output

#         self.fc1 = nn.Linear(self.n_inputs, self.n_hidden) # Hidden layer
#         self.fc2 = nn.Linear(self.n_hidden, self.n_outputs) # Output layer

#         self.optimizer = optim.Adam(self.parameters(), lr) # Adam optimizer

#         self.device = device
#         self.to(self.device)

#     def forward(self, x):
#         h_relu = F.relu(self.fc1(x))
#         y = self.fc2(h_relu)

#         if self.softmax:
#             y = F.softmax(self.fc2(h_relu), dim=-1).clamp(min=1e-9, max=1-1e-9)

#         return y



 # def __init__(self, future_time_horizon, state_belief = None, action_at_tau_minus_one = None, parent = None):
    #     """
    #     A class to instantiate a node in the AcT planning tree
    #     """
    #     self.action_space = np.array([0, 1]) # the action space for CartPole-v1
    #     # self.raw_efe = 0
    #     self.predictive_EFE = 0 # The EFE computed at time tau: delta^tau * G(*, v_tau).
    #     self.state_belief = state_belief # x_tau = Q(s_t | pi)
    #     self.visit_count = 0 # the number of times this node has been visited
    #     self.depth = 0 # the depth of the node from the root. Any time a new node is created, must specify its depth?
    #     self.parent = parent # this node's parent node
    #     self.children = [] # this node's children
    #     self.action_at_tau_minus_one = action_at_tau_minus_one # the action that led to the visitation of the present node
    #     self.action_posterior_belief = np.ones(len(self.action_space)) / len(self.action_space) # this is a distribution over all possible actions

    #     # self.unused_actions = [] # a list of all actions that have not been used to transition from this node to a subsequent node.
    #     self.used_actions = [] # a list of all actions that HAVE been used to transition from this node to a subsequent node.

    #     # the "policy so far". Actions are iteratively added to this field upon execution
    #     self.policy = [-1.0] * future_time_horizon # padded with -1 (not a valid action)

    #     # self.gamma = sampled from Gama distribution ?????????????????????????????
    #     # self.policy_prior = E