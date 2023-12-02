import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import datetime
import sys

import gymnasium as gym

import math
import random
import matplotlib.pyplot as plt
import scipy.stats as stats
import copy



'''
Deep Active Inference Tree Search, with diagonal multivariate Gaussian model densities.
state_belief
'''


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title("100 episode moving average of scores")
    plt.show()
    plt.savefig(figure_file)


class ReplayMemory():

    # def __init__(self, capacity, obs_shape, device='cpu'):
    def __init__(self, capacity, obs_shape, device='cuda:0'):

        self.device=device

        self.capacity = capacity # The maximum number of items to be stored in memory

        # Initialize (empty) memory tensors
        self.obs_mem = torch.empty([capacity]+[dim for dim in obs_shape], dtype=torch.float32, device=self.device)
        self.action_mem = torch.empty(capacity, dtype=torch.int64, device=self.device)
        self.reward_mem = torch.empty(capacity, dtype=torch.int8, device=self.device)
        self.terminated_mem = torch.empty(capacity, dtype=torch.int8, device=self.device) # ADDED CODE
        self.truncated_mem = torch.empty(capacity, dtype=torch.int8, device=self.device) # ADDED CODE

        self.push_count = 0 # The number of times new data has been pushed to memory

    def push(self, obs, action, reward, terminated, truncated):

        # Store data to memory
        self.obs_mem[self.position()] = obs
        self.action_mem[self.position()] = action
        self.reward_mem[self.position()] = reward
        self.terminated_mem[self.position()] = terminated # ADDED CODE
        self.truncated_mem[self.position()] = truncated # ADDED CODE

        self.push_count += 1

    def position(self):
        # Returns the next position (index) to which data is pushed
        return self.push_count % self.capacity

    def sample(self, obs_indices, action_indices, reward_indices, terminated_indicies, truncated_indicies, max_n_indices, batch_size):
        # Fine as long as max_n is not greater than the fewest number of time steps an episode can take

        # Pick indices at random
        end_indices = np.random.choice(min(self.push_count, self.capacity)-max_n_indices*2, batch_size, replace=False) + max_n_indices

        # Correct for sampling near the position where data was last pushed
        for i in range(len(end_indices)):
            if end_indices[i] in range(self.position(), self.position()+max_n_indices):
                end_indices[i] += max_n_indices

        # Retrieve the specified indices that come before the end_indices
        obs_batch = self.obs_mem[np.array([index-obs_indices for index in end_indices])]
        action_batch = self.action_mem[np.array([index-action_indices for index in end_indices])]
        reward_batch = self.reward_mem[np.array([index-reward_indices for index in end_indices])]
        # done_batch = self.done_mem[np.array([index-done_indices for index in end_indices])] # OG CODE
        terminated_batch = self.terminated_mem[np.array([index-terminated_indicies for index in end_indices])]
        truncated_batch = self.truncated_mem[np.array([index-truncated_indicies for index in end_indices])]

        # Correct for sampling over multiple episodes
        for i in range(len(end_indices)):
            index = end_indices[i]
            for j in range(1, max_n_indices):
                if self.terminated_mem[index-j] or self.truncated_mem[index-j]: # if self.done_mem[index-j]: # OG CODE
                    for k in range(len(obs_indices)):
                        if obs_indices[k] >= j:
                            obs_batch[i, k] = torch.zeros_like(self.obs_mem[0])
                    for k in range(len(action_indices)):
                        if action_indices[k] >= j:
                            action_batch[i, k] = torch.zeros_like(self.action_mem[0]) # Assigning action '0' might not be the best solution, perhaps as assigning at random, or adding an action for this specific case would be better
                    for k in range(len(reward_indices)):
                        if reward_indices[k] >= j:
                            reward_batch[i, k] = torch.zeros_like(self.reward_mem[0]) # Reward of 0 will probably not make sense for every environment
                    for k in range(len(terminated_indicies)):
                        if terminated_indicies[k] >= j:
                            terminated_batch[i, k] = torch.zeros_like(self.terminated_mem[0])
                    for k in range(len(truncated_indicies)):
                        if truncated_indicies[k] >= j:
                            truncated_batch[i, k] = torch.zeros_like(self.truncated_mem[0])
                    break

        # return obs_batch, action_batch, reward_batch, done_batch # OG CODE
        return obs_batch, action_batch, reward_batch, terminated_batch, truncated_batch

class MVGaussianModel(nn.Module):
    """
    A neural network to predict the suficient statistics of a 
    multivariate Gaussian distribution, with a diagonal covariance matrix.

    Outputs:
        - predicted mean vector of the distribution
        - predicted, main-diagonal covariance vector of the distribution
    """

    def __init__(self, n_inputs, n_outputs, n_hidden, lr=1e-3, device='cpu'):
        super(MVGaussianModel, self).__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden

        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_hidden) # added

        self.mean_fc = nn.Linear(n_hidden, n_outputs)
        self.var_fc = nn.Linear(n_hidden, n_outputs)

        self.optimizer = optim.Adam(self.parameters(), lr) # Adam optimizer
        # self.optimizer = optim.SGD(self.parameters(), lr) # SGD optimizer

        self.device = device
        self.to(self.device)

    def forward(self, x):

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        mean = self.mean_fc(x)

        # var = torch.log(1 + torch.exp(self.var_fc(x))) # softplus to ensure positivity - must use inverse_softplus 
        # var = torch.relu(self.var_fc(x)) # ReLU to ensure positivity
        # var = torch.abs(self.var_fc(x)) # abs to ensure positivity # suare this instead 
        var = self.var_fc(x) ** 2 # square the predicted stdev

        # print(f"\nmean: {mean}")
        # print(f"var: {var}\n")

        return mean, var

class Node:

    def __init__(self, future_time_horizon, pred_mean_state = None, action_at_tau_minus_one = None, parent = None):
        self.action_space = np.array([0, 1]) # the action space for CartPole-v1
        self.raw_efe = 0
        self.predictive_EFE = 0 # The EFE computed at time tau: delta^tau * G(*, v_tau).
        self.pred_mean_state = pred_mean_state # x_tau = Q(s_t | pi)
        self.visit_count = 0 # the number of times this node has been visited
        self.depth = 0 # the depth of the node from the root. Any time a new node is created, must specify its depth?
        self.parent = parent # this node's parent node
        self.children = [] # this node's children
        self.action_at_tau_minus_one = action_at_tau_minus_one # the action that led to the visitation of the present node
        self.action_posterior_belief = torch.ones(len(self.action_space)) / len(self.action_space) # this is a distribution over all possible actions

        # self.unused_actions = [] # a list of all actions that have not been used to transition from this node to a subsequent node.
        self.used_actions = [] # a list of all actions that HAVE been used to transition from this node to a subsequent node.

        # the "policy so far". Actions are iteratively added to this field upon execution
        self.policy = [-1.0] * future_time_horizon # padded with -1 (not a valid action)

        # self.gamma = sampled from Gama distribution ??
        # self.policy_prior = E

    def __iter__(self):
        """
        Iterate over the children of the node.
        """
        return iter(self.children)

    def __str__(self):

        return str(hash(tuple(self.policy)))


class Agent():

    def __init__(self, argv):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # define the exploration factor, 𝜅𝑝:
        self.exploration_factor = 1.0

        # the number of timesteps to plan into the future
        self.planning_horizon = 4

        self.set_parameters(argv) # Set parameters

        self.obs_shape = self.env.observation_space.shape # The shape of observations
        self.obs_size = np.prod(self.obs_shape) # The size of the observation
        self.n_actions = self.env.action_space.n # The number of actions available to the agent
        self.all_actions = np.array([0, 1]) # ADDED
        self.freeze_cntr = 0 # Keeps track of when to (un)freeze the target network

        # # Initialize the networks: OG VERSION
        # self.generative_transition_net = Model(self.obs_size+1, self.obs_size, self.n_hidden_gen_trans, lr=self.lr_gen_trans, device=self.device)
        # self.policy_net = Model(self.obs_size, self.n_actions, self.n_hidden_pol, lr=self.lr_pol, softmax=True, device=self.device)
        # self.efe_value_net = Model(self.obs_size, self.n_actions, self.n_hidden_val, lr=self.lr_val, device=self.device) # input: state_t, output: EFE for all actions in action space

        # Initialize the networks:
        self.generative_transition_net = MVGaussianModel(self.obs_size+1, self.obs_size, self.n_hidden_gen_trans, lr=self.lr_gen_trans, device=self.device)
        self.generative_observation_net = MVGaussianModel(self.obs_size, self.obs_size, self.n_hidden_gen_obs, lr=self.lr_gen_obs, device=self.device)
        # self.variational_transition_net = MVGaussianModel(1, self.obs_size, self.n_hidden_var_trans, lr=self.lr_var_trans, device=self.device)
        self.variational_transition_net = MVGaussianModel(1, self.obs_size, self.n_hidden_var_trans, lr=self.lr_var_trans, device=self.device)

        # self.policy_net = MVGaussianModel(self.obs_size, self.n_actions, self.n_hidden_pol, lr=self.lr_pol, softmax=True, device=self.device) # for use with EFE-value net only 
        # self.efe_value_net = Model(self.obs_size, self.n_actions, self.n_hidden_val, lr=self.lr_val, device=self.device) # input: state_t, output: EFE for all actions in action space

        if self.load_network: # If true: load the networks given paths
            self.generative_transition_net.load_state_dict(torch.load(self.network_load_path.format("trans")))
            self.generative_transition_net.eval()
            # self.policy_net.load_state_dict(torch.load(self.network_load_path.format("pol")))
            # self.policy_net.eval()
            # self.efe_value_net.load_state_dict(torch.load(self.network_load_path.format("val")))
            # self.efe_value_net.eval()

            self.generative_observation_net.load_state_dict(torch.load(self.network_load_path.format("obs")))
            self.generative_observation_net.eval()

            self.variational_transition_net.load_state_dict(torch.load(self.network_load_path.format("var_trans")))
            self.variational_transition_net.eval()

        # # self.target_net = Model(self.obs_size, self.n_actions, self.n_hidden_val, lr=self.lr_val, device=self.device) # OG
        # self.target_net = Model(int(self.obs_size), int(self.n_actions), self.n_hidden_val, lr=self.lr_val, device=self.device) # OG VERSION
        # # self.target_net = MVGaussianModel(int(self.obs_size), int(self.n_actions), self.n_hidden_val, lr=self.lr_val, device=self.device) # OG VERSION
        # self.target_net.load_state_dict(self.efe_value_net.state_dict())

        # Initialize the replay memory
        self.memory = ReplayMemory(self.memory_capacity, self.obs_shape, device=self.device)

        # When sampling from memory at index i, obs_indices indicates that we want observations with indices i-obs_indices, works the same for the others
        self.obs_indices = [2, 1, 0]
        self.action_indices = [2, 1]
        self.reward_indices = [1]

        self.terminated_indicies = [0]
        self.truncated_indicies = [0]

        self.max_n_indices = max(max(self.obs_indices, self.action_indices, self.reward_indices, self.terminated_indicies, self.truncated_indicies)) + 1

    def set_parameters(self, argv):

        # The default parameters
        default_parameters = {
            'run_id':"_rX", 'device':self.device,
            # 'env':'LunarLander-v2', 'n_episodes':50,
            'env':'CartPole-v1', 'n_episodes':5000, # OG VERSION FOR CARTPOLE-V1
            'n_hidden_gen_trans':64, 'lr_gen_trans':1e-3,
            'n_hidden_var_trans':64, 'lr_var_trans':1e-3,
            # 'n_hidden_pol':64, # 'lr_pol':1e-3,
            'n_hidden_gen_obs':64, 'lr_gen_obs': 1e-3,
            'n_hidden_val':64, 'lr_val':1e-4,
            'memory_capacity':65536, 'batch_size':64, 'freeze_period':25,
            'Beta':0.99, 'gamma':1.00,
            'print_timer':100,
            'keep_log':True, 'log_path':"logs/ai_mdp_log{}.txt", 'log_save_timer':10,
            'save_results':True, 'results_path':"results/ai_mdp_results{}.npz", 'results_save_timer':500,
            'save_network':True, 'network_save_path':"networks/ai_mdp_{}net{}.pth", 'network_save_timer':500,
            'load_network':False, 'network_load_path':"networks/ai_mdp_{}net_rX.pth",
            # 'delta': 1.25, 'd': 5, 'epsilon': 90.0, 'k_p': 1.0 # delta probably needs to be less than 1?
            # 'delta': 1.1, 'd': 4, 'epsilon': 1.5, 'k_p': 1.0 # epsilon probably needs to be 0.9?
            'rho': 1.0, 'delta': 0.95, 'd': 4, 'epsilon': 0.86, 'k_p': 1.0 # results in temporal planning horizon = 3  
            # 'delta': 0.95, 'd': 4, 'epsilon': 0.99, 'k_p': 1.0 # results in temporal planning horizon = 3  

        }

        # Possible command:
            # python ai_mdp_agent.py device=cuda:0

        # Adjust the custom parameters according to the arguments in argv
        custom_parameters = default_parameters.copy()
        custom_parameter_msg = "Custom parameters:\n"
        for arg in argv:
            key, value = arg.split('=')
            if key in custom_parameters:
                custom_parameters[key] = value
                custom_parameter_msg += "  {}={}\n".format(key, value)
            else:
                print("Argument {} is unknown, terminating.".format(arg))
                sys.exit()

        def interpret_boolean(param):
            if type(param) == bool:
                return param
            elif param in ['True', '1']:
                return True
            elif param in ['False', '0']:
                return False
            else:
                sys.exit("param '{}' cannot be interpreted as boolean".format(param))

        # Set all parameters
        self.run_id = custom_parameters['run_id'] # Is appended to paths to distinguish between runs
        self.device = custom_parameters['device'] # The device used to run the code

        # self.env = gym.make(custom_parameters['env'], desc=None, map_name="4x4", is_slippery=False) # FrozenLake
        self.env = gym.make(custom_parameters['env']) # OG
        self.n_episodes = int(custom_parameters['n_episodes']) # The number of episodes for which to train

        # set the params for the AcT algorithm:
        self.delta = float(custom_parameters['delta'])
        self.d = int(custom_parameters['d'])
        self.epsilon = float(custom_parameters['epsilon'])
        self.k_p = float(custom_parameters['k_p'])

        # Set number of hidden nodes and learning rate for each network
        self.n_hidden_gen_trans = int(custom_parameters['n_hidden_gen_trans'])
        self.lr_gen_trans = float(custom_parameters['lr_gen_trans'])
        # self.n_hidden_pol = int(custom_parameters['n_hidden_pol'])
        # self.lr_pol = float(custom_parameters['lr_pol'])

        self.n_hidden_var_trans = int(custom_parameters['n_hidden_var_trans'])
        self.lr_var_trans = float(custom_parameters['lr_var_trans'])

        self.n_hidden_gen_obs = int(custom_parameters['n_hidden_gen_obs'])
        self.lr_gen_obs = float(custom_parameters['lr_gen_obs'])

        self.n_hidden_val = int(custom_parameters['n_hidden_val'])
        self.lr_val = float(custom_parameters['lr_val'])

        self.memory_capacity = int(custom_parameters['memory_capacity']) # The maximum number of items to be stored in memory
        self.batch_size = int(custom_parameters['batch_size']) # The mini-batch size
        self.freeze_period = int(custom_parameters['freeze_period']) # The number of time-steps the target network is frozen

        self.gamma = float(custom_parameters['gamma']) # A precision parameter
        self.Beta = float(custom_parameters['Beta']) # The discount rate

        self.rho = float(custom_parameters['rho']) # The EFE entropy discount

        self.print_timer = int(custom_parameters['print_timer']) # Print progress every print_timer episodes

        self.keep_log = interpret_boolean(custom_parameters['keep_log']) # If true keeps a (.txt) log concerning data of this run
        self.log_path = custom_parameters['log_path'].format(self.run_id) # The path to which the log is saved
        self.log_save_timer = int(custom_parameters['log_save_timer']) # The number of episodes after which the log is saved

        self.save_results = interpret_boolean(custom_parameters['save_results']) # If true saves the results to an .npz file
        self.results_path = custom_parameters['results_path'].format(self.run_id) # The path to which the results are saved
        self.results_save_timer = int(custom_parameters['results_save_timer']) # The number of episodes after which the results are saved

        self.save_network = interpret_boolean(custom_parameters['save_network']) # If true saves the policy network (state_dict) to a .pth file
        self.network_save_path = custom_parameters['network_save_path'].format("{}", self.run_id) # The path to which the network is saved
        self.network_save_timer = int(custom_parameters['network_save_timer']) # The number of episodes after which the network is saved

        self.load_network = interpret_boolean(custom_parameters['load_network']) # If true loads a (policy) network (state_dict) instead of initializing a new one
        self.network_load_path = custom_parameters['network_load_path'] # The path from which to laod the network

        msg = "Default parameters:\n"+str(default_parameters)+"\n"+custom_parameter_msg
        # print(msg)

        print("\nDEFAULT PARAMETERS:")
        for key, value in default_parameters.items():
            print(f"{key}: {value}")
        print()

        if self.keep_log: # If true: write a message to the log
            self.record = open(self.log_path, "a")
            self.record.write("\n\n-----------------------------------------------------------------\n")
            self.record.write("File opened at {}\n".format(datetime.datetime.now()))
            self.record.write(msg+"\n")

    def softmax(self, x):
        """
        Return the softmax of x
        """
        sm = nn.Softmax(dim = 0)
        return sm(x)

    def kl_divergence_diag_cov_gaussian(self, mu1, sigma1_sq, mu2, sigma2_sq):
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

        kl_div = 0.5 * torch.sum(
            (sigma2_sq / sigma1_sq) + ((mu1 - mu2)**2 / sigma1_sq) - 1 + torch.log(sigma1_sq / sigma2_sq)
        )

        return kl_div

    def nll_diag_cov_gaussian(self, predicted_cur_obs_mean, predicted_cur_obs_var, input_mean_predicted_next_hidden_state, D):

        # nll = 0.5 * torch.sum(
        #     torch.log(predicted_cur_obs_var) + (input_mean_predicted_next_hidden_state - predicted_cur_obs_mean)**2 / predicted_cur_obs_var + (D/2)*np.log(2 * np.pi)
        # )

        sum_term = torch.sum(
            torch.log(predicted_cur_obs_var) + ((input_mean_predicted_next_hidden_state - predicted_cur_obs_mean) ** 2)/(predicted_cur_obs_var)
        )

        nll = 0.5 * (D * np.log(2 * np.pi) + sum_term)

        return nll

    def diagonal_gaussian_entropy(self, sigma_squared, D):
        """
        Calculate the entropy of a diagonal multivariate Gaussian distribution.

        Parameters:
            sigma_squared (list or numpy.ndarray): Diagonal elements of the covariance matrix.
            D (int): Dimensionality of the data.

        Returns:
            entropy (float): Entropy of the diagonal Gaussian distribution.
        """

        log_det_sigma = torch.sum(torch.log(sigma_squared))
        entropy = 0.5 * (D + D*np.log(2 * np.pi) + log_det_sigma)

        return entropy

    def sample_diagonal_multivariate_gaussian(self, mean, var):
        '''
        Sample from a diagonal multivariate Gaussian distribution using PyTorch.

        Parameters:
        mean (torch.Tensor): Mean vector.
        var (torch.Tensor): variances (diagonal elements of the covariance matrix).

        Returns:
        torch.Tensor: Sampled vector from the Gaussian distribution.
        '''
        sampled_values = torch.normal(mean=mean, std=torch.sqrt(var))
        return sampled_values

    def diagonal_multivariate_gaussian_probs(self, sample, mean, std_devs):
        '''
        Computes the PDF of a diagonal multivariate Gaussian distribution at a given sample using PyTorch.

        Parameters:
        sample (torch.Tensor): The sample point.
        mean (torch.Tensor): Mean vector of the distribution.
        std_devs (torch.Tensor): Standard deviations (diagonal elements of covariance matrix) for each dimension.

        Returns:
        torch.Tensor: Probability density at the sample point.

        # Example usage
        mean_vector = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        std_deviations = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
        sample = torch.tensor([1.2, 2.3, 2.8], dtype=torch.float32)

        pdf_value = diagonal_multivariate_gaussian_probs_torch(sample, mean_vector, std_deviations)
        print("PDF value:", pdf_value.item())
        '''
        exponent = -0.5 * torch.sum(((sample - mean) / std_devs)**2)
        normalization = torch.prod(1.0 / (torch.sqrt(2 * torch.tensor(np.pi)) * std_devs))
        pdf = normalization * torch.exp(exponent)

        return pdf

    def calculate_approximate_EFE(self, mean_state_theta, var_state_theta, mean_obs_xi, var_obs_xi):
        '''
        The hidden state prior information should eventually just be object attributes

        Returns an upper bound aproximation of the Expected Free Energy.
        '''

        with torch.no_grad():

            # 1. Construct the state prior - I think this is: [0.0, 0.0, 0.0, 0.0] 

            # 1.1 Define the mean vector over preffered hidden states:
            mean_x = 0.0
            mean_x_dot = 0.0
            mean_theta = 0.0
            mean_theta_dot = 0.0
            mean_state_prior = torch.tensor([mean_x, mean_x_dot, mean_theta, mean_theta_dot])

            # 1.2 Define the (diagonal) covariance matrix over preffered hidden states:
            var_x = 1.0  # Adjust as needed for x
            var_x_dot = 1.0  # Adjust as needed for x_dot
            var_theta = 0.001  # Small variance for theta to make it highly peaked at 0
            var_theta_dot = 0.001  # Small variance for theta_dot to make it highly peaked at 0

            # 1.3 Create the "covariance matrix" - array of diagonal entries of the covariance matrix: 
            var_state_prior = torch.tensor([var_x, var_x_dot, var_theta, var_theta_dot])

            # 2. Calculate the KL divergence between Q(s_t | pi) and the state prior P(s_t | C):
            state_divergence = self.kl_divergence_diag_cov_gaussian(
                mean_state_theta, var_state_theta,
                mean_state_prior, var_state_prior
            )

            # 3. calculate the approximate expectatied unconditional observation entropy, under Q(s_t) - using Monte Carlo sampling
            N = 60
            entropy = self.diagonal_gaussian_entropy(var_obs_xi, 4) # D = 4 for CartPole-v1
            expected_state_dist = 0
            for i in range(N):
                expected_state_dist += self.diagonal_multivariate_gaussian_probs(
                    self.sample_diagonal_multivariate_gaussian(mean_state_theta, var_state_theta),
                    mean_state_theta,
                    var_state_theta
                ) # the probability of sample from sample_diagonal_multivariate_gaussian
            expected_entropy = (entropy/N)*expected_state_dist


            # 4. Calculate the approximate expected free energy (upper bound on the actual EFE)
            expected_free_energy = state_divergence + (1/self.rho) * expected_entropy

        return expected_free_energy

    def get_mini_batches_AcT(self):

        # Retrieve transition data in mini batches
        all_obs_batch, all_actions_batch, reward_batch_t1, terminated_batch_t2, truncated_batch_t2 = self.memory.sample(
            self.obs_indices, self.action_indices,
            self.reward_indices, self.terminated_indicies, self.truncated_indicies,
            self.max_n_indices, self.batch_size
        )

        # 1. Retrieve a batch of observations for 2 consecutive points in time
        obs_batch_t0 = all_obs_batch[:, 0].view([self.batch_size] + [dim for dim in self.obs_shape]) # time t-1
        obs_batch_t1 = all_obs_batch[:, 1].view([self.batch_size] + [dim for dim in self.obs_shape]) # time t 

        # 2. Retrieve the agent's action history for time t0 and time t1
        action_batch_t0 = all_actions_batch[:, 0].unsqueeze(1) # time t-1
        action_batch_t1 = all_actions_batch[:, 1].unsqueeze(1) # time t

        # cast action_batch_t0 to right dtype
        action_batch_t0 = action_batch_t0.float()

        # print(f"action_batch_t0: {action_batch_t0}")

        # At time t0 predict the state at time t1 - using the variational model:
        pred_mean_state_batch_t1, pred_var_state_batch_t1 = self.variational_transition_net(action_batch_t0) # might need to normalise the inputs

        # print(f"pred_mean_state_batch_t1: {pred_mean_state_batch_t1}, pred_var_state_batch_t1: {pred_var_state_batch_t1}")

        # pred_var_state_batch_t1 = self.inverse_softplus(pred_var_state_batch_t1) # apply inverse softplus

        return (
            obs_batch_t0, obs_batch_t1, action_batch_t0,
            action_batch_t1, reward_batch_t1, terminated_batch_t2, truncated_batch_t2,
            pred_mean_state_batch_t1, pred_var_state_batch_t1
        )


    def select_action_AcT(self, node):

        action_probabilities = node.action_posterior_belief

        action_indices = np.arange(len(action_probabilities))
        chosen_action_index = np.random.choice(action_indices, p = action_probabilities)
        chosen_action = self.all_actions[chosen_action_index]

        return chosen_action


    # def tree_policy_AcT(self, node, B):
    def tree_policy_AcT(self, node):

        while not self.node_is_terminal_leaf_AcT(node):
            if not self.node_fully_expanded_AcT(node):
                node = self.expand_AcT(node)
            else:
                node = self.variational_inference_AcT(node)

        return node

    def get_actions_AcT(self, node):

        '''
        If there remain actions that have note yet been used to make a transition
        from this node to a new child node, return the list of all such unused actions.
        If all actions have been exhausted, return the action: (a_t)* that resulted in the minimum
        predictive_EFE for the resulting child node, upon performing (a_t)*.
        '''

        unused_actions = []

        for action in self.all_actions:

            if action not in node.used_actions:
                unused_actions.append(action)

                # might need to return in a "batch" dimension
                return unused_actions

            else:
                # All actions have been used, so we simply return the best known action
                sorted_children = sorted(node.children, key=lambda child: child.predictive_EFE)
                action_with_minimum_EFE = sorted_children[0].action_at_tau_minus_one

                return action_with_minimum_EFE

    # def expand_AcT(self, node, B):
    def expand_AcT(self, node):

        # print("expand_AcT")

        # perform an unused action:
        unused_actions = self.get_actions_AcT(node)

        rnd_idx = torch.randint(0, len(unused_actions), (1,))

        a_prime_scalar = unused_actions[rnd_idx].item()

        a_prime = torch.tensor([a_prime_scalar], dtype=torch.int64, device=self.device)

        # a_prime = torch.unsqueeze(a_prime, 0) # ADDED
        node.used_actions.append(a_prime)
        node.policy[node.depth] = a_prime.item() # add this action to the "policy so far"

        # At time t0 predict the state belief at t1, after performing action a_prime in state node.pred_mean_state:
        pred_mean_cur_state_and_cur_action = torch.cat((node.pred_mean_state, a_prime), dim = 0) # node.pred_mean_state is now sufficient stats of multivariate gaussian 
        mean_next_state_theta, var_next_state_theta = self.generative_transition_net(pred_mean_cur_state_and_cur_action)

        # At time t1 predict the observation given the predicted state at time t1:
        mean_next_obs_xi, var_next_obs_xi = self.generative_observation_net(mean_next_state_theta)

        # instantiate a child node as a consequence of performing action a_prime
        child_node = Node(future_time_horizon = self.planning_horizon - (node.depth + 1)) # correct time horizon?
        child_node.parent = node
        child_node.depth = node.depth + 1
        # child_node.pred_mean_state = (mean_next_state_theta, var_next_state_theta) # x_tau = Q(s_t | pi)
        child_node.pred_mean_state = mean_next_state_theta
        child_node.action_at_tau_minus_one = a_prime # the action that led to the visitation of the present node
        child_node.policy = copy.deepcopy(node.policy) # copy the policy generated from the history

        # print(f"\nmean_next_state_theta: {mean_next_state_theta}")
        # print(f"var_next_state_theta: {var_next_state_theta}")
        # print(f"mean_next_obs_xi: {mean_next_obs_xi}")
        # print(f"var_next_obs_xi: {var_next_obs_xi}\n")

        # Calculate the approximate Expected Free Energy for the predicted time step - t1:
        raw_efe = self.calculate_approximate_EFE(mean_next_state_theta, var_next_state_theta, mean_next_obs_xi, var_next_obs_xi)

        # print(f"raw_efe: {raw_efe}")

        # store the raw efe as intermediate stage in comuting predictive efe
        child_node.raw_efe = raw_efe

        # finally, add the child node to the node's children
        node.children.append(child_node)

        return child_node

    def node_is_terminal_leaf_AcT(self, node):
        '''
        A node is terminal if at least: not (delta ** root_node.visit_count < epsilon)
        '''

        return self.delta ** node.depth < self.epsilon

    def update_precision(self, depth, alpha, beta):

        per_depth_precision = stats.gamma.rvs(alpha, scale=beta)

        if depth > 0:
            per_depth_precision *= depth

        return per_depth_precision

    def update_action_posterior(self, node, prior_belief_about_policy, precision_tau, EFE_tau):

        # compute the argument to the Boltzmann distribution over actions:
        action_dist_arg = torch.tensor([(self.exploration_factor * np.log(policy_prior) - precision_tau * EFE_tau.detach().numpy()) for policy_prior in prior_belief_about_policy])

        # Construct the Boltzmann distribution over actions - posterior belief about actions (posterior action probabilities):
        action_probs = self.softmax(action_dist_arg)

        # print(f"action_probs: {action_probs}")

        node.action_posterior_belief = action_probs

        return action_probs

    def variational_inference_AcT(self, node):

        # Compute the policy prior - 𝐄:
        prior_belief_about_policy = np.array([math.sqrt(2 * math.log(node.visit_count) / child_node.visit_count) for child_node in node.children]) # differential length through time? set node.children as empty fixed size array?

        # Compute the precision 𝛾_𝜏: for the current time
        precision_tau = self.update_precision(depth = node.depth, alpha = 1, beta = 1)

        # Get delta^tau * G(𝜋_𝜏, 𝑣_𝜏):
        EFE_tau = node.predictive_EFE 

        # Construct the Boltzmann distribution over actions - posterior belief about actions (posterior action probabilities):
        action_probs = self.update_action_posterior(node, prior_belief_about_policy, precision_tau, EFE_tau)

        # sample an action from this Boltzmann distribution:
        selected_child_node = random.choices(node.children, weights=action_probs)[0]

        return selected_child_node

    # def evaluate_AcT(self, node, A, B, C, delta):
    def evaluate_AcT(self, node, delta):

        # calculate the "predictive" EFE from the "raw" efe:
        g_delta = (delta ** node.depth) * node.raw_efe

        # store the "predictive" EFE
        node.predictive_EFE = g_delta

        return g_delta

    # def path_integrate_AcT(self, node, g_delta):
    def path_integrate_AcT(self, node, new_efe_value):

        while node != None:
            node.visit_count += 1
            node.predictive_EFE += (1 / node.visit_count) * (node.predictive_EFE - new_efe_value) # INCREMENT OR ASSIGN HERE?
            # node.predictive_EFE = (1 / node.visit_count) * (node.predictive_EFE - new_efe_value) # INCREMENT OR ASSIGN HERE?
            node = node.parent

    # def halting_conditions_satisfied_AcT(self, t, epsilon, delta):

    #     # print("\tINSIDE halting_conditions_satisfied_AcT")

    #     # simply test to see if within alllowable time horizon - can make more sophisticated
    #     # return not (delta ** t < epsilon)
    #     # return not delta ** t < epsilon

    #     if (delta ** t < epsilon):
    #         # print(f"halting_conditions_satisfied_AcT - NOT satisfied: not delta ** t < epsilon: {not delta ** t < epsilon}")
    #         return False
    #     else:
    #         # print(f"halting_conditions_satisfied_AcT - SATISFIED: not delta ** t < epsilon: {not delta ** t < epsilon}")
    #         return True

    # def halting_conditions_not_satisfied_AcT(self, t, epsilon, delta):

    #     if (delta ** t < epsilon):
    #         # print(f"halting_conditions_satisfied_AcT - NOT satisfied: not delta ** t < epsilon: {not delta ** t < epsilon}")
    #         return True
    #     else:
    #         # print(f"halting_conditions_satisfied_AcT - SATISFIED: not delta ** t < epsilon: {not delta ** t < epsilon}")
    #         return False

    def node_fully_expanded_AcT(self, node):
        '''
        Returns True iff the input node has exhausted the 
        action space in making transitions between itself and its children.
        '''

        if len(node.children) == len(self.all_actions):
            return True
        else:
            return False

    # WORKING VERSION 5
    # def active_inferece_tree_search(self, initial_pred_mean_state_mean, initial_pred_mean_state_var, delta, epsilon):
    def active_inferece_tree_search(self, initial_state_belief_mean, delta, epsilon):
        '''
        The focal_node is "focal" in the sense that it has either already been expanded,
        and is simply being visited again, or it has just been expanded for the first time. 

        initial_pred_mean_state will have to be multivariate gaussian suficient statistics.
        '''

        # print("\nInstantiating the root node")

        # Initializing the planning tree
        root_node = Node(
            pred_mean_state = initial_state_belief_mean, # will have to modify the node class
            action_at_tau_minus_one = None, 
            parent = None, 
            future_time_horizon = self.planning_horizon
        )

        # print("Beginning AcT planning loop")

        # Begin the planning time horizon - hardcoded planning horizon of 3 time steps.
        for t in range(1, self.planning_horizon):

            # print("##########################################")
            # print("Beginning tree_policy_AcT")

            # Perform tree policy - either variational_inference or expand
            focal_node = self.tree_policy_AcT(root_node)

            # print("Beginning evaluate_AcT")

            # Evaluate the expanded node
            g_delta = self.evaluate_AcT(focal_node, delta)

            # print(f"Discounted EFE: {g_delta}")

            # print("Beginning path_integrate_AcT")

            # Perform path integration
            self.path_integrate_AcT(focal_node, g_delta)

            # print("End AcT planning loop")
            # print("##########################################\n")

        return root_node

    def calculate_free_energy_loss(self, pred_mean_state_batch_t1, action_batch_t1):
        """
        Loss function for each neural network, inspired by Catal et al. 

        Consideres the KL divergence between the predicted state belief for the 
        generative and variational models, in addition to the negative log likelihood of 
        the generative observation model.
        """

        # generative transition net inputs:
        pred_mean_cur_state_and_cur_action = torch.cat((pred_mean_state_batch_t1.detach(), action_batch_t1.float()), dim=1)

        # variational transition net inputs:
        current_action = action_batch_t1.float() # must cast to float

        # generative observation net inputs:
        pred_mean_cur_state = pred_mean_state_batch_t1

        # Forward pass through the networks:
        generative_next_state_distribution_mean, generative_next_state_distribution_var = self.generative_transition_net(pred_mean_cur_state_and_cur_action)  # predicted next state prob (gen)

        variational_next_state_distribution_mean, variational_next_state_distribution_var = self.variational_transition_net(current_action) # predicted next state prob (var)

        # print(f"variational_next_state_distribution_mean: {variational_next_state_distribution_mean}, variational_next_state_distribution_var: {variational_next_state_distribution_var}")

        generative_cur_observation_distribution_mean, generative_cur_observation_distribution_var = self.generative_observation_net(pred_mean_cur_state) # predicted current obs (gen)

        # Calculate the predicted state, KL divergence
        kl_divergence = self.kl_divergence_diag_cov_gaussian(
            variational_next_state_distribution_mean, variational_next_state_distribution_var,
            generative_next_state_distribution_mean, generative_next_state_distribution_var
        )

        # Calculate the negative log likelihood of the observation
        neg_log_likelihood = self.nll_diag_cov_gaussian(
            predicted_cur_obs_mean = generative_cur_observation_distribution_mean, 
            predicted_cur_obs_var = generative_cur_observation_distribution_var, 
            input_mean_predicted_next_hidden_state = generative_next_state_distribution_mean,
            D = 4
        )

        # Compute the final loss
        # loss = kl_divergence - neg_log_likelihood
        loss = kl_divergence - neg_log_likelihood

        print(f"\nkl_divergence: {kl_divergence}")
        print(f"neg_log_likelihood: {neg_log_likelihood}")
        print(f"free_energy_loss: {loss}\n")

        return loss


    def learn(self):

        # If there are not enough transitions stored in memory, return:
        if self.memory.push_count - self.max_n_indices*2 < self.batch_size:
            return

        # Retrieve transition data in mini batches:
        (
            obs_batch_t0, obs_batch_t1, action_batch_t0,
            action_batch_t1, reward_batch_t1, terminated_batch_t2, truncated_batch_t2,
            pred_mean_state_batch_t1, pred_var_state_batch_t1
        ) = self.get_mini_batches_AcT()

        # print(f"\naction_batch_t1: {action_batch_t1}")
        # print(f"pred_mean_state_batch_t1: {pred_mean_state_batch_t1}")

        # Compute the free energy loss for the neural networks:
        free_energy_loss = self.calculate_free_energy_loss(pred_mean_state_batch_t1, action_batch_t1) # pred_mean_state_batch_t1 becomes nan!

        # Reset the gradients:
        self.generative_transition_net.optimizer.zero_grad()
        self.variational_transition_net.zero_grad()
        self.generative_observation_net.optimizer.zero_grad()

        # backpropagate 
        free_energy_loss.backward()

        # Perform gradient descent:
        self.generative_transition_net.optimizer.step()
        self.variational_transition_net.optimizer.step()
        self.generative_observation_net.optimizer.step()


    def train(self):

        filename = f"Deep_AIF_MDP_Cart_Pole_v1"
        figure_file = f"plots/{filename}.png"

        msg = "Environment is: {}\nTraining started at {}".format(self.env.unwrapped.spec.id, datetime.datetime.now())
        print(msg)
        print(f"\n----------------- Deep Active Inference Tree Search -----------------\n")
        if self.keep_log:
            self.record.write(msg+"\n")

        results = []

        if torch.cuda.is_available():
            print("CUDA is available")
        else:
            print("CUDA is NOT available")
        print(f"self.device: {self.device}")
        print(f"Playing {self.n_episodes} episodes")

        # Define the standard deviation of the Gaussian noise
        noise_std = 0.1

        for ith_episode in range(self.n_episodes):

            total_reward = 0
            obs, info = self.env.reset()
            noisy_obs = obs + noise_std * np.random.randn(*obs.shape)

            terminated = False
            truncated = False
            done = terminated or truncated

            reward = 0

            while not done:

                # Calculate the initial hidden state belief:
                state_belief_mean = torch.tensor([noisy_obs[0], noisy_obs[1], noisy_obs[2], noisy_obs[3]], dtype=torch.float32, device=self.device)

                # ACTIVE INFERENCE TREE SEARCH: Do Active Inference Tree Search:
                # root_node = self.active_inferece_tree_search(state_belief_mean, self.delta, self.epsilon)

                # cast the noisy obs to tensor
                noisy_obs = torch.tensor(noisy_obs, dtype = torch.float32, device = self.device)

                # ACTIVE INFERENCE TREE SEARCH: Select the best action based on the updated tree
                action = self.env.action_space.sample()
                # action = self.select_action_AcT(root_node)
                action = torch.tensor([action], dtype=torch.int64, device=self.device) # cast action back to tensor

                # print(f"action: {action}")

                # Push the noisy tuple to self.memory
                self.memory.push(noisy_obs, action, reward, terminated, truncated)

                # ACTIVE INFERENCE TREE SEARCH: Execute the selected action and transition to a new state - AcT:
                # obs, reward, terminated, truncated, _  = self.env.step(action)
                obs, reward, terminated, truncated, _  = self.env.step(action[0].item())

                # update observation
                noisy_obs = obs + noise_std * np.random.randn(*obs.shape)

                # increment the per-episode reward:
                total_reward += reward

                # learn
                self.learn()

                if terminated or truncated:

                    done = True

                    noisy_obs = torch.tensor(noisy_obs, dtype = torch.float32, device = self.device)

                    self.memory.push(noisy_obs, -99, -99, terminated, truncated)

            results.append(total_reward)

            # Print and keep a (.txt) record of stuff
            if ith_episode > 0 and ith_episode % self.print_timer == 0:
                avg_reward = np.mean(results)
                last_x = np.mean(results[-self.print_timer:])
                msg = "Episodes: {:4d}, avg score: {:3.2f}, over last {:d}: {:3.2f}".format(ith_episode, avg_reward, self.print_timer, last_x)
                print(msg)

                if self.keep_log:
                    self.record.write(msg+"\n")

                    if ith_episode % self.log_save_timer == 0:
                        self.record.close()
                        self.record = open(self.log_path, "a")

            # If enabled, save the results and the network (state_dict)
            if self.save_results and ith_episode > 0 and ith_episode % self.results_save_timer == 0:
                np.savez("results/intermediary/intermediary_results{}_{:d}".format(self.run_id, ith_episode), np.array(results))
            if self.save_network and ith_episode > 0 and ith_episode % self.network_save_timer == 0:
                torch.save(self.generative_transition_net.state_dict(), "networks/intermediary/intermediary_transnet{}_{:d}.pth".format(self.run_id, ith_episode))
                torch.save(self.variational_transition_net.state_dict(), "networks/intermediary/intermediary_vartransnet{}_{:d}.pth".format(self.run_id, ith_episode))
                # torch.save(self.policy_net.state_dict(), "networks/intermediary/intermediary_polnet{}_{:d}.pth".format(self.run_id, ith_episode))
                # torch.save(self.efe_value_net.state_dict(), "networks/intermediary/intermediary_valnet{}_{:d}.pth".format(self.run_id, ith_episode))

                torch.save(self.generative_observation_net.state_dict(), "networks/intermediary/intermediary_obsnet{}_{:d}.pth".format(self.run_id, ith_episode))

        self.env.close()

        # If enabled, save the results and the network (state_dict)
        if self.save_results:
            np.savez("results/intermediary/intermediary_results{}_end".format(self.run_id), np.array(results))
            np.savez(self.results_path, np.array(results))
        if self.save_network:
            torch.save(self.generative_transition_net.state_dict(), "networks/intermediary/intermediary_transnet{}_end.pth".format(self.run_id))
            # torch.save(self.policy_net.state_dict(), "networks/intermediary/intermediary_polnet{}_end.pth".format(self.run_id))
            # torch.save(self.efe_value_net.state_dict(), "networks/intermediary/intermediary_valnet{}_end.pth".format(self.run_id))

            torch.save(self.generative_observation_net.state_dict(), "networks/intermediary/intermediary_obsnet{}_end.pth".format(self.run_id))

            torch.save(self.variational_transition_net.state_dict(), "networks/intermediary/intermediary_vartransnet{}_end.pth".format(self.run_id))

            torch.save(self.generative_transition_net.state_dict(), self.network_save_path.format("trans"))
            torch.save(self.variational_transition_net.state_dict(), self.network_save_path.format("vartrans"))
            # torch.save(self.policy_net.state_dict(), self.network_save_path.format("pol"))
            # torch.save(self.efe_value_net.state_dict(), self.network_save_path.format("val"))

            torch.save(self.generative_observation_net.state_dict(), self.network_save_path.format("obs"))

        # Print and keep a (.txt) record of stuff
        msg = "Training finished at {}".format(datetime.datetime.now())
        print(msg)
        if self.keep_log:
            self.record.write(msg)
            self.record.close()

        x = [i + 1 for i in range(self.n_episodes)]
        plot_learning_curve(x, results, figure_file)


if __name__ == "__main__":
    agent = Agent(sys.argv[1:])
    agent.train()

#     def train(self):

#         filename = f"Deep_AIF_MDP_Cart_Pole_v1"
#         figure_file = f"plots/{filename}.png"

#         msg = "Environment is: {}\nTraining started at {}".format(self.env.unwrapped.spec.id, datetime.datetime.now())
#         print(msg)
#         print(f"\n----------------- Deep Active Inference Tree Search -----------------\n")
#         if self.keep_log:
#             self.record.write(msg+"\n")

#         results = []

#         if torch.cuda.is_available():
#             print("CUDA is available")
#         else:
#             print("CUDA is NOT available")
#         print(f"self.device: {self.device}")
#         print(f"Playing {self.n_episodes} episodes")

#         # Define the standard deviation of the Gaussian noise
#         noise_std = 0.1

#         for ith_episode in range(self.n_episodes):

#             total_reward = 0
#             obs, info = self.env.reset()
#             noisy_obs = obs + noise_std * np.random.randn(*obs.shape)

#             terminated = False
#             truncated = False
#             done = terminated or truncated

#             reward = 0

#             while not done:

#                 # Calculate the initial hidden state belief:
#                 state_belief_mean = torch.tensor([noisy_obs[0], noisy_obs[1], noisy_obs[2], noisy_obs[3]], dtype=torch.float32, device=self.device)

#                 # ACTIVE INFERENCE TREE SEARCH: Do Active Inference Tree Search:
#                 root_node = self.active_inferece_tree_search(state_belief_mean, self.delta, self.epsilon)

#                 # cast the noisy obs to tensor
#                 noisy_obs = torch.tensor(noisy_obs, dtype = torch.float32, device = self.device)

#                 # ACTIVE INFERENCE TREE SEARCH: Select the best action based on the updated tree
#                 action = self.select_action_AcT(root_node)
#                 action = torch.tensor([action], dtype=torch.int64, device=self.device) # cast action back to tensor

#                 # Push the noisy tuple to self.memory
#                 self.memory.push(noisy_obs, action, reward, terminated, truncated)

#                 # ACTIVE INFERENCE TREE SEARCH: Execute the selected action and transition to a new state - AcT:
#                 # obs, reward, terminated, truncated, _  = self.env.step(action)
#                 obs, reward, terminated, truncated, _  = self.env.step(action[0].item())

#                 # update observation
#                 noisy_obs = obs + noise_std * np.random.randn(*obs.shape)

#                 # increment the per-episode reward:
#                 total_reward += reward

#                 # learn
#                 self.learn()

#                 if terminated or truncated:

#                     done = True

#                     noisy_obs = torch.tensor(noisy_obs, dtype = torch.float32, device = self.device)

#                     self.memory.push(noisy_obs, -99, -99, terminated, truncated)

#             results.append(total_reward)

#             # Print and keep a (.txt) record of stuff
#             if ith_episode > 0 and ith_episode % self.print_timer == 0:
#                 avg_reward = np.mean(results)
#                 last_x = np.mean(results[-self.print_timer:])
#                 msg = "Episodes: {:4d}, avg score: {:3.2f}, over last {:d}: {:3.2f}".format(ith_episode, avg_reward, self.print_timer, last_x)
#                 print(msg)

#                 if self.keep_log:
#                     self.record.write(msg+"\n")

#                     if ith_episode % self.log_save_timer == 0:
#                         self.record.close()
#                         self.record = open(self.log_path, "a")

#             # If enabled, save the results and the network (state_dict)
#             if self.save_results and ith_episode > 0 and ith_episode % self.results_save_timer == 0:
#                 np.savez("results/intermediary/intermediary_results{}_{:d}".format(self.run_id, ith_episode), np.array(results))
#             if self.save_network and ith_episode > 0 and ith_episode % self.network_save_timer == 0:
#                 torch.save(self.generative_transition_net.state_dict(), "networks/intermediary/intermediary_transnet{}_{:d}.pth".format(self.run_id, ith_episode))
#                 torch.save(self.variational_transition_net.state_dict(), "networks/intermediary/intermediary_vartransnet{}_{:d}.pth".format(self.run_id, ith_episode))
#                 # torch.save(self.policy_net.state_dict(), "networks/intermediary/intermediary_polnet{}_{:d}.pth".format(self.run_id, ith_episode))
#                 # torch.save(self.efe_value_net.state_dict(), "networks/intermediary/intermediary_valnet{}_{:d}.pth".format(self.run_id, ith_episode))

#                 torch.save(self.generative_observation_net.state_dict(), "networks/intermediary/intermediary_obsnet{}_{:d}.pth".format(self.run_id, ith_episode))

#         self.env.close()

#         # If enabled, save the results and the network (state_dict)
#         if self.save_results:
#             np.savez("results/intermediary/intermediary_results{}_end".format(self.run_id), np.array(results))
#             np.savez(self.results_path, np.array(results))
#         if self.save_network:
#             torch.save(self.generative_transition_net.state_dict(), "networks/intermediary/intermediary_transnet{}_end.pth".format(self.run_id))
#             # torch.save(self.policy_net.state_dict(), "networks/intermediary/intermediary_polnet{}_end.pth".format(self.run_id))
#             # torch.save(self.efe_value_net.state_dict(), "networks/intermediary/intermediary_valnet{}_end.pth".format(self.run_id))

#             torch.save(self.generative_observation_net.state_dict(), "networks/intermediary/intermediary_obsnet{}_end.pth".format(self.run_id))

#             torch.save(self.variational_transition_net.state_dict(), "networks/intermediary/intermediary_vartransnet{}_end.pth".format(self.run_id))

#             torch.save(self.generative_transition_net.state_dict(), self.network_save_path.format("trans"))
#             torch.save(self.variational_transition_net.state_dict(), self.network_save_path.format("vartrans"))
#             # torch.save(self.policy_net.state_dict(), self.network_save_path.format("pol"))
#             # torch.save(self.efe_value_net.state_dict(), self.network_save_path.format("val"))

#             torch.save(self.generative_observation_net.state_dict(), self.network_save_path.format("obs"))

#         # Print and keep a (.txt) record of stuff
#         msg = "Training finished at {}".format(datetime.datetime.now())
#         print(msg)
#         if self.keep_log:
#             self.record.write(msg)
#             self.record.close()

#         x = [i + 1 for i in range(self.n_episodes)]
#         plot_learning_curve(x, results, figure_file)


# if __name__ == "__main__":
#     agent = Agent(sys.argv[1:])
#     agent.train()


'''
Current AcT Exec Order:

tree_policy_AcT
node_is_terminal_leaf_AcT
node_fully_expanded_AcT
variational_inference_AcT
node_is_terminal_leaf_AcT
node_fully_expanded_AcT
expand_AcT
node_is_terminal_leaf_AcT
node_fully_expanded_AcT
expand_AcT
node_is_terminal_leaf_AcT
evaluate_AcT
path_integrate_AcT

'''

 # def calculate_approximate_EFE(self, mean_state_theta, var_state_theta, mean_obs_xi, var_obs_xi):
    #     '''
    #     The hidden state prior information should eventually just be object attributes

    #     Returns an upper bound aproximation of the Expected Free Energy.
    #     '''

    #     with torch.no_grad():

    #         # 1. Construct the state prior - I think this is: [x, x_dot, 0.0, 0.0] where x \in (-2.4, 2.4)
    #         # 1.1 Define the mean vector over preffered hidden states:
    #         mean_x = 0.0
    #         mean_x_dot = 0.0
    #         mean_theta = 0.0
    #         mean_theta_dot = 0.0
    #         mean_state_prior = np.array([mean_x, mean_x_dot, mean_theta, mean_theta_dot])

    #         # 1.2 Define the (diagonal) covariance matrix over preffered hidden states:
    #         var_x = 0.5  # Adjust as needed for x
    #         var_x_dot = 1.0  # Adjust as needed for x_dot
    #         var_theta = 0.001  # Small variance for theta to make it highly peaked at 0
    #         var_theta_dot = 0.001  # Small variance for theta_dot to make it highly peaked at 0

    #         # 1.3 Create the "covariance matrix" - array of diagonal entries of the covariance matrix: 
    #         var_state_prior = np.array([var_x, var_x_dot, var_theta, var_theta_dot])

    #         # 2. Calculate the KL divergence between Q(s_t | pi) and the state prior P(s_t | C):
    #         state_divergence = self.kl_divergence_diag_cov_gaussian(
    #             mean_state_theta, var_state_theta,
    #             mean_state_prior, var_state_prior
    #         )

    #         # 3. calculate the unconditional entropy of the observation model:
    #         entropy = self.diagonal_gaussian_entropy(var_obs_xi, 4) # D = 4 for CartPole-v1

    #         # 4. Calculate the expected free energy (upper bound on the actual EFE)
    #         expected_free_energy = state_divergence + (1/self.rho) * entropy

    #     return expected_free_energy