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
from matplotlib.animation import FuncAnimation
import scipy.stats as stats
import torch.distributions.multivariate_normal as mvn
from networkx.drawing.nx_pydot import graphviz_layout
import networkx as nx
import copy

import pdb


'''
Questions and Problems:

* increasing KL divergence:
    - indicates inability to fit the variational posterior to the true posterior
* log likelihood should typically be negative or close to 0 (so says ChatGPT)
    - high values indicate model asigns high probability to the observed data (CGPT)
* Final loss:
    - The loss value being -2528803028992.0 is consistent with the large positive log-likelihood. 
    The loss is typically a function that you want to minimize during training. 
    A large negative value like this suggests that the model is trying to maximize 
    the log-likelihood, which is not a typical behavior. (CGPT)

* We want to minimize the loss function
    - isn't it the case that PyTorch can only minimise/maximise?
    - LOOK INTO THIS


'''


unique_node_id = 0

losses = []
kl_divs = []
avg_log_likelis = []

line = None


def draw_tree(root_node):

    G = nx.DiGraph()

    # Create a queue for bfs
    queue = [(None, root_node)]

    while queue:
        parent, node = queue.pop(0)

        # Add the current node to the graph
        G.add_node(str(node))


        if parent is not None:
            G.add_edge(str(parent), str(node))

        # Add child nodes to the queue
        for child in node.children:
            queue.append((node, child))

    # draw the tree
    pos = graphviz_layout(G, prog = "dot")
    nx.draw(G, pos, with_labels = True, node_size = 2000)
    plt.show()



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

        return obs_batch, action_batch, reward_batch, terminated_batch, truncated_batch

class MVGaussianModel(nn.Module):

    def __init__(self, n_inputs, n_outputs, n_hidden, lr = 1e-4, device = 'cpu', model = None):
        super(MVGaussianModel, self).__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden

        self.model = model

        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)

        self.mean_fc = nn.Linear(n_hidden, n_outputs)
        self.stdev = nn.Linear(n_hidden, n_outputs)

        self.optimizer = optim.Adam(self.parameters(), lr) # Adam optimizer

        self.device = device
        self.to(self.device)

    def forward(self, x):

        x_1 = torch.relu(self.fc1(x))
        x_2 = torch.relu(self.fc2(x_1))

        mean = self.mean_fc(x_2)
        var = self.stdev(x_2) ** 2
        log_var = torch.log(var)

        return mean, log_var # mean and log variance

    # def reparameterize(self, mean, var): # this should be called in get_mini_batches_AcT and self.forward - as per H and L
    #     std = torch.sqrt(var)
    #     epsilon = torch.randn_like(std)  # Sample from standard Gaussian
    #     sampled_value = mean + epsilon * std  # Reparameterization trick

    #     return sampled_value

    def reparameterize(self, mean, log_var):
        # Apply reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        sampled_value = mean + eps * std

        return sampled_value

class Node:

    def __init__(self, pred_mean_state = None, pred_var_state = None, action_at_tau_minus_one = None, parent = None, name = None):
        # self.action_space = np.array([0, 1]) # the action space for CartPole-v1
        self.action_space = torch.tensor([0.0, 1.0]) # the action space for CartPole-v1
        self.raw_efe = 0 # the output of the AcT planning prediction
        self.predictive_EFE = 0 # The temporally-discounted EFE prediction.
        self.pred_mean_state = pred_mean_state # predicted mean of state distributon: parameterizing the state belief
        self.pred_var_state = pred_var_state # predicted state variance: parameterizing the state belief
        self.visit_count = 0 # the number of times this node has been visited
        self.depth = 0 # the depth of the node from the root. Any time a new node is created, must specify its depth?
        self.parent = parent # this node's parent node
        self.children = [] # this node's children
        self.action_at_tau_minus_one = action_at_tau_minus_one # the action that led to the visitation of the present node
        self.action_posterior_belief = self.softmax(self.action_space)
        self.used_actions = [] # a list of all actions that HAVE been used to transition from this node to a subsequent node.
        self.name = name

    def softmax(self, x):
        sm = nn.Softmax(dim = 0)

        return sm(x)

    def __iter__(self):

        return iter(self.children)

    def __str__(self):

        return (
            f"id = {str(self.name)},\n"
            f"depth = {str(self.depth)},\n"
            f"prev_action = {str(self.action_at_tau_minus_one)},\n"
            f"used_actions = {str(self.used_actions)},\n"
            f"raw_efe = {str(self.raw_efe)},\n"
            f"predictive_EFE = {str(self.predictive_EFE)},\n"
        )


class Agent():

    def __init__(self, argv):

        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cuda:0')

        self.set_parameters(argv) # Set parameters

        # define the exploration factor, 𝜅𝑝:
        self.exploration_factor = 1.0

        # the number of timesteps to plan into the future
        self.planning_horizon = 1

        # number of times to call the AcT procedure - for debug only
        self.number_AcT_procedure_execs = 2

        self.obs_shape = self.env.observation_space.shape # The shape of observations
        self.obs_size = np.prod(self.obs_shape) # The size of the observation
        self.n_actions = self.env.action_space.n # The number of actions available to the agent
        # self.all_actions = np.array([0, 1]) 
        self.all_actions = [0, 1]

        # Initialize the networks:
        self.generative_transition_net = MVGaussianModel(self.obs_size+1, self.obs_size, self.n_hidden_gen_trans, lr=self.lr_gen_trans, device=self.device, model = 'gen_trans')
        self.generative_observation_net = MVGaussianModel(self.obs_size, self.obs_size, self.n_hidden_gen_obs, lr=self.lr_gen_obs, device=self.device, model = 'gen_obs')
        self.variational_transition_net = MVGaussianModel(1, self.obs_size, self.n_hidden_var_trans, lr=self.lr_var_trans, device=self.device, model = 'var_trans')

        if self.load_network: # If true: load the networks given paths

            self.generative_transition_net.load_state_dict(torch.load(self.network_load_path.format("trans")))
            self.generative_transition_net.eval()

            self.generative_observation_net.load_state_dict(torch.load(self.network_load_path.format("obs")))
            self.generative_observation_net.eval()

            self.variational_transition_net.load_state_dict(torch.load(self.network_load_path.format("var_trans")))
            self.variational_transition_net.eval()

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
            'env':'CartPole-v1', 'n_episodes':5000, # OG VERSION FOR CARTPOLE-V1
            'n_hidden_gen_trans':64, 'lr_gen_trans':1e-3,
            'n_hidden_var_trans':64, 'lr_var_trans':1e-3,
            'n_hidden_gen_obs':64, 'lr_gen_obs': 1e-3,
            'n_hidden_val':64, 'lr_val':1e-4,
            'memory_capacity':65536, 'batch_size':64, 'freeze_period':25,
            'Beta':0.99, 'gamma':1.00,
            'print_timer':100,
            'keep_log':True, 'log_path':"logs/ai_mdp_log{}.txt", 'log_save_timer':10,
            'save_results':True, 'results_path':"results/ai_mdp_results{}.npz", 'results_save_timer':500,
            'save_network':True, 'network_save_path':"networks/ai_mdp_{}net{}.pth", 'network_save_timer':500,
            'load_network':False, 'network_load_path':"networks/ai_mdp_{}net_rX.pth",
            # 'rho': 1.0, 'delta': 0.95, 'd': 4, 'epsilon': 0.86, 'k_p': 1.0 # results in temporal planning horizon = 3  
            'rho': 1.0, 'delta': 0.5, 'd': 4, 'epsilon': 1.0, 'k_p': 1.0 # results in temporal planning horizon = 1 (for myopic planning)  
        }

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
        # self.env = gym.make(custom_parameters['env'], render_mode = "human")

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
        sm = nn.Softmax(dim = 0)
        return sm(x)

    def kl_divergence_diag_cov_gaussian(self, mu1, sigma1_sq, mu2, sigma2_sq):
        '''
        Returns the average KL divergence for batched inputs. 
        In case of scaler inputs, simply returns the KL divergence for these two distributions
        '''

        avg_kl_div = 0

        for i in range(len(mu1)):

            kl_div_i = 0.5 * torch.sum(
                (torch.sqrt(sigma2_sq[i]) / torch.sqrt(sigma1_sq[i])) + ((mu1[i] - mu2[i])**2 / sigma1_sq[i]) - 1 + torch.log(torch.sqrt(sigma1_sq[i]) / torch.sqrt(sigma2_sq[i]))
            )

            avg_kl_div += kl_div_i

        # avg_kl_div = avg_kl_div / len(mu1)
        avg_kl_div = avg_kl_div.mean()


        return avg_kl_div

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

    def diag_cov_gaussian_log_li(self, predicted_cur_obs_mean, predicted_cur_obs_var, observation_samples, D):
        '''
        Returns the average log likelihood for batched inputs. 
        In case of scaler inputs, simply returns the log likelihood for the istribution.
        '''

        avg_ll = 0

        for i in range(len(predicted_cur_obs_mean)):

            ll_i = 0.5 * torch.sum(
                torch.log(predicted_cur_obs_var[i]) + (((observation_samples[i] - predicted_cur_obs_mean[i]) ** 2) / predicted_cur_obs_var[i]) + D * torch.log(torch.tensor(2 * np.pi))
            )

            avg_ll += ll_i

        # avg_ll = avg_ll / len(predicted_cur_obs_mean)
        avg_ll = avg_ll.mean()

        return avg_ll

    def diagonal_gaussian_entropy(self, sigma_squared, D):

        log_det_sigma = torch.log(torch.sum(sigma_squared))
        entropy = (D / 2) * (1 + np.log(2 * np.pi)) + 0.5 * log_det_sigma

        return entropy

    def calculate_approximate_EFE(self, mean_state_phi, var_state_phi, mean_obs_xi, var_obs_xi):
        '''
        var_state_phi must be a log var 
        '''

        with torch.no_grad():

            # 1. Construct the state prior - this is: [0.0, 0.0, 0.0, 0.0] 

            # 1.1 Define the mean vector over preffered hidden states:
            mean_x = 0.0
            mean_x_dot = 0.0
            mean_theta = 0.0
            mean_theta_dot = 0.0
            mean_state_prior = torch.tensor([mean_x, mean_x_dot, mean_theta, mean_theta_dot])

            # 1.2 Define the (diagonal) covariance matrix over preffered hidden states:
            var_x = 0.1  # Adjust as needed for x
            var_x_dot = 0.1  # Adjust as needed for x_dot
            var_theta = 0.01  # Small variance for theta to make it highly peaked at 0
            var_theta_dot = 0.01  # Small variance for theta_dot to make it highly peaked at 0

            # 1.3 Create the "covariance matrix" - array of diagonal entries of the covariance matrix: 
            var_state_prior = torch.tensor([var_x, var_x_dot, var_theta, var_theta_dot])

            # 2. Calculate the KL divergence between Q(s_t | pi) and the state prior P(s_t | C):
            state_divergence = self.kl_divergence_diag_cov_gaussian(
                mean_state_phi, var_state_phi,
                mean_state_prior, var_state_prior
            )

            # Generate the raparamed state sample:
            hidden_state_sample = self.variational_transition_net.reparameterize(mean_state_phi, var_state_phi)

            # get the associated predicted observation:
            mean_observation, var_observation = self.generative_observation_net(hidden_state_sample)

            # Generate the reparamed observation samples:
            observation_sample = self.generative_observation_net.reparameterize(mean_observation, var_observation)

            # 3. Calculate the entropy of the observation model:
            expected_entropy = self.diagonal_gaussian_entropy(
                sigma_squared = var_observation,
                D = 4
            )

            # 4. Calculate the expected free energy (1 sample version)
            expected_free_energy = state_divergence + (1/self.rho) * expected_entropy

            # print(f"\nEFE - state_divergence: {state_divergence}")
            # print(f"EFE - expected_entropy: {expected_entropy}")
            # print(f"EFE - expected_free_energy: {expected_free_energy}\n")


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

        # At time t0 predict the state at time t1 - using the variational model:
        pred_mean_state_batch_t1, pred_var_state_batch_t1 = self.variational_transition_net(action_batch_t0)
        # pred_var_state_batch_t1 = torch.exp(pred_var_state_batch_t1)

        return (
            obs_batch_t0, obs_batch_t1, action_batch_t0,
            action_batch_t1, reward_batch_t1, terminated_batch_t2, truncated_batch_t2,
            pred_mean_state_batch_t1, pred_var_state_batch_t1
        )


    def select_action_AcT(self, node):

        action_probabilities = node.action_posterior_belief

        np_action_probabilities = action_probabilities.numpy()

        action_indices = np.arange(len(action_probabilities))

        # chosen_action_index = np.random.choice(action_indices, p = action_probabilities) # ??
        chosen_action_index = np.random.choice(action_indices, p = np_action_probabilities)

        chosen_action = self.all_actions[chosen_action_index]

        return chosen_action


    # def tree_policy_AcT(self, node, B):
    def tree_policy_AcT(self, node):

        while not self.node_is_terminal_leaf_AcT(node):

            # print("\nnode not terminal")
            if not self.node_fully_expanded_AcT(node):
                # print("node NOT fully expanded\n")
                node = self.expand_AcT(node)
                global unique_node_id
                node.name = str(unique_node_id)
                unique_node_id += 1

            else:

                node = self.variational_inference_AcT(node)

        return node

    def get_actions_AcT(self, node):

        unused_actions = []

        for action in self.all_actions:

            if action not in node.used_actions:

                unused_actions.append(action)

        if len(unused_actions) == 0:

            # All actions have been used, so we simply return the best known action
            sorted_children = sorted(node.children, key = lambda child: child.predictive_EFE)
            action_with_minimum_EFE = sorted_children[0].action_at_tau_minus_one

            return action_with_minimum_EFE

        else:

            return unused_actions

    def expand_AcT(self, node):

        # perform an unused action:
        unused_actions = self.get_actions_AcT(node)

        # rnd_idx = torch.randint(0, len(unused_actions), (1,))
        # a_prime_scalar = unused_actions[rnd_idx].item()

        a_prime_scalar = random.choice(unused_actions)

        a_prime = torch.tensor([a_prime_scalar], dtype = torch.int64, device = self.device)

        # node.used_actions.append(a_prime)
        node.used_actions.append(a_prime.item())

        # At time t0 predict the state belief at t1, after performing action a_prime in state node.pred_mean_state:
        # pred_mean_cur_state_and_cur_action = torch.cat((node.pred_mean_state, a_prime), dim = 0) # node.pred_mean_state is now sufficient stats of multivariate gaussian 
        # mean_next_state_theta, var_next_state_theta = self.generative_transition_net(pred_mean_cur_state_and_cur_action) # WRONG! SHOULD BE VARIATIONAL MODEL!!!!!!

        mean_next_state_phi, var_next_state_phi = self.variational_transition_net(a_prime.float()) # action_batch_t1.float()
        var_next_state_phi = torch.exp(var_next_state_phi)

        # At time t1 predict the observation given the predicted state at time t1:
        mean_next_obs_xi, var_next_obs_xi = self.generative_observation_net(mean_next_state_phi)
        var_next_obs_xi = torch.exp(var_next_obs_xi)

        # instantiate a child node as a consequence of performing action a_prime
        child_node = Node()
        child_node.parent = node
        child_node.depth = node.depth + 1
        child_node.pred_mean_state = mean_next_state_phi
        child_node.pred_var_state = var_next_state_phi
        child_node.action_at_tau_minus_one = a_prime # the action that led to the visitation of the present node

        # Calculate the approximate Expected Free Energy for the predicted time step - t1:
        raw_efe = self.calculate_approximate_EFE(mean_next_state_phi, var_next_state_phi, mean_next_obs_xi, var_next_obs_xi)
        # store the raw efe as intermediate stage in comuting predictive efe
        child_node.raw_efe = raw_efe

        # finally, add the child node to the node's children
        node.children.append(child_node)

        return child_node

    def node_is_terminal_leaf_AcT(self, node):

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

    def node_fully_expanded_AcT(self, node):
        '''
        Returns True iff the input node has exhausted the 
        action space in making transitions between itself and its children.
        '''

        if len(node.children) == len(self.all_actions):
            return True
        else:
            return False

    def active_inferece_tree_search(self, initial_state_belief_mean, initial_state_belief_var, delta, epsilon):

        global unique_node_id

        # Initializing the planning tree
        root_node = Node(
            pred_mean_state = initial_state_belief_mean,
            pred_var_state = initial_state_belief_var,
            action_at_tau_minus_one = None, 
            parent = None,
            name = str(unique_node_id)
        )

        unique_node_id += 1

        # Begin AcT planning - hardcoded for 1-ply "planning" for the bootstrap phase
        for t in range(1, self.number_AcT_procedure_execs + 1):

            # treepolicy for 1-ply planning
            focal_node = self.tree_policy_AcT(root_node)

            # Evaluate the expanded node
            g_delta = self.evaluate_AcT(focal_node, delta)

            # Perform path integration
            self.path_integrate_AcT(focal_node, g_delta)

        return root_node

    # def calculate_free_energy_loss(self, pred_mean_state_batch_t1, action_batch_t1):

    #     # generative transition net inputs:
    #     pred_mean_cur_state_and_cur_action = torch.cat((pred_mean_state_batch_t1.detach(), action_batch_t1.float()), dim=1)

    #     # variational transition net inputs:
    #     current_action = action_batch_t1.float() # must cast to float

    #     # Forward pass through the networks:
    #     generative_next_state_distribution_mean, generative_next_state_distribution_var = self.generative_transition_net(pred_mean_cur_state_and_cur_action)  # predicted next state prob (gen)
    #     # generative_next_state_distribution_var = torch.exp(generative_next_state_distribution_var)

    #     variational_next_state_distribution_mean, variational_next_state_distribution_var = self.variational_transition_net(current_action) # predicted next state prob (var)
    #     # variational_next_state_distribution_var = torch.exp(variational_next_state_distribution_var)

    #     generative_cur_observation_distribution_mean, generative_cur_observation_distribution_var = self.generative_observation_net(pred_mean_state_batch_t1) # predicted current obs (gen)
    #     # generative_cur_observation_distribution_var = torch.exp(generative_cur_observation_distribution_var)

    #     # Calculate the predicted state, KL divergence between the generative and variational models
    #     kl_divergence = self.kl_divergence_diag_cov_gaussian(
    #         variational_next_state_distribution_mean, variational_next_state_distribution_var, # q_\phi
    #         generative_next_state_distribution_mean, generative_next_state_distribution_var # p_\theta
    #     )

    #     # Use the reparameterization trick to sample from the variational state distribution
    #     reparamed_hidden_state_samples = self.generative_transition_net.reparameterize(generative_next_state_distribution_mean, generative_next_state_distribution_var)
    #     # reparamed_hidden_state_samples = self.generative_transition_net.reparameterize(generative_next_state_distribution_mean, generative_next_state_distribution_log_var) # keept as logvar 

    #     # predict the corresponding observations for the 'reparamed_hidden_state_samples':
    #     generative_observation_mean, generative_observation_var = self.generative_observation_net(reparamed_hidden_state_samples)
    #     # generative_observation_mean, generative_observation_log_var = self.generative_observation_net(reparamed_hidden_state_samples)
    #     generative_observation_var = torch.exp(generative_observation_log_var)

    #     # Use the reparameterization trick to sample from the generative observation model:
    #     reparamed_observation_samples = self.generative_observation_net.reparameterize(generative_observation_mean, generative_observation_var)
    #     # reparamed_observation_samples = self.generative_observation_net.reparameterize(generative_observation_mean, generative_observation_log_var)

    #     # Calculate the negative log likelihood of the observation
    #     avg_log_likelihood = self.diag_cov_gaussian_log_li(
    #         predicted_cur_obs_mean = generative_cur_observation_distribution_mean, 
    #         predicted_cur_obs_var = generative_cur_observation_distribution_var, 
    #         observation_samples = reparamed_observation_samples,
    #         D = 4
    #     )

    #     # Compute the final loss
    #     loss = kl_divergence - avg_log_likelihood

    #     # print(f"\nVFE - kl_divergence: {kl_divergence}")
    #     # print(f"VFE - avg_log_likelihood: {avg_log_likelihood}")
    #     # print(f"VFE - loss: {loss}\n")

    #     global losses
    #     global kl_divs
    #     global avg_log_likelis

    #     losses.append(loss.detach().numpy())
    #     kl_divs.append(kl_divergence.detach().numpy())
    #     avg_log_likelis.append(avg_log_likelihood.detach().numpy())

    #     # breakpoint()

    #     return loss, kl_divergence, avg_log_likelihood

    def calculate_free_energy_loss(self, pred_mean_state_batch_t1, action_batch_t1):

        # generative transition net inputs - for batched inputs:
        # pred_mean_cur_state_and_cur_action = torch.cat((pred_mean_state_batch_t1.detach(), action_batch_t1.float()), dim = 1)

        # for non-batched inputs only:
        pred_mean_cur_state_and_cur_action = torch.cat((pred_mean_state_batch_t1.detach(), action_batch_t1.float()), dim = 0)

        # variational transition net inputs:
        current_action = action_batch_t1.float() # must cast to float

        # Forward pass through the networks:
        generative_next_state_distribution_mean, generative_next_state_distribution_log_var = self.generative_transition_net(pred_mean_cur_state_and_cur_action)  # predicted next state prob (gen)
        generative_next_state_distribution_var = torch.exp(generative_next_state_distribution_log_var)

        variational_next_state_distribution_mean, variational_next_state_distribution_log_var = self.variational_transition_net(current_action) # predicted next state prob (var)
        variational_next_state_distribution_var = torch.exp(variational_next_state_distribution_log_var)

        generative_cur_observation_distribution_mean, generative_cur_observation_distribution_log_var = self.generative_observation_net(pred_mean_state_batch_t1) # predicted current obs (gen)
        generative_cur_observation_distribution_var = torch.exp(generative_cur_observation_distribution_log_var)

        # Calculate the predicted state, KL divergence between the generative and variational models
        kl_divergence = self.kl_divergence_diag_cov_gaussian(
            variational_next_state_distribution_mean, variational_next_state_distribution_var, # q_\phi
            generative_next_state_distribution_mean, generative_next_state_distribution_var # p_\theta
        )

        # Use the reparameterization trick to sample from the variational state distribution
        reparamed_hidden_state_samples = self.generative_transition_net.reparameterize(generative_next_state_distribution_mean, generative_next_state_distribution_log_var)
        # reparamed_hidden_state_samples = self.generative_transition_net.reparameterize(generative_next_state_distribution_mean, generative_next_state_distribution_log_var) # keept as logvar 

        # predict the corresponding observations for the 'reparamed_hidden_state_samples':
        generative_observation_mean, generative_observation_log_var = self.generative_observation_net(reparamed_hidden_state_samples)
        # generative_observation_mean, generative_observation_log_var = self.generative_observation_net(reparamed_hidden_state_samples)
        # generative_observation_var = torch.exp(generative_observation_log_var)

        # Use the reparameterization trick to sample from the generative observation model:
        reparamed_observation_samples = self.generative_observation_net.reparameterize(generative_observation_mean, generative_observation_log_var)
        # reparamed_observation_samples = self.generative_observation_net.reparameterize(generative_observation_mean, generative_observation_log_var)

        # Calculate the negative log likelihood of the observation
        avg_log_likelihood = self.diag_cov_gaussian_log_li(
            predicted_cur_obs_mean = generative_cur_observation_distribution_mean, 
            predicted_cur_obs_var = generative_cur_observation_distribution_var, 
            observation_samples = reparamed_observation_samples,
            D = 4
        )

        # Compute the final loss
        loss = kl_divergence - avg_log_likelihood

        # print(f"\nVFE - kl_divergence: {kl_divergence}")
        # print(f"VFE - avg_log_likelihood: {avg_log_likelihood}")
        # print(f"VFE - loss: {loss}\n")

        global losses
        global kl_divs
        global avg_log_likelis

        losses.append(loss.cpu().detach().numpy())
        kl_divs.append(kl_divergence.cpu().detach().numpy())
        avg_log_likelis.append(avg_log_likelihood.cpu().detach().numpy())

        # breakpoint()

        return loss, kl_divergence, avg_log_likelihood


    # def learn(self):

    #     # If there are not enough transitions stored in memory, return:
    #     if self.memory.push_count - self.max_n_indices*2 < self.batch_size:
    #         free_energy_loss = 0.0
    #         kl_divergence = 0.0
    #         avg_log_likelihood = 0.0
    #         return free_energy_loss, kl_divergence, avg_log_likelihood

    #     # Retrieve transition data in mini batches:
    #     (
    #         obs_batch_t0, obs_batch_t1, action_batch_t0,
    #         action_batch_t1, reward_batch_t1, terminated_batch_t2, truncated_batch_t2,
    #         pred_mean_state_batch_t1, pred_var_state_batch_t1
    #     ) = self.get_mini_batches_AcT()

    #     # Compute the free energy loss for the neural networks:
    #     free_energy_loss, kl_divergence, avg_log_likelihood = self.calculate_free_energy_loss(pred_mean_state_batch_t1, action_batch_t1)

    #     # print(f"free_energy_loss: {free_energy_loss}")

    #     # Reset the gradients:
    #     self.generative_transition_net.optimizer.zero_grad()
    #     self.variational_transition_net.zero_grad()
    #     self.generative_observation_net.optimizer.zero_grad()

    #     # backpropagate 
    #     free_energy_loss.backward()

    #     # Perform gradient descent:
    #     self.generative_transition_net.optimizer.step()
    #     self.variational_transition_net.optimizer.step()
    #     self.generative_observation_net.optimizer.step()

    #     return free_energy_loss, kl_divergence, avg_log_likelihood

    def learn(self, action, observation):
        '''
        For use without minibatches
        Require:
            * action_batch_t1
            * pred_mean_state_batch_t1
        '''

        # # Retrieve transition data in mini batches:
        # (
        #     obs_batch_t0, obs_batch_t1, action_batch_t0,
        #     action_batch_t1, reward_batch_t1, terminated_batch_t2, truncated_batch_t2,
        #     pred_mean_state_batch_t1, pred_var_state_batch_t1
        # ) = self.get_mini_batches_AcT()

        # Compute the free energy loss for the neural networks:
        # free_energy_loss, kl_divergence, avg_log_likelihood = self.calculate_free_energy_loss(pred_mean_state_batch_t1, action_batch_t1)
        free_energy_loss, kl_divergence, avg_log_likelihood = self.calculate_free_energy_loss(observation, action)

        # print(f"free_energy_loss: {free_energy_loss}")

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

        return free_energy_loss, kl_divergence, avg_log_likelihood


    def train(self):

        filename = f"Deep_AIF_MDP_Cart_Pole_v1"
        figure_file = f"plots/{filename}.png"

        msg = "Environment is: {}\nTraining started at {}".format(self.env.unwrapped.spec.id, datetime.datetime.now())
        print(msg)
        print(f"\n----------------- Deep Active Inference Tree Search -----------------\n")
        if self.keep_log:
            self.record.write(msg+"\n")

        results = []
        avg_avg_lls = []
        avg_kls = []
        avg_fes = []

        if torch.cuda.is_available():
            print("CUDA is available")
        else:
            print("CUDA is NOT available")
        print(f"self.device: {self.device}")
        print(f"Playing {self.n_episodes} episodes")

        # Define the standard deviation of the Gaussian noise
        noise_std = 0.05

        # for ith_episode in range(self.n_episodes):
        for ith_episode in range(500):

            total_reward = 0
            obs, info = self.env.reset()
            noisy_obs = obs + noise_std * np.random.randn(*obs.shape)

            terminated = False
            truncated = False
            done = terminated or truncated

            reward = 0

            avg_avg_ll = 0
            kl_div = 0
            fe_loss = 0

            num_iters = 0

            while not done:

                # Assign the initial hidden state belief's sufficient statistics:
                state_belief_mean = torch.tensor([noisy_obs[0], noisy_obs[1], noisy_obs[2], noisy_obs[3]], dtype=torch.float32, device=self.device)
                state_belief_var = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32, device=self.device)

                # ACTIVE INFERENCE TREE SEARCH: Do Active Inference Tree Search:
                root_node = self.active_inferece_tree_search(
                    state_belief_mean, 
                    state_belief_var,
                    self.delta, 
                    self.epsilon
                )

                # self.env.render()
                draw_tree(root_node)
                breakpoint()

                # cast the noisy obs to tensor
                noisy_obs = torch.tensor(noisy_obs, dtype = torch.float32, device = self.device)

                # ACTIVE INFERENCE TREE SEARCH: Select the best action based on the updated tree
                action = self.select_action_AcT(root_node)

                # RANDOM ACTION SELECTION FOR INITIAL TRAINING
                # action = self.env.action_space.sample()

                action = torch.tensor([action], dtype = torch.int64, device = self.device) # cast action back to tensor

                # Push the noisy tuple to self.memory
                # self.memory.push(noisy_obs, action, reward, terminated, truncated)

                # ACTIVE INFERENCE TREE SEARCH: Execute the selected action and transition to a new state - AcT:
                # obs, reward, terminated, truncated, _  = self.env.step(action)
                obs, reward, terminated, truncated, _  = self.env.step(action[0].item())

                # update observation
                noisy_obs = obs + noise_std * np.random.randn(*obs.shape)
                noisy_obs = torch.tensor(noisy_obs, dtype = torch.float32, device = self.device)

                # increment the per-episode reward:
                total_reward += reward

                # learn
                # free_energy_loss, kl_divergence, avg_log_likelihood = self.learn()
                free_energy_loss, kl_divergence, avg_log_likelihood = self.learn(action, noisy_obs)

                # print(f"\nhidden state: {obs}")
                # print(f"noisy_obs: {noisy_obs}")
                # print(f"action: {action}")
                # print(f"total_reward: {total_reward}\n")

                # print(f"\ntype(free_energy_loss): {type(free_energy_loss)}")
                # print(f"type(kl_divergence): {type(kl_divergence)}")
                # print(f"type(avg_log_likelihood): {type(avg_log_likelihood)}\n")

                num_iters += 1

                avg_avg_ll += avg_log_likelihood.cpu().detach().item()
                kl_div += kl_divergence.cpu().detach().item()
                fe_loss += free_energy_loss.cpu().detach().item()

                # print(f"\ntype(avg_avg_ll): {type(avg_avg_ll)}")
                # print(f"type(kl_div): {type(kl_div)}")
                # print(f"type(fe_loss): {type(fe_loss)}\n")

                # breakpoint()


                if terminated or truncated:

                    done = True

                    noisy_obs = torch.tensor(noisy_obs, dtype = torch.float32, device = self.device)

                    print(f"\nfree_energy_loss: {free_energy_loss}")
                    print(f"kl_divergence: {kl_divergence}")
                    print(f"avg_log_likelihood: {avg_log_likelihood}")
                    print(f"total_reward: {total_reward}\n")

                    # self.memory.push(noisy_obs, -99, -99, terminated, truncated)

            results.append(total_reward)
            avg_avg_lls.append(avg_avg_ll)
            avg_kls.append(kl_div)
            avg_fes.append(fe_loss)

            # Print and keep a (.txt) record of stuff
            if ith_episode > 0 and ith_episode % self.print_timer == 0:

                avg_reward = np.mean(results)
                # avg_avg_ll = np.mean(avg_avg_lls)
                # avg_kl = np.mean(avg_kls)
                # avg_fe = np.mean(avg_fes)

                last_x = np.mean(results[-self.print_timer:])
                # msg = "Episodes: {:4d}, avg score: {:3.2f}, over last {:d}: {:3.2f}, free_energy_loss: {:3.2f}, kl_divergence: {:3.2f}, avg_log_likelihood: {:3.2f}".format(ith_episode, avg_reward, self.print_timer, last_x, avg_fe, avg_kl, avg_avg_ll)
                # print(msg)
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

                torch.save(self.generative_observation_net.state_dict(), "networks/intermediary/intermediary_obsnet{}_{:d}.pth".format(self.run_id, ith_episode))

        self.env.close()

        # If enabled, save the results and the network (state_dict)
        if self.save_results:
            np.savez("results/intermediary/intermediary_results{}_end".format(self.run_id), np.array(results))
            np.savez(self.results_path, np.array(results))
        if self.save_network:
            torch.save(self.generative_transition_net.state_dict(), "networks/intermediary/intermediary_transnet{}_end.pth".format(self.run_id))

            torch.save(self.generative_observation_net.state_dict(), "networks/intermediary/intermediary_obsnet{}_end.pth".format(self.run_id))

            torch.save(self.variational_transition_net.state_dict(), "networks/intermediary/intermediary_vartransnet{}_end.pth".format(self.run_id))

            torch.save(self.generative_transition_net.state_dict(), self.network_save_path.format("trans"))
            torch.save(self.variational_transition_net.state_dict(), self.network_save_path.format("vartrans"))

            torch.save(self.generative_observation_net.state_dict(), self.network_save_path.format("obs"))

        # global losses
        # global kl_divs
        # global avg_log_likelis

        # plt.plot(losses, label = 'Losses')
        # plt.plot(kl_divs, label = 'KL Divergence')
        plt.plot(avg_avg_lls, label = 'Avg Log Likelihood')

        # Add a legend
        plt.legend()

        # Show the plot
        plt.show()



        plt.plot(avg_kls, label = 'Avg KL Divergence')

        # Add a legend
        plt.legend()

        # Show the plot
        plt.show()



        plt.plot(avg_fes, label = 'Avg VFE')

        # Add a legend
        plt.legend()

        # Show the plot
        plt.show()



        plt.plot(results, label = 'Rewards')

        # Add a legend
        plt.legend()

        # Show the plot
        plt.show()



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