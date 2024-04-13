import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import datetime
import sys
import gym
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# the sensor uncertainty branch

'''
Probably can't use a VAE for the observation model, since we need the parameterise the mean and stdev of the obs model.
Perhaps it's best to go back to feedforward nets only.

Deal with the batch issue in gaussian_kl_div
'''

unique_node_id = 0


def plot_learning_curve(x, scores, figure_file, version):
    # Calculate the moving average and standard deviation
    running_avg = np.zeros(len(scores))
    stds = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
        stds[i] = np.std(scores[max(0, i - 100):(i + 1)])
    
    # Create a figure and plot the moving average
    plt.figure(figsize=(12, 6))
    plt.plot(x, running_avg, label='100 episode MAVG of scores', color='b')
    
    # Plot the standard deviation bands
    plt.fill_between(x, running_avg - stds, running_avg + stds, color='b', alpha=0.3, label='Standard Deviation')
    
    # plt.title(f"100 episode MAVG of scores with Standard Deviation: {version}")
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.legend()
    
    # Save or display the plot
    plt.show()
    plt.savefig(figure_file)

class ReplayMemory():
    
    # def __init__(self, capacity, obs_shape, device='cuda:0'):
    def __init__(self, capacity, obs_shape, device='cpu'):
        
        self.device=device
        
        self.capacity = capacity # The maximum number of items to be stored in memory
        
        self.obs_shape = obs_shape # the shape of observations
        
        # Initialize (empty) memory tensors
        self.obs_mem = torch.empty([capacity]+[dim for dim in self.obs_shape], dtype=torch.float32, device=self.device)
        self.action_mem = torch.empty(capacity, dtype=torch.int64, device=self.device)
        self.reward_mem = torch.empty(capacity, dtype=torch.int8, device=self.device)
        # self.done_mem = torch.empty(capacity, dtype=torch.int8, device=self.device)
        self.terminated_mem = torch.empty(capacity, dtype=torch.int8, device=self.device)
        self.truncated_mem = torch.empty(capacity, dtype=torch.int8, device=self.device)
        
        self.push_count = 0 # The number of times new data has been pushed to memory
        
    # def push(self, obs, action, reward, done):
    def push(self, obs, action, reward, terminated, truncated):
        
        # Store data in memory
        self.obs_mem[self.position()] = obs 
        self.action_mem[self.position()] = action
        self.reward_mem[self.position()] = reward
        # self.done_mem[self.position()] = done
        self.terminated_mem[self.position()] = terminated
        self.truncated_mem[self.position()] = truncated
        
        self.push_count += 1
    
    def position(self):
        # Returns the next position (index) to which data is pushed
        return self.push_count % self.capacity
    
    
    # def sample(self, obs_indices, action_indices, reward_indices, done_indices, max_n_indices, batch_size):
    def sample(self, obs_indices, action_indices, reward_indices, terminated_indices, truncated_indices, max_n_indices, batch_size):
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
        # done_batch = self.done_mem[np.array([index-done_indices for index in end_indices])]
        terminated_batch = self.terminated_mem[np.array([index-terminated_indices for index in end_indices])]
        truncated_batch = self.truncated_mem[np.array([index-truncated_indices for index in end_indices])]

        # print(f"\n\nend_indices.shape: {end_indices.shape}\n")

        # print(f"\nobs_batch.shape: {obs_batch.shape}")
        # print(f"action_batch.shape: {action_batch.shape}")
        # print(f"reward_batch.shape: {reward_batch.shape}")
        # print(f"terminated_batch.shape: {terminated_batch.shape}")
        # print(f"truncated_batch.shape: {truncated_batch.shape}\n\n")

        # breakpoint()
        
        # Correct for sampling over multiple episodes
        for i in range(len(end_indices)):
            index = end_indices[i]
            for j in range(1, max_n_indices):
                # if self.done_mem[index-j]:
                if self.terminated_mem[index-j] or self.truncated_mem[index-j]:
                    for k in range(len(obs_indices)):
                        if obs_indices[k] >= j:
                            obs_batch[i, k] = torch.zeros_like(self.obs_mem[0]) 
                    for k in range(len(action_indices)):
                        if action_indices[k] >= j:
                            action_batch[i, k] = torch.zeros_like(self.action_mem[0]) # Assigning action '0' might not be the best solution, perhaps as assigning at random, or adding an action for this specific case would be better
                    for k in range(len(reward_indices)):
                        if reward_indices[k] >= j:
                            reward_batch[i, k] = torch.zeros_like(self.reward_mem[0]) # Reward of 0 will probably not make sense for every environment
                    # for k in range(len(done_indices)):
                    #     if done_indices[k] >= j:
                    #         done_batch[i, k] = torch.zeros_like(self.done_mem[0]) 
                    for k in range(len(terminated_indices)):
                        if terminated_indices[k] >= j:
                            terminated_batch[i, k] = torch.zeros_like(self.terminated_mem[0])
                    for k in range(len(truncated_indices)):
                        if truncated_indices[k] >= j:
                            truncated_batch[i, k] = torch.zeros_like(self.truncated_mem[0]) 
                    break
                
        # return obs_batch, action_batch, reward_batch, done_batch
        return obs_batch, action_batch, reward_batch, terminated_batch, truncated_batch
        
    def get_last_n_obs(self, n):
        """ Get the last n observations stored in memory (of a single episode) """
        last_n_obs = torch.zeros([n]+[dim for dim in self.obs_shape], device=self.device)
        
        n = min(n, self.push_count)
        for i in range(1, n+1):
            if self.position() >= i:
                # if self.done_mem[self.position()-i]:
                if self.terminated_mem[self.position()-i] or self.truncated_mem[self.position()-i]:
                    return last_n_obs
                last_n_obs[-i] = self.obs_mem[self.position()-i]
            else:
                # if self.done_mem[-i+self.position()]:
                if self.terminated_mem[-i+self.position()] or self.truncated_mem[-i+self.position()]:
                    return last_n_obs
                last_n_obs[-i] = self.obs_mem[-i+self.position()]
        
        return last_n_obs
    
class Model(nn.Module):
    
    # def __init__(self, n_inputs, n_outputs, n_hidden=64, lr=1e-3, softmax=False, device='cuda:0'):
    def __init__(self, n_inputs, n_outputs, n_hidden=64, lr=1e-3, softmax=False, device='cpu'):
        super(Model, self).__init__()
        
        self.n_inputs = n_inputs # Number of inputs
        self.n_hidden = n_hidden # Number of hidden units
        self.n_outputs = n_outputs # Number of outputs
        self.softmax = softmax # If true apply a softmax function to the output
        
        self.fc1 = nn.Linear(self.n_inputs, self.n_hidden) # Hidden layer
        self.fc2 = nn.Linear(self.n_hidden, self.n_outputs) # Output layer
        
        self.optimizer = optim.Adam(self.parameters(), lr) # Adam optimizer
        
        self.device = device
        self.to(self.device)
    
    def forward(self, x):
        # Define the forward pass:
        h_relu = F.relu(self.fc1(x))
        y = self.fc2(h_relu)
        
        if self.softmax: # If true apply a softmax function to the output
            y = F.softmax(self.fc2(h_relu), dim=-1).clamp(min=1e-9, max=1-1e-9)
        
        return y

class MVGaussianModel(nn.Module):

    # def __init__(self, n_inputs, n_outputs, n_hidden, lr=1e-3, dropout_prob=0.0, device='cuda:0', model=None):
    # def __init__(self, n_inputs, n_outputs, n_hidden, lr=1e-3, device='cuda:0'):
    def __init__(self, n_inputs, n_outputs, n_hidden, lr=1e-3, device='cpu'):

        super(MVGaussianModel, self).__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden

        self.fc1 = nn.Linear(n_inputs, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)

        self.mean_fc = nn.Linear(n_hidden, n_outputs)
        self.stdev = nn.Linear(n_hidden, n_outputs)

        self.optimizer = optim.Adam(self.parameters(), lr)
        self.device = device
        self.to(self.device)

    def forward(self, x):

        x_1 = torch.relu(self.fc1(x))

        x_2 = torch.relu(self.fc2(x_1))

        mean = self.mean_fc(x_2)
        log_var = torch.log((self.stdev(x_2) ** 2))

        return mean, log_var

    def reparameterize(self, mean, var):
        std = torch.sqrt(var)
        epsilon = torch.randn_like(std)  # Sample from standard Gaussian
        sampled_value = mean + epsilon * std  # Reparameterization trick

        return sampled_value

class VAE(nn.Module):
    # def __init__(self, input_size, n_screens, n_latent_states, lr=1e-5, device='cuda:0'):
    def __init__(self, input_size, n_screens, n_latent_states, lr=1e-5, device='cpu'):
        super(VAE, self).__init__()

        self.device = device
        self.input_size = input_size
        self.n_screens = n_screens
        self.total_input_size = input_size * n_screens
        self.n_latent_states = n_latent_states

        # Encoder
        self.fc1 = nn.Linear(self.total_input_size, self.total_input_size // 2)
        self.fc2_mu = nn.Linear(self.total_input_size // 2, self.n_latent_states)
        self.fc2_logvar = nn.Linear(self.total_input_size // 2, self.n_latent_states)

        # Decoder
        self.fc3 = nn.Linear(self.n_latent_states, self.total_input_size // 2)
        self.fc4 = nn.Linear(self.total_input_size // 2, self.total_input_size)

        self.optimizer = optim.Adam(self.parameters(), lr)

        self.to(self.device)

    def encode(self, x):
        h1 = F.relu(self.fc1(x.view(-1, self.total_input_size)))
        mu, logvar = self.fc2_mu(h1), self.fc2_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        recon_x = torch.sigmoid(self.fc4(h3))
        return recon_x.view(-1, self.input_size, self.n_screens)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, batch=True):
        '''
        Returns the ELBO
        '''
        if batch: 
            BCE = F.binary_cross_entropy(recon_x.view(-1, self.total_input_size),
                                         x.view(-1, self.total_input_size), reduction='sum')
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

class Node:

    def __init__(self, pred_mean_state = None, action_at_tau_minus_one = None, parent = None, name = None):
        # self.action_space = np.array([0, 1]) # the action space for CartPole-v1
        self.action_space = torch.tensor([0.0, 1.0]) # the action space for CartPole-v1
        self.raw_efe = 0 # the output of the AcT planning prediction
        self.predictive_EFE = 0 # The temporally-discounted EFE prediction.
        self.pred_mean_state = pred_mean_state # predicted mean of state distributon: parameterizing the state belief
        # self.pred_var_state = pred_var_state # predicted state variance: parameterizing the state belief
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
            # f"id = {str(self.name)},\n"
            f"depth = {str(self.depth)},\n"
            f"prev_action = {str(self.action_at_tau_minus_one)},\n"
            f"used_actions = {str(self.used_actions)},\n"
            # f"raw_efe = {str(self.raw_efe)},\n"
            f"predictive_EFE = \n{str(self.predictive_EFE.item())[:8]}\n"
        )

class Agent():
    
    def __init__(self, argv):
        
        self.set_parameters(argv) # Set parameters
        
        # self.c = 3 # The number of (color) channels of observations
        # self.h = 37 # The height of observations (screens)
        # self.w = 85 # The width of observations (screens)
        self.obs_shape = (4,) 
        
        self.n_actions = self.env.action_space.n # The number of actions available to the agent
        
        self.freeze_cntr = 0 # Keeps track of when to (un)freeze the target network
        
        # Initialize the networks:
        # self.vae = VAE(self.n_screens, self.n_latent_states, lr=self.lr_vae, device=self.device)
        self.vae = VAE(self.obs_shape[0], self.n_screens, self.n_latent_states, lr=self.lr_vae, device=self.device)

        # self.transition_net = Model(self.n_latent_states*2+1, self.n_latent_states, self.n_hidden_trans, lr=self.lr_trans, device=self.device)
        self.transition_net = MVGaussianModel(self.n_latent_states*2+1, self.n_latent_states, self.n_hidden_trans, lr=self.lr_trans, device=self.device)

        self.policy_net = Model(self.n_latent_states*2, self.n_actions, self.n_hidden_pol, lr=self.lr_pol, softmax=True, device=self.device)
        self.value_net = Model(self.n_latent_states*2, self.n_actions, self.n_hidden_val, lr=self.lr_val, device=self.device)

        self.target_net = Model(self.n_latent_states*2, self.n_actions, self.n_hidden_val, lr=self.lr_val, device=self.device)
        self.target_net.load_state_dict(self.value_net.state_dict())
            
        if self.load_network: # If true: load the networks given paths
            self.vae.load_state_dict(torch.load(self.network_load_path.format("vae"), map_location=self.device))
            self.vae.eval()
            self.transition_net.load_state_dict(torch.load(self.network_load_path.format("trans"), map_location=self.device))
            self.transition_net.eval()
            self.policy_net.load_state_dict(torch.load(self.network_load_path.format("pol"), map_location=self.device))
            self.policy_net.eval()
            self.value_net.load_state_dict(torch.load(self.network_load_path.format("val"), map_location=self.device))
            self.value_net.eval()
        
        if self.load_pre_trained_vae: # If true: load a pre-trained VAE
            self.vae.load_state_dict(torch.load(self.pt_vae_load_path, map_location=self.device))
            self.vae.eval()
        
        # Initialize the replay memory
        self.memory = ReplayMemory(self.memory_capacity, self.obs_shape, device=self.device)
        
        # # Used to pre-process the observations (screens)        
        # self.resize = T.Compose([T.ToPILImage(),
        #             T.Resize(40, interpolation=Image.CUBIC),
        #             T.ToTensor()])
    
        # When sampling from memory at index i, obs_indices indicates that we want observations with indices i-obs_indices, works the same for the others
        self.obs_indices = [(self.n_screens+1)-i for i in range(self.n_screens+2)]
        self.action_indices = [2, 1]
        self.reward_indices = [1]
        self.terminated_indices = [0]
        self.truncated_indices = [0]
        # self.max_n_indices = max(max(self.obs_indices, self.action_indices, self.reward_indices, self.done_indices)) + 1
        self.max_n_indices = max(max(self.obs_indices, self.action_indices, self.reward_indices, self.terminated_indices, self.truncated_indices)) + 1
        
    def set_parameters(self, argv):
        
        # The default parameters
        default_parameters = {'run_id':"_rX", 'device':'cpu', # 'device':'cuda:0',
              'env':'CartPole-v1', 'n_episodes':2000, 
              'n_screens':4, 'n_latent_states':32, 'lr_vae':1e-5, 'alpha':25000,
              'n_hidden_trans':64, 'lr_trans':1e-3,
              'n_hidden_pol':64, 'lr_pol':1e-3,
              'n_hidden_val':64, 'lr_val':1e-4,
              'memory_capacity':65536, 'batch_size':32, 'freeze_period':25, 
              'Beta':0.99, 'gamma':12.00,
              'print_timer':100,
              'keep_log':True, 'log_path':"logs/ai_pomdp_log{}.txt", 'log_save_timer':10,
              'save_results':True, 'results_path':"results/ai_pomdp_results{}.npz", 'results_save_timer':500,
              'save_network':True, 'network_save_path':"networks/ai_pomdp_{}net{}.pth", 'network_save_timer':500,
              'load_network':False, 'network_load_path':"networks/ai_pomdp_{}net_rX.pth",
              # 'pre_train_vae':False, 'pt_vae_n_episodes':500, 'pt_vae_plot':False,
              'pre_train_vae':False, 'pt_vae_n_episodes':500, 'pt_vae_plot':False,
              'load_pre_trained_vae':False, 'pt_vae_load_path':"networks/pre_trained_vae/vae_n{}_end.pth"}
        # Possible commands:
            # python ai_pomdp_agent.py device=cuda:0
            # python ai_pomdp_agent.py device=cuda:0 load_pre_trained_vae=False pre_train_vae=True
        
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
        
        self.env = gym.make(custom_parameters['env'], render_mode='rgb_array') # The environment in which to train
        # self.env = gym.make(custom_parameters['env']) # The environment in which to train
        self.n_episodes = int(custom_parameters['n_episodes']) # The number of episodes for which to train
        
        # Set number of hidden nodes and learning rate for each network
        self.n_screens = int(custom_parameters['n_screens']) # The number of obervations (screens) that are passed to the VAE
        self.n_latent_states = int(custom_parameters['n_latent_states'])
        self.lr_vae = float(custom_parameters['lr_vae'])
        self.alpha = int(custom_parameters['alpha']) # Used to scale down the VAE's loss
        self.n_hidden_trans = int(custom_parameters['n_hidden_trans'])
        self.lr_trans = float(custom_parameters['lr_trans'])
        self.n_hidden_val = int(custom_parameters['n_hidden_val'])
        self.lr_val = float(custom_parameters['lr_val'])
        self.n_hidden_pol = int(custom_parameters['n_hidden_pol'])
        self.lr_pol = float(custom_parameters['lr_pol'])
        
        self.memory_capacity = int(custom_parameters['memory_capacity']) # The maximum number of items to be stored in memory
        self.batch_size = int(custom_parameters['batch_size']) # The mini-batch size
        self.freeze_period = int(custom_parameters['freeze_period']) # The number of time-steps the target network is frozen
        
        self.Beta = float(custom_parameters['Beta']) # The discount rate
        self.gamma = float(custom_parameters['gamma']) # A precision parameter
        
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
        
        self.pre_train_vae = interpret_boolean(custom_parameters['pre_train_vae']) # If true pre trains the vae
        self.pt_vae_n_episodes = custom_parameters['pt_vae_n_episodes'] # The amount of episodes for which to pre train the vae
        self.pt_vae_plot = interpret_boolean(custom_parameters['pt_vae_plot']) # If true plots stuff while training the vae
        
        self.load_pre_trained_vae = interpret_boolean(custom_parameters['load_pre_trained_vae']) # If true loads a pre trained vae
        self.pt_vae_load_path = custom_parameters['pt_vae_load_path'].format(self.n_latent_states) # The path from which to load the pre trained vae
        
        msg = "Default parameters:\n"+str(default_parameters)+"\n"+custom_parameter_msg
        print(msg)
        
        if self.keep_log: # If true: write a message to the log
            self.record = open(self.log_path, "a")
            self.record.write("\n\n-----------------------------------------------------------------\n")
            self.record.write("File opened at {}\n".format(datetime.datetime.now()))
            self.record.write(msg+"\n")

    def select_action_AcT(self, node):

        action_probabilities = node.action_posterior_belief

        np_action_probabilities = action_probabilities.numpy()

        action_indices = np.arange(len(action_probabilities))

        # chosen_action_index = np.random.choice(action_indices, p = action_probabilities) # ??
        chosen_action_index = np.random.choice(action_indices, p = np_action_probabilities)

        chosen_action = self.all_actions[chosen_action_index]

        return chosen_action

    def select_action_myopic_AcT(self, node):
        '''
        returns the action correspondng to the node with the minimum EFE for 1-ply planning.
        '''

        sorted_children = sorted(node.children, key = lambda child: child.predictive_EFE)
        action_with_minimum_EFE = sorted_children[0].action_at_tau_minus_one

        return action_with_minimum_EFE


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

    def monte_carlo_EFE(mu1, sigma1, mu2, sigma2, observation_model, N = 1):
        """
        Calculates the 'N' Monte Carlo approximation of the Expected Free Energy (EFE).
        sigma1, mu2 parameterise the distribution W.R.T to which the samples are drawn. 

        Parameters:
            mu1 (torch.Tensor): Mean of the first distribution.
            sigma1 (torch.Tensor): Diagonal standard deviations of the first distribution.
            mu2 (torch.Tensor): Mean of the second distribution.
            sigma2 (torch.Tensor): Diagonal standard deviations of the second distribution.
            N: The number of Monte Carlo samples to use when approximating the expected entropy. 

        Returns:
            torch.Tensor: Monte Carlo EFE approximation.
        """

        # calculate the KL divergenced:
        kl_div = gaussian_kl_div(mu1, sigma1, mu2, sigma2)

        # draw N samples from the first distribution:
        epsilon = torch.randn_like(mu1)
        
        # Reparameterization trick: transform the sample to match the target distribution
        sampled_latent_states = torch.tensor([(mu + epsilon * sigma) for i in range(N)])

        # get the predicted/reconstructed observation distribution
        mu_xi, sigma_xi = observation_model(sampled_latent_states) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        # compute the observation entropy:
        H_obs = Gaussian_entropy(sigma_xi)

        # Compute the Monte Carlo observation entropy:
        mc_obs_ent = (1/N)*torch.sum(H_obs)

        # compute the final approx EFE:
        mc_efe = kl_div - mc_obs_ent

        return mc_efe

    # OG VERSION:
    # def expand_AcT(self, node):

    #     # perform an unused action:
    #     unused_actions = self.get_actions_AcT(node)

    #     a_prime_scalar = random.choice(unused_actions)
    #     a_prime = torch.tensor([a_prime_scalar], dtype = torch.int64, device = self.device)
    #     node.used_actions.append(a_prime.item())

    #     # At time t0 predict the state belief at t1, after performing action a_prime in state node.pred_mean_state:
    #     mean_next_state_phi, var_next_state_phi = self.variational_transition_net(
    #         a_prime.float()
    #     ) # action_batch_t1.float()

    #     # At time t1 predict the observation given the predicted state at time t1:
    #     mean_next_obs_xi, var_next_obs_xi = self.generative_observation_net(
    #         mean_next_state_phi
    #     )

    #     # instantiate a child node as a consequence of performing action a_prime
    #     child_node = Node()
    #     child_node.parent = node
    #     child_node.depth = node.depth + 1
    #     child_node.pred_mean_state = mean_next_state_phi
    #     child_node.action_at_tau_minus_one = a_prime # the action that led to the visitation of the present node

    #     # Calculate the approximate Expected Free Energy for the predicted time step - t1:
    #     raw_efe = self.calculate_approximate_EFE(
    #         mean_next_state_phi, var_next_state_phi, 
    #         mean_next_obs_xi, var_next_obs_xi
    #     )

    #     # print(f"\nraw_efe: {raw_efe}\n")
    #     # breakpoint()

    #     # store the raw efe as intermediate stage in comuting predictive efe
    #     child_node.raw_efe = raw_efe

    #     # finally, add the child node to the node's children
    #     node.children.append(child_node)

    #     return child_node

    def expand_AcT(self, node):

        # perform an unused action:
        unused_actions = self.get_actions_AcT(node)

        a_prime_scalar = random.choice(unused_actions)
        a_prime = torch.tensor([a_prime_scalar], dtype = torch.int64, device = self.device)
        node.used_actions.append(a_prime.item())

        # At time t0 predict the state belief at t1, after performing action a_prime in state node.pred_mean_state:
        mean_next_state_theta, log_var_next_state_theta = self.vae.encode(
            a_prime.float()
        ) # action_batch_t1.float()

        # turn the log variance back intoa variance
        var_next_state_theta = torch.exp(var_next_state_theta)

        # Reparameterize the distribution over states for time t1
        z_batch_theta = self.vae.reparameterize(mean_next_state_theta, var_next_state_theta)

        # At time t1 predict the observation given the predicted state at time t1:
        recon_batch_nu = self.vae.decode(z_batch_theta)

        # instantiate a child node as a consequence of performing action a_prime
        child_node = Node()
        child_node.parent = node
        child_node.depth = node.depth + 1
        child_node.pred_mean_state = mean_next_state_theta
        child_node.action_at_tau_minus_one = a_prime # the action that led to the visitation of the present node

        # Calculate the approximate Expected Free Energy for the predicted time step - t1:
        raw_efe = self.calculate_approximate_EFE(
            mean_next_state_theta, var_next_state_theta, 
            mean_next_obs_xi, var_next_obs_xi
        )

        # print(f"\nraw_efe: {raw_efe}\n")
        # breakpoint()

        # store the raw efe as intermediate stage in comuting predictive efe
        child_node.raw_efe = raw_efe

        # finally, add the child node to the node's children
        node.children.append(child_node)

        return child_node

    def node_is_terminal_leaf_AcT(self, node):

        # return self.delta ** node.depth < self.epsilon
        # return node.depth == 1
        return node.depth == 2 # artificially limit the AcT tree depth to 2.
        # return node.depth == 3

    def update_precision(self, depth, alpha, beta):

        per_depth_precision = stats.gamma.rvs(alpha, scale=beta)

        if depth > 0:
            per_depth_precision *= depth

        return per_depth_precision

    # OG VERSION
    # def update_action_posterior(self, node, prior_belief_about_policy, precision_tau, EFE_tau):

    #     # # Placeholder values for demonstration (replace with actual values)
    #     # log_N_nu = torch.tensor(5.0)
    #     # N_nu_prime = torch.tensor(10.0)
    #     # G_tilde = torch.tensor([0.4, 21.2, 8.08, 717.3])

    #     # # Calculate E
    #     # E = torch.sqrt((2 * torch.log(N_nu)) / (N_nu_prime))

    #     # # Calculate the expression
    #     # result = torch.sigmoid(torch.log(E) - gamma * G_tilde)

    #     # compute the argument to the Boltzmann distribution over actions:
    #     action_dist_arg = torch.tensor(
    #         [(self.exploration_factor * np.log(policy_prior) - precision_tau * EFE_tau.cpu().detach().numpy()) \
    #         for policy_prior in prior_belief_about_policy]
    #     )

    #     # Construct the Boltzmann distribution over actions - posterior belief about actions (posterior action probabilities):
    #     action_probs = self.softmax(action_dist_arg)

    #     node.action_posterior_belief = action_probs

    #     return action_probs

    # def update_action_posterior(self, node, prior_belief_about_policy, precision_tau, EFE_tau):
    def update_action_posterior(self, node, precision_tau):
        '''
        node: the "child node" in the E prior distribution
        node.parent: the "current node" in the E prior distribution.
        '''

        N_nu = node.parent.visit_count
        N_nu_prime = node.visit_count
        G_tilde = node.predictive_EFE

        E_prior = torch.sqrt((2 * torch.log(N_nu)) / (N_nu_prime))

        action_posterior = torch.sigmoid(torch.log(E_prior) - precision_tau * G_tilde)

        # node.action_posterior_belief = action_posterior

        return action_posterior

    # OG VERSION
    # def variational_inference_AcT(self, node):

    #     # # Compute the policy prior - 𝐄:
    #     # prior_belief_about_policy = np.array(
    #     #     [math.sqrt((2 * math.log(node.visit_count)) / child_node.visit_count) \
    #     #     for child_node in node.children]
    #     # )
        
    #     # # PUSH E THROUGH A SOFTMAX TO MAKE IT A PROB DIST - TOTAL HACK 
    #     # prior_belief_about_policy = self.softmax(torch.tensor(prior_belief_about_policy, dtype = torch.float32, device = self.device))

    #     # Compute the precision 𝛾_𝜏: for the current time
    #     precision_tau = self.update_precision(
    #         depth = node.depth, alpha = 1, beta = 1
    #     )

    #     # Get delta^tau * G(𝜋_𝜏, 𝑣_𝜏):
    #     # EFE_tau = node.predictive_EFE 

    #     # Construct the Boltzmann distribution over actions - posterior belief about actions (posterior action probabilities):
    #     # action_probs = self.update_action_posterior(node, prior_belief_about_policy, precision_tau, EFE_tau)
    #     action_posterior = self.update_action_posterior(node, precision_tau)

    #     node.action_posterior_belief = action_posterior

    #     # sample an action from this Boltzmann distribution:
    #     selected_child_node = random.choices(node.children, weights = action_posterior)[0]

    #     return selected_child_node

    def variational_inference_AcT(self, node):

        # Compute the precision 𝛾_𝜏: for the current time
        precision_tau = self.update_precision(
            depth = node.depth, alpha = 1, beta = 1
        )

        action_posterior = self.update_action_posterior(node, precision_tau)

        node.action_posterior_belief = action_posterior

        # sample an action from this Boltzmann distribution:
        selected_child_node = random.choices(node.children, weights = action_posterior)[0]

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
            # node.predictive_EFE += (1 / node.visit_count) * (node.predictive_EFE - new_efe_value) ## OG
            node.predictive_EFE += (1 / node.visit_count) * (new_efe_value - node.predictive_EFE)
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

    def active_inferece_tree_search(self, initial_state_belief_mean, delta, epsilon):

        global unique_node_id

        # Initializing the planning tree
        root_node = Node(
            pred_mean_state = initial_state_belief_mean,
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

    def select_action(self, obs):
        with torch.no_grad():
            # Derive a distribution over states from the last n observations (vectors):
            prev_n_obs = self.memory.get_last_n_obs(self.n_screens - 1)

            # Add a batch dimension to the obs
            obs_reshaped = obs.unsqueeze(0)

            x = torch.cat((prev_n_obs, obs_reshaped), dim=0).view(1, -1)  # Flatten the input vectors

            state_mu, state_logvar = self.vae.encode(x)
            
            # Determine a distribution over actions given the current observation:
            x = torch.cat((state_mu, torch.exp(state_logvar)), dim=1)
            policy = self.policy_net(x)

            return torch.multinomial(policy, 1)

    def gaussian_kl_div(self, mu_1, sigma_sq_1, mu_2, sigma_sq_2):
        '''
        Calculates the KL Divergence between P(mu_1, sigma_sq_1) and Q(mu_2, sigma_sq_2)
        D_KL[P || Q], where P and Q are Univariate Gaussians.
        '''
        return (1/2)*(
            torch.log(sigma_sq_2 / sigma_sq_1) + \
            ((sigma_sq_1)/(sigma_sq_2)) + \
            ((mu_1 - mu_2) ** 2) / (sigma_sq_2) - 1
        )

    def Gaussian_entropy(mu, sigma):
        """
        Calculate entropy of the parameterised multivariate Gaussian distribution.

        Parameters:
            mu (torch.Tensor): Mean of the distribution.
            sigma (torch.Tensor): Diagonal standard deviations of the distribution.

        Returns:
            torch.Tensor: Gaussian Entropy.
        """
        k = mu1.size(-1)  # Dimensionality of the distributions

        log_term = torch.sum(torch.log(sigma))

        entropy = 0.5 * (k + k*torch.log(2*torch.tensor(np.pi)) + log_term)

        return entropy

    # # OG VERSION:
    # def get_mini_batches(self):

    #     # # Retrieve transition data in mini batches - OG VERSION
    #     # all_obs_batch, all_actions_batch, reward_batch_t1, terminated_batch_t2, truncated_batch_t2 = self.memory.sample(
    #     #         self.obs_indices, self.action_indices, self.reward_indices,
    #     #         self.terminated_indices, self.truncated_indices, self.max_n_indices, self.batch_size)

    #     # Retrieve transition data in mini batches
    #     all_obs_batch, all_actions_batch, reward_batch_t1, terminated_batch_t0, truncated_batch_t0 = self.memory.sample(
    #             self.obs_indices, self.action_indices, self.reward_indices,
    #             self.terminated_indices, self.truncated_indices, self.max_n_indices, self.batch_size)

    #     '''
    #     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #     PRETTY SURE THAT terminated_batch_t2, truncated_batch_t2 SHOULD BOTH BE T0 (CURRENT TIME STEP) !!!!!!!!!!!!!!
    #     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #     '''

    #     # print(f"\nall_obs_batch:\n{all_obs_batch}") 
    #     # print(f"\nall_actions_batch:\n{all_actions_batch}\n") 
    #     # print(f"\nreward_batch_t1:\n{reward_batch_t1}\n") 
    #     # print(f"terminated_batch_t2:\n{terminated_batch_t2}") 
    #     # print(f"truncated_batch_t2:\n{truncated_batch_t2}\n\n")

    #     # print(f"\nall_obs_batch:\n{all_obs_batch}\n") 

    #     # ACTIONS:
    #     print("\nall_actions_batch")
    #     for i, actions in enumerate(all_actions_batch): # i is the location index within the minibatch
    #         print(i, actions)
    #     print("\n")

    #     print("\nall_actions_batch - INDEXED")
    #     for t in range(all_actions_batch.shape[1]): # t is the time index  for each entry in all_actions_batch
    #         print(t, all_actions_batch[:, t].unsqueeze(1))
    #     print("\n")

    #     # Retrieve the agent's action history for time t0 and time t1
    #     action_batch_t0 = all_actions_batch[:, 0] # delete these, only for debug purposes - it's calculate below anyway.
    #     action_batch_t1 = all_actions_batch[:, 1] # delete these, only for debug purposes - it's calculate below anyway.

    #     print(f"\naction_batch_t0: {action_batch_t0}")
    #     print(f"action_batch_t1: {action_batch_t1}\n")

    #     breakpoint()
        
    #     # Retrieve a batch of observations for n_screens consecutive points in time
    #     obs_batch_t0 = all_obs_batch[:, 0:self.n_screens, :].view(self.batch_size, -1)
    #     obs_batch_t1 = all_obs_batch[:, 1:self.n_screens+1, :].view(self.batch_size, -1)
    #     obs_batch_t2 = all_obs_batch[:, 2:self.n_screens+2, :].view(self.batch_size, -1)
        
    #     # Retrieve a batch of distributions over states for n_screens consecutive points in time
    #     state_mu_batch_t0, state_logvar_batch_t0 = self.vae.encode(obs_batch_t0)
    #     state_mu_batch_t1, state_logvar_batch_t1 = self.vae.encode(obs_batch_t1)
    #     state_mu_batch_t2, state_logvar_batch_t2 = self.vae.encode(obs_batch_t2)
        
    #     # Combine the sufficient statistics (mean and variance) into a single vector
    #     state_batch_t0 = torch.cat((state_mu_batch_t0, torch.exp(state_logvar_batch_t0)), dim=1)
    #     state_batch_t1 = torch.cat((state_mu_batch_t1, torch.exp(state_logvar_batch_t1)), dim=1)
    #     state_batch_t2 = torch.cat((state_mu_batch_t2, torch.exp(state_logvar_batch_t2)), dim=1)
        
    #     # Reparameterize the distribution over states for time t1
    #     z_batch_t1 = self.vae.reparameterize(state_mu_batch_t1, state_logvar_batch_t1)
        
    #     # Retrieve the agent's action history for time t0 and time t1
    #     action_batch_t0 = all_actions_batch[:, 0].unsqueeze(1)
    #     action_batch_t1 = all_actions_batch[:, 1].unsqueeze(1)


        
    #     # At time t0 predict the state at time t1:
    #     X = torch.cat((state_batch_t0.detach(), action_batch_t0.float()), dim=1)
    #     pred_batch_mean_t0t1, pred_batch_logvar_t0t1 = self.transition_net(X)

    #     # Determine the prediction error wrt time t0-t1 using state KL Divergence:
    #     pred_error_batch_t0t1 = torch.sum(
    #         self.gaussian_kl_div(
    #             pred_batch_mean_t0t1, torch.exp(pred_batch_logvar_t0t1), 
    #             state_mu_batch_t1, torch.exp(state_logvar_batch_t1)
    #         ), dim = 1
    #     ).unsqueeze(1)

    #     # # OG VERSION
    #     # return (state_batch_t1, state_batch_t2, action_batch_t1,
    #     #         reward_batch_t1, terminated_batch_t2, truncated_batch_t2, pred_error_batch_t0t1,
    #     #         obs_batch_t1, state_mu_batch_t1,
    #     #         state_logvar_batch_t1, z_batch_t1)

    #     return (state_batch_t1, state_batch_t2, action_batch_t1,
    #             reward_batch_t1, terminated_batch_t0, truncated_batch_t0, pred_error_batch_t0t1,
    #             obs_batch_t1, state_mu_batch_t1,
    #             state_logvar_batch_t1, z_batch_t1)






    # # FAILSAFE WORKING VERSION - BEFORE CHANGING ACTION-BATCH-T0 TO T2:
    # def get_mini_batches(self):

    #     # # Retrieve transition data in mini batches - OG VERSION
    #     # all_obs_batch, all_actions_batch, reward_batch_t1, terminated_batch_t2, truncated_batch_t2 = self.memory.sample(
    #     #         self.obs_indices, self.action_indices, self.reward_indices,
    #     #         self.terminated_indices, self.truncated_indices, self.max_n_indices, self.batch_size)

    #     # Retrieve transition data in mini batches
    #     all_obs_batch, all_actions_batch, reward_batch_t1, terminated_batch_t0, truncated_batch_t0 = self.memory.sample(
    #             self.obs_indices, self.action_indices, self.reward_indices,
    #             self.terminated_indices, self.truncated_indices, self.max_n_indices, self.batch_size)

    #     # # ACTIONS:
    #     # print("\nall_actions_batch")
    #     # for i, actions in enumerate(all_actions_batch): # i is the location index within the minibatch
    #     #     print(i, actions)
    #     # print("\n")

    #     # print("\nall_actions_batch - INDEXED")
    #     # for t in range(all_actions_batch.shape[1]): # t is the time index  for each entry in all_actions_batch
    #     #     print(t, all_actions_batch[:, t].unsqueeze(1))
    #     # print("\n")

    #     # # # Retrieve the agent's action history for time t0 and time t1
    #     # # action_batch_t0 = all_actions_batch[:, 0] # delete these, only for debug purposes - it's calculate below anyway.
    #     # # action_batch_t1 = all_actions_batch[:, 1] # delete these, only for debug purposes - it's calculate below anyway.

    #     # # print(f"\naction_batch_t0: {action_batch_t0}")
    #     # # print(f"action_batch_t1: {action_batch_t1}\n")

    #     # # Retrieve the agent's action history for time t0 and time t1
    #     # action_batch_t2 = all_actions_batch[:, 0] # delete these, only for debug purposes - it's calculate below anyway.
    #     # action_batch_t1 = all_actions_batch[:, 1] # delete these, only for debug purposes - it's calculate below anyway.

    #     # print(f"\naction_batch_t2: {action_batch_t2}")
    #     # print(f"action_batch_t1: {action_batch_t1}\n")

    #     # breakpoint()
        
    #     # Retrieve a batch of observations for n_screens consecutive points in time
    #     obs_batch_t0 = all_obs_batch[:, 0:self.n_screens, :].view(self.batch_size, -1)
    #     obs_batch_t1 = all_obs_batch[:, 1:self.n_screens+1, :].view(self.batch_size, -1)
    #     obs_batch_t2 = all_obs_batch[:, 2:self.n_screens+2, :].view(self.batch_size, -1)
        
    #     # Retrieve a batch of distributions over states for n_screens consecutive points in time
    #     state_mu_batch_t0, state_logvar_batch_t0 = self.vae.encode(obs_batch_t0)
    #     state_mu_batch_t1, state_logvar_batch_t1 = self.vae.encode(obs_batch_t1)
    #     state_mu_batch_t2, state_logvar_batch_t2 = self.vae.encode(obs_batch_t2)
        
    #     # Combine the sufficient statistics (mean and variance) into a single vector
    #     state_batch_t0 = torch.cat((state_mu_batch_t0, torch.exp(state_logvar_batch_t0)), dim=1)
    #     state_batch_t1 = torch.cat((state_mu_batch_t1, torch.exp(state_logvar_batch_t1)), dim=1)
    #     state_batch_t2 = torch.cat((state_mu_batch_t2, torch.exp(state_logvar_batch_t2)), dim=1)
        
    #     # Reparameterize the distribution over states for time t1
    #     z_batch_t1 = self.vae.reparameterize(state_mu_batch_t1, state_logvar_batch_t1)
        
    #     # # OG action_batch_t0 naming
    #     # # Retrieve the agent's action history for time t0 and time t1
    #     # action_batch_t0 = all_actions_batch[:, 0].unsqueeze(1)
    #     # action_batch_t1 = all_actions_batch[:, 1].unsqueeze(1)

    #     # Retrieve the agent's action history for time t0 and time t1
    #     action_batch_t0 = all_actions_batch[:, 0].unsqueeze(1)
    #     action_batch_t1 = all_actions_batch[:, 1].unsqueeze(1)


    #     # At time t0 predict the state at time t1:
    #     X = torch.cat((state_batch_t0.detach(), action_batch_t0.float()), dim=1)
    #     pred_batch_mean_t0t1, pred_batch_logvar_t0t1 = self.transition_net(X)

    #     # Determine the prediction error wrt time t0-t1 using state KL Divergence:
    #     pred_error_batch_t0t1 = torch.sum(
    #         self.gaussian_kl_div(
    #             pred_batch_mean_t0t1, torch.exp(pred_batch_logvar_t0t1), 
    #             state_mu_batch_t1, torch.exp(state_logvar_batch_t1)
    #         ), dim = 1
    #     ).unsqueeze(1)

    #     # # OG VERSION
    #     # return (state_batch_t1, state_batch_t2, action_batch_t1,
    #     #         reward_batch_t1, terminated_batch_t2, truncated_batch_t2, pred_error_batch_t0t1,
    #     #         obs_batch_t1, state_mu_batch_t1,
    #     #         state_logvar_batch_t1, z_batch_t1)

    #     return (state_batch_t1, state_batch_t2, action_batch_t1,
    #             reward_batch_t1, terminated_batch_t0, truncated_batch_t0, pred_error_batch_t0t1,
    #             obs_batch_t1, state_mu_batch_t1,
    #             state_logvar_batch_t1, z_batch_t1)



    def get_mini_batches(self):

        # # Retrieve transition data in mini batches - OG VERSION
        # all_obs_batch, all_actions_batch, reward_batch_t1, terminated_batch_t2, truncated_batch_t2 = self.memory.sample(
        #         self.obs_indices, self.action_indices, self.reward_indices,
        #         self.terminated_indices, self.truncated_indices, self.max_n_indices, self.batch_size)

        # Retrieve transition data in mini batches
        all_obs_batch, all_actions_batch, reward_batch_t1, terminated_batch_t0, truncated_batch_t0 = self.memory.sample(
                self.obs_indices, self.action_indices, self.reward_indices,
                self.terminated_indices, self.truncated_indices, self.max_n_indices, self.batch_size)

        # # ACTIONS:
        # print("\nall_actions_batch")
        # for i, actions in enumerate(all_actions_batch): # i is the location index within the minibatch
        #     print(i, actions)
        # print("\n")

        # print("\nall_actions_batch - INDEXED")
        # for t in range(all_actions_batch.shape[1]): # t is the time index  for each entry in all_actions_batch
        #     print(t, all_actions_batch[:, t].unsqueeze(1))
        # print("\n")

        # # # Retrieve the agent's action history for time t0 and time t1
        # # action_batch_t0 = all_actions_batch[:, 0] # delete these, only for debug purposes - it's calculate below anyway.
        # # action_batch_t1 = all_actions_batch[:, 1] # delete these, only for debug purposes - it's calculate below anyway.

        # # print(f"\naction_batch_t0: {action_batch_t0}")
        # # print(f"action_batch_t1: {action_batch_t1}\n")

        # # Retrieve the agent's action history for time t0 and time t1
        # action_batch_t2 = all_actions_batch[:, 0] # delete these, only for debug purposes - it's calculate below anyway.
        # action_batch_t1 = all_actions_batch[:, 1] # delete these, only for debug purposes - it's calculate below anyway.

        # print(f"\naction_batch_t2: {action_batch_t2}")
        # print(f"action_batch_t1: {action_batch_t1}\n")

        # breakpoint()
        
        # Retrieve a batch of observations for n_screens consecutive points in time
        obs_batch_t0 = all_obs_batch[:, 0:self.n_screens, :].view(self.batch_size, -1)
        obs_batch_t1 = all_obs_batch[:, 1:self.n_screens+1, :].view(self.batch_size, -1)
        obs_batch_t2 = all_obs_batch[:, 2:self.n_screens+2, :].view(self.batch_size, -1)
        
        # Retrieve a batch of distributions over states for n_screens consecutive points in time
        state_mu_batch_t0, state_logvar_batch_t0 = self.vae.encode(obs_batch_t0)
        state_mu_batch_t1, state_logvar_batch_t1 = self.vae.encode(obs_batch_t1)
        state_mu_batch_t2, state_logvar_batch_t2 = self.vae.encode(obs_batch_t2)
        
        # Combine the sufficient statistics (mean and variance) into a single vector
        state_batch_t0 = torch.cat((state_mu_batch_t0, torch.exp(state_logvar_batch_t0)), dim=1)
        state_batch_t1 = torch.cat((state_mu_batch_t1, torch.exp(state_logvar_batch_t1)), dim=1)
        state_batch_t2 = torch.cat((state_mu_batch_t2, torch.exp(state_logvar_batch_t2)), dim=1)
        
        # Reparameterize the distribution over states for time t1
        z_batch_t1 = self.vae.reparameterize(state_mu_batch_t1, state_logvar_batch_t1)
        
        # # OG action_batch_t0 naming
        # # Retrieve the agent's action history for time t0 and time t1
        # action_batch_t0 = all_actions_batch[:, 0].unsqueeze(1)
        # action_batch_t1 = all_actions_batch[:, 1].unsqueeze(1)

        # Retrieve the agent's action history for time t2 and time t1 (second-last and last action)
        action_batch_t2 = all_actions_batch[:, 0].unsqueeze(1)
        action_batch_t1 = all_actions_batch[:, 1].unsqueeze(1)


        # At time t2 (two time steps ago) predict the state at time t1 (one time step ago):
        X = torch.cat((state_batch_t2.detach(), action_batch_t2.float()), dim = 1) # was t0
        pred_batch_mean_t1, pred_batch_logvar_t1 = self.transition_net(X)

        # Determine the prediction error wrt time t2-t1 using KL Divergence:
        pred_error_batch_t2t1 = torch.sum(
            self.gaussian_kl_div(
                pred_batch_mean_t1, torch.exp(pred_batch_logvar_t1), 
                state_mu_batch_t1, torch.exp(state_logvar_batch_t1)
            ), dim = 1
        ).unsqueeze(1)

        # # OG VERSION
        # return (state_batch_t1, state_batch_t2, action_batch_t1,
        #         reward_batch_t1, terminated_batch_t2, truncated_batch_t2, pred_error_batch_t0t1,
        #         obs_batch_t1, state_mu_batch_t1,
        #         state_logvar_batch_t1, z_batch_t1)

        return (state_batch_t1, state_batch_t2, action_batch_t1,
                reward_batch_t1, terminated_batch_t0, truncated_batch_t0, pred_error_batch_t2t1,
                obs_batch_t1, state_mu_batch_t1,
                state_logvar_batch_t1, z_batch_t1)


        
    # def compute_value_net_loss(self, state_batch_t1, state_batch_t2,
    #                        action_batch_t1, reward_batch_t1,
    #                        done_batch_t2, pred_error_batch_t0t1):

        
    # # def compute_value_net_loss(self, state_batch_t1, state_batch_t2,
    # #                        action_batch_t1, reward_batch_t1,
    # #                        done_batch_t2, pred_error_batch_t0t1):

    # # OG VERSION OF TERMINATED AND TRUNCATED
    # def compute_value_net_loss(
    #     self, state_batch_t1, state_batch_t2,
    #     action_batch_t1, reward_batch_t1,
    #     terminated_batch_t2, truncated_batch_t2, pred_error_batch_t0t1
    # ):

    # def compute_value_net_loss(
    #     self, state_batch_t1, state_batch_t2,
    #     action_batch_t1, reward_batch_t1,
    #     terminated_batch_t0, truncated_batch_t0, pred_error_batch_t0t1
    # ):

    def compute_value_net_loss(
        self, state_batch_t1, state_batch_t2,
        action_batch_t1, reward_batch_t1,
        terminated_batch_t0, truncated_batch_t0, pred_error_batch_t2t1
    ):
    
        with torch.no_grad():
            # Determine the action distribution for time t2:
            policy_batch_t2 = self.policy_net(state_batch_t2)
            
            # Determine the target EFEs for time t2:
            target_EFEs_batch_t2 = self.target_net(state_batch_t2)
            
            # Weigh the target EFEs according to the action distribution:
            # weighted_targets = ((1-done_batch_t2) * policy_batch_t2 *
            #                     target_EFEs_batch_t2).sum(-1).unsqueeze(1)

            weighted_targets = ((1-(terminated_batch_t0 | truncated_batch_t0)) * policy_batch_t2 *
                                target_EFEs_batch_t2).sum(-1).unsqueeze(1)
            
            # Determine the batch of bootstrapped estimates of the EFEs:
            EFE_estimate_batch = -reward_batch_t1 + pred_error_batch_t2t1 + self.Beta * weighted_targets
        
        # Determine the EFE at time t1 according to the value network:
        EFE_batch_t1 = self.value_net(state_batch_t1).gather(1, action_batch_t1)
        
        # Determine the MSE loss between the EFE estimates and the value network output:
        value_net_loss = F.mse_loss(EFE_estimate_batch, EFE_batch_t1)
        
        return value_net_loss
    
    # def compute_VFE(self, vae_loss, state_batch_t1, pred_error_batch_t0t1):
    def compute_VFE(self, vae_loss, state_batch_t1, pred_error_batch_t2t1):
        
        # Determine the action distribution for time t1:
        policy_batch_t1 = self.policy_net(state_batch_t1)
        
        # Determine the EFEs for time t1:
        EFEs_batch_t1 = self.value_net(state_batch_t1) # ONLY REPLACE THIS WITH ACTS PLANNING ????????????????????

        # Take a gamma-weighted Boltzmann distribution over the EFEs:
        boltzmann_EFEs_batch_t1 = torch.softmax(-self.gamma * EFEs_batch_t1, dim=1).clamp(min=1e-9, max=1-1e-9)
        
        # Weigh them according to the action distribution:
        energy_term_batch = -(policy_batch_t1 * torch.log(boltzmann_EFEs_batch_t1)).sum(-1).unsqueeze(1)
        
        # Determine the entropy of the action distribution
        entropy_batch = -(policy_batch_t1 * torch.log(policy_batch_t1)).sum(-1).unsqueeze(1)
        
        # Determine the VFE, then take the mean over all batch samples:
        VFE_batch = vae_loss + pred_error_batch_t2t1 + (energy_term_batch - entropy_batch)

        VFE = torch.mean(VFE_batch)
        
        return VFE
    
    def learn(self):
        
        # If there are not enough transitions stored in memory, return
        if self.memory.push_count - self.max_n_indices*2 < self.batch_size:
            return
        
        # After every freeze_period time steps, update the target network
        if self.freeze_cntr % self.freeze_period == 0:
            self.target_net.load_state_dict(self.value_net.state_dict())
        self.freeze_cntr += 1
        
        # Retrieve mini-batches of data from memory
        # (state_batch_t1, state_batch_t2, action_batch_t1,
        # reward_batch_t1, done_batch_t2, pred_error_batch_t0t1,
        # obs_batch_t1, state_mu_batch_t1,
        # state_logvar_batch_t1, z_batch_t1) = self.get_mini_batches()

        # # OG VERSION OF TERMINATED AND TRUNCATED:
        # (state_batch_t1, state_batch_t2, action_batch_t1,
        # reward_batch_t1, terminated_batch_t2, truncated_batch_t2, pred_error_batch_t0t1,
        # obs_batch_t1, state_mu_batch_t1,
        # state_logvar_batch_t1, z_batch_t1) = self.get_mini_batches()

        # (state_batch_t1, state_batch_t2, action_batch_t1,
        # reward_batch_t1, terminated_batch_t0, truncated_batch_t0, pred_error_batch_t0t1,
        # obs_batch_t1, state_mu_batch_t1,
        # state_logvar_batch_t1, z_batch_t1) = self.get_mini_batches()

        (state_batch_t1, state_batch_t2, action_batch_t1,
        reward_batch_t1, terminated_batch_t0, truncated_batch_t0, pred_error_batch_t2t1,
        obs_batch_t1, state_mu_batch_t1,
        state_logvar_batch_t1, z_batch_t1) = self.get_mini_batches()
        
        # Determine the reconstruction loss for time t1
        # recon_batch = self.vae.decode(z_batch_t1, self.batch_size)
        recon_batch = self.vae.decode(z_batch_t1)
        vae_loss = self.vae.loss_function(recon_batch, obs_batch_t1, state_mu_batch_t1, state_logvar_batch_t1, batch=True) / self.alpha
        
        # Compute the value network loss:
        # value_net_loss = self.compute_value_net_loss(state_batch_t1, state_batch_t2,
        #                    action_batch_t1, reward_batch_t1,
        #                    done_batch_t2, pred_error_batch_t0t1)

        # # OG VERSION OF TERMINATED AND TRUNCATED:
        # value_net_loss = self.compute_value_net_loss(
        #     state_batch_t1, state_batch_t2,
        #     action_batch_t1, reward_batch_t1,
        #     terminated_batch_t2, truncated_batch_t2, pred_error_batch_t0t1
        # )

        value_net_loss = self.compute_value_net_loss(
            state_batch_t1, state_batch_t2,
            action_batch_t1, reward_batch_t1,
            terminated_batch_t0, truncated_batch_t0, pred_error_batch_t2t1
        )
        
        # Compute the variational free energy:
        VFE = self.compute_VFE(vae_loss, state_batch_t1.detach(), pred_error_batch_t2t1)
        
        # Reset the gradients:
        self.vae.optimizer.zero_grad()
        self.policy_net.optimizer.zero_grad()
        self.transition_net.optimizer.zero_grad()
        self.value_net.optimizer.zero_grad()
        
        # Compute the gradients:
        VFE.backward(retain_graph=True)
        value_net_loss.backward()
        
        # Perform gradient descent:
        self.vae.optimizer.step()
        self.policy_net.optimizer.step()
        self.transition_net.optimizer.step()
        self.value_net.optimizer.step()
    
    def train_vae(self):
        """ Train the VAE separately. """
        
        vae_batch_size = 256
        vae_obs_indices = [self.n_screens-i for i in range(self.n_screens)]

        noise_std = 0.1
        
        losses = []
        for ith_episode in range(self.pt_vae_n_episodes):
            
            # self.env.reset()
            # obs = self.get_screen(self.env, self.device)
            obs, _ = self.env.reset()
            obs = torch.tensor(obs, dtype = torch.float32, device = self.device)
            noisy_obs = obs.cpu() + noise_std * np.random.randn(*obs.shape)
            noisy_obs = torch.tensor(noisy_obs, dtype = torch.float32, device = self.device)

            terminated = False
            truncated = False
            while not (terminated or truncated):
                
                action = self.env.action_space.sample()

                self.memory.push(obs, -99, -99, terminated, truncated)
                
                # _, _, terminated, truncated, _ = self.env.step(action)
                # obs = self.get_screen(self.env, self.device)

                obs, _, terminated, truncated, _ = self.env.step(action)
                obs = torch.tensor(obs, dtype = torch.float32, device = self.device)
                noisy_obs = obs.cpu() + noise_std * np.random.randn(*obs.shape)
                noisy_obs = torch.tensor(noisy_obs, dtype = torch.float32, device = self.device)
                
                if self.memory.push_count > vae_batch_size + self.n_screens*2:
                    obs_batch, _, _, _, _ = self.memory.sample(vae_obs_indices, [], [], [], [], len(vae_obs_indices), vae_batch_size)
                    # obs_batch = obs_batch.view(vae_batch_size, self.c, self.h, self.w, self.n_screens)
                    obs_batch = obs_batch.view(vae_batch_size, -1)

                    # recon, mu, logvar = self.vae.forward(obs_batch, vae_batch_size) # TypeError: VAE.forward() takes 2 positional arguments but 3 were given
                    recon, mu, logvar = self.vae.forward(obs_batch)
                    loss = torch.mean(self.vae.loss_function(recon, obs_batch, mu, logvar))
                    
                    self.vae.optimizer.zero_grad()
                    loss.backward()
                    self.vae.optimizer.step()
                    
                    losses.append(loss)
                    print("episode %4d: vae_loss=%5.2f"%(ith_episode, loss.item()))
                    
                    if (terminated or truncated):
                        if ith_episode > 0 and ith_episode % 10 > 0 and self.pt_vae_plot:
                            plt.plot(losses)
                            plt.show()
                            plt.plot(losses[-1000:])
                            plt.show()
                            for i in range(self.n_screens):
                                plt.imshow(obs_batch[0, :, :, :, i].detach().cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
                                plt.show()
                                plt.imshow(recon[0, :, :, :, i].detach().cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
                                plt.show()
                
                if (terminated or truncated):
                    # self.memory.push(obs, -99, -99, terminated, truncated)
                    self.memory.push(noisy_obs, -99, -99, terminated, truncated)

                    
                    if ith_episode > 0 and ith_episode % 100 == 0:
                        torch.save(self.vae.state_dict(), "networks/pre_trained_vae/vae_n{}_{:d}.pth".format(
                                self.n_latent_states, ith_episode))
            
        self.memory.push_count = 0
        torch.save(self.vae.state_dict(), "networks/pre_trained_vae/vae_n{}_end.pth".format(self.n_latent_states))
            
    def train(self):

        # THE VECTOR BASED VERSION !!!!!!!!!!!!

        filename = f"Deep_AIF_MDP_Cart_Pole_v1"
        figure_file = f"plots/{filename}.png"
        
        if self.pre_train_vae: # If True: pre-train the VAE
            msg = "Environment is: {}\nPre-training vae. Starting at {}\n".format(self.env.unwrapped.spec.id, datetime.datetime.now())
            print(msg)
            if self.keep_log:
                self.record.write(msg+"\n")
            self.train_vae()
            
        msg = "Environment is: {}\nTraining started at {}".format(self.env.unwrapped.spec.id, datetime.datetime.now())
        print(msg)
        if self.keep_log:
            self.record.write(msg+"\n")

        if torch.cuda.is_available():
            print(f'CUDA is available on device: {torch.cuda.get_device_name(0)}')
        else:
            print('CUDA is NOT Available')

        noise_std = 0.1
        
        results = []
        for ith_episode in range(self.n_episodes):
            
            total_reward = 0
            # self.env.reset()
            # obs = self.get_screen(self.env, self.device)
            obs, _ = self.env.reset()
            noisy_obs = obs + noise_std * np.random.randn(*obs.shape)
            noisy_obs = torch.tensor(noisy_obs, dtype = torch.float32, device = self.device)

            # print(f"\ntype(obs): {type(obs)}")
            # print(f"obs: {obs}\n")
            # breakpoint()

            # done = False
            terminated = False
            truncated = False
            reward = 0
            
            while not (terminated or truncated):
                
                # action = self.select_action(obs)
                action = self.select_action(noisy_obs)

                # self.memory.push(obs, action, reward, done)
                # self.memory.push(obs, action, reward, terminated, truncated)
                self.memory.push(noisy_obs, action, reward, terminated, truncated)
                
                # _, reward, done, _ = self.env.step(action[0].item())
                # _, reward, terminated, truncated, _ = self.env.step(action[0].item())
                # obs = self.get_screen(self.env, self.device)
                obs, reward, terminated, truncated, _ = self.env.step(action[0].item())
                noisy_obs = obs + noise_std * np.random.randn(*obs.shape)
                noisy_obs = torch.tensor(noisy_obs, dtype = torch.float32, device = self.device) 

                total_reward += reward
                
                self.learn()
                
                if (terminated or truncated):
                    # self.memory.push(obs, -99, -99, terminated, truncated)
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
                torch.save(self.value_net.state_dict(), "networks/intermediary/intermediary_networks{}_{:d}.pth".format(self.run_id, ith_episode))
        
        self.env.close()
        
        # If enabled, save the results and the network (state_dict)
        if self.save_results:
            np.savez("results/intermediary/intermediary_results{}_end".format(self.run_id), np.array(results))
            np.savez(self.results_path, np.array(results))
        if self.save_network:
            torch.save(self.value_net.state_dict(), "networks/intermediary/intermediary_networks{}_end.pth".format(self.run_id))
            torch.save(self.value_net.state_dict(), self.network_save_path)
        
        # Print and keep a (.txt) record of stuff
        msg = "Training finished at {}".format(datetime.datetime.now())
        print(msg)
        if self.keep_log:
            self.record.write(msg)
            self.record.close()

        x = [i + 1 for i in range(agent.n_episodes)]
        plot_learning_curve(x, results, figure_file, "AcT Action Selection")
                
if __name__ == "__main__":

    # filename = f"Deep_AIF_MDP_Cart_Pole_v1"
    # figure_file = f"plots/{filename}.png"

    agent = Agent(sys.argv[1:])
    agent.train()

    # x = [i + 1 for i in range(agent.n_episodes)]
    # plot_learning_curve(x, agent.results, figure_file, "AcT Action Selection")

    # memes !!!!!!!!!!!!!!!!!!!!!!!!!!!!




    # def gaussian_kl_div(self, mu1, sigma1, mu2, sigma2):
    #     """
    #     Calculate KL divergence between two diagonal multivariate Gaussian distributions.

    #     Parameters:
    #         mu1 (torch.Tensor): Mean of the first distribution.
    #         sigma1 (torch.Tensor): Diagonal standard deviations of the first distribution.
    #         mu2 (torch.Tensor): Mean of the second distribution.
    #         sigma2 (torch.Tensor): Diagonal standard deviations of the second distribution.

    #     Returns:
    #         torch.Tensor: KL divergence.
    #     """

    #     print(f"\nmu1.shape: {mu1.shape}")
    #     print(f"sigma1.shape: {sigma1.shape}")
    #     print(f"mu2.shape: {mu2.shape}")
    #     print(f"sigma2.shape: {sigma2.shape}\n")

    #     k = mu1.size(-1)  # Dimensionality of the distributions

    #     matrix = torch.matmul(
    #         torch.diag_embed(
    #             sigma2.reciprocal()
    #         ), torch.diag_embed(sigma1)
    #     ).squeeze()

    #     trace_term = torch.trace(
    #         matrix
    #     )

    #     Mahalanobis_term = torch.t(mu2 - mu1) @ torch.diag_embed(
    #         sigma2.reciprocal()
    #     ) @ (mu2 - mu1)

    #     log_term = -k + torch.sum(torch.log(sigma2)) - torch.sum(torch.log(sigma1))

    #     kl = 0.5 * (trace_term + Mahalanobis_term + log_term)

    #     return kl


    # OG VERSION:
    # def get_mini_batches(self):
    #     # Retrieve transition data in mini batches
    #     # all_obs_batch, all_actions_batch, reward_batch_t1, done_batch_t2 = self.memory.sample(
    #     #         self.obs_indices, self.action_indices, self.reward_indices,
    #     #         self.done_indices, self.max_n_indices, self.batch_size)
    #     all_obs_batch, all_actions_batch, reward_batch_t1, terminated_batch_t2, truncated_batch_t2 = self.memory.sample(
    #             self.obs_indices, self.action_indices, self.reward_indices,
    #             self.terminated_indices, self.truncated_indices, self.max_n_indices, self.batch_size)
        
    #     # Retrieve a batch of observations for 3 consecutive points in time
    #     obs_batch_t0 = all_obs_batch[:, 0:self.n_screens, :, :, :].view(self.batch_size, self.c, self.h, self.w, self.n_screens)
    #     obs_batch_t1 = all_obs_batch[:, 1:self.n_screens+1, :, :, :].view(self.batch_size, self.c, self.h, self.w, self.n_screens)
    #     obs_batch_t2 = all_obs_batch[:, 2:self.n_screens+2, :, :, :].view(self.batch_size, self.c, self.h, self.w, self.n_screens)
        
    #     # Retrieve a batch of distributions over states for 3 consecutive points in time
    #     state_mu_batch_t0, state_logvar_batch_t0 = self.vae.encode(obs_batch_t0)
    #     state_mu_batch_t1, state_logvar_batch_t1 = self.vae.encode(obs_batch_t1)
    #     state_mu_batch_t2, state_logvar_batch_t2 = self.vae.encode(obs_batch_t2)
        
    #     # Combine the sufficient statistics (mean and variance) into a single vector
    #     state_batch_t0 = torch.cat((state_mu_batch_t0, torch.exp(state_logvar_batch_t0)), dim=1)
    #     state_batch_t1 = torch.cat((state_mu_batch_t1, torch.exp(state_logvar_batch_t1)), dim=1)
    #     state_batch_t2 = torch.cat((state_mu_batch_t2, torch.exp(state_logvar_batch_t2)), dim=1)
        
    #     # Reparameterize the distribution over states for time t1
    #     z_batch_t1 = self.vae.reparameterize(state_mu_batch_t1, state_logvar_batch_t1)
        
    #     # Retrieve the agent's action history for time t0 and time t1
    #     action_batch_t0 = all_actions_batch[:, 0].unsqueeze(1)
    #     action_batch_t1 = all_actions_batch[:, 1].unsqueeze(1)
        
    #     # At time t0 predict the state at time t1:
    #     X = torch.cat((state_batch_t0.detach(), action_batch_t0.float()), dim=1)
    #     pred_batch_t0t1 = self.transition_net(X)

    #     # Determine the prediction error wrt time t0-t1:
    #     pred_error_batch_t0t1 = torch.mean(F.mse_loss(
    #             pred_batch_t0t1, state_mu_batch_t1, reduction='none'), dim=1).unsqueeze(1)
        
    #     # return (state_batch_t1, state_batch_t2, action_batch_t1,
    #     #         reward_batch_t1, done_batch_t2, pred_error_batch_t0t1,
    #     #         obs_batch_t1, state_mu_batch_t1,
    #     #         state_logvar_batch_t1, z_batch_t1)

    #     # print(f"\nget_mini_batches - type(terminated_batch_t2): {type(terminated_batch_t2)}")
    #     # print(f"get_mini_batches - type(truncated_batch_t2): {type(truncated_batch_t2)}\n")

    #     return (state_batch_t1, state_batch_t2, action_batch_t1,
    #             reward_batch_t1, terminated_batch_t2, truncated_batch_t2, pred_error_batch_t0t1,
    #             obs_batch_t1, state_mu_batch_t1,
    #             state_logvar_batch_t1, z_batch_t1)

    # def get_mini_batches(self):
    #     # Retrieve transition data in mini batches
    #     all_obs_batch, all_actions_batch, reward_batch_t1, terminated_batch_t2, truncated_batch_t2 = self.memory.sample(
    #             self.obs_indices, self.action_indices, self.reward_indices,
    #             self.terminated_indices, self.truncated_indices, self.max_n_indices, self.batch_size)
        
    #     # Retrieve a batch of observations for 3 consecutive points in time
    #     obs_batch_t0 = all_obs_batch[:, 0:self.n_screens, :, :, :].view(self.batch_size, self.c, self.h, self.w, self.n_screens)
    #     obs_batch_t1 = all_obs_batch[:, 1:self.n_screens+1, :, :, :].view(self.batch_size, self.c, self.h, self.w, self.n_screens)
    #     obs_batch_t2 = all_obs_batch[:, 2:self.n_screens+2, :, :, :].view(self.batch_size, self.c, self.h, self.w, self.n_screens)
        
    #     # Retrieve a batch of distributions over states for 3 consecutive points in time
    #     state_mu_batch_t0, state_logvar_batch_t0 = self.vae.encode(obs_batch_t0)
    #     state_mu_batch_t1, state_logvar_batch_t1 = self.vae.encode(obs_batch_t1)
    #     state_mu_batch_t2, state_logvar_batch_t2 = self.vae.encode(obs_batch_t2)
        
    #     # Combine the sufficient statistics (mean and variance) into a single vector
    #     state_batch_t0 = torch.cat((state_mu_batch_t0, torch.exp(state_logvar_batch_t0)), dim=1)
    #     state_batch_t1 = torch.cat((state_mu_batch_t1, torch.exp(state_logvar_batch_t1)), dim=1)
    #     state_batch_t2 = torch.cat((state_mu_batch_t2, torch.exp(state_logvar_batch_t2)), dim=1)
        
    #     # Reparameterize the distribution over states for time t1
    #     z_batch_t1 = self.vae.reparameterize(state_mu_batch_t1, state_logvar_batch_t1)
        
    #     # Retrieve the agent's action history for time t0 and time t1
    #     action_batch_t0 = all_actions_batch[:, 0].unsqueeze(1)
    #     action_batch_t1 = all_actions_batch[:, 1].unsqueeze(1)
        
    #     # At time t0 predict the state at time t1:
    #     X = torch.cat((state_batch_t0.detach(), action_batch_t0.float()), dim=1)
    #     pred_batch_t0t1 = self.transition_net(X)

    #     # Determine the prediction error wrt time t0-t1:
    #     pred_error_batch_t0t1 = torch.mean(F.mse_loss(
    #             pred_batch_t0t1, state_mu_batch_t1, reduction='none'), dim=1).unsqueeze(1)

    #     return (state_batch_t1, state_batch_t2, action_batch_t1,
    #             reward_batch_t1, terminated_batch_t2, truncated_batch_t2, pred_error_batch_t0t1,
    #             obs_batch_t1, state_mu_batch_t1,
    #             state_logvar_batch_t1, z_batch_t1)

    # # OG VERSION:
    # def kl_div(self, mu_1, sigma_sq_1, mu_2, sigma_sq_2):
    #     '''
    #     Calculates the KL Divergence between P(mu_1, sigma_sq_1) and Q(mu_2, sigma_sq_2)
    #     D_KL[P || Q], where P and Q are Univariate Gaussians.
    #     '''
    #     return (1/2)*(
    #         2*torch.log(sigma_sq_2 / sigma_sq_1) + \ # get rid of the 2 coeff !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #         ((sigma_sq_1 ** 2)/(sigma_sq_2 ** 2)) + \
    #         ((mu_1 - mu_2) ** 2) / (sigma_sq_2 ** 2) - 1
    #     )

    # # def get_screen(self, env, device='cuda', displacement_h=0, displacement_w=0):
    # def get_screen(self, env, device='cpu', displacement_h=0, displacement_w=0):
    #     """
    #     Get a (pre-processed, i.e. cropped, cart-focussed) observation/screen
    #     For the most part taken from:
    #         https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    #     """
        
    #     def get_cart_location(env, screen_width):
    #         world_width = env.x_threshold * 2
    #         scale = screen_width / world_width
    #         return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART
        
    #     # Returned screen requested by gym is 400x600x3, but is sometimes larger
    #     # such as 800x1200x3. Transpose it into torch order (CHW).
    #     # screen = env.render(render_mode='rgb_array').transpose((2, 0, 1))
    #     screen = env.render().transpose((2, 0, 1))
        
    #     # Cart is in the lower half, so strip off the top and bottom of the screen
    #     _, screen_height, screen_width = screen.shape
    #     screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    #     view_width = int(screen_width * 0.6)
    #     cart_location = get_cart_location(env, screen_width)+displacement_w
    #     if cart_location < view_width // 2:
    #         slice_range = slice(view_width)
    #     elif cart_location > (screen_width - view_width // 2):
    #         slice_range = slice(-view_width, None)
    #     else:
    #         slice_range = slice(cart_location - view_width // 2,
    #                             cart_location + view_width // 2)
    #     # Strip off the edges, so that we have a square image centered on a cart
    #     screen = screen[:, :, slice_range]
    #     # Convert to float, rescale, convert to torch tensor
    #     # (this doesn't require a copy)
    #     screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    #     screen = torch.from_numpy(screen)
    #     # Resize, and add a batch dimension (BCHW)
    #     screen = self.resize(screen).unsqueeze(0).to(device)
    #     screen = screen[:, :, 2:-1, 3:-2]
    #     return screen
    
    # OG VERSION:
    # def select_action(self, obs):
    #     with torch.no_grad():
    #         # Derive a distribution over states state from the last n observations (screens):
    #         prev_n_obs = self.memory.get_last_n_obs(self.n_screens-1)
    #         x = torch.cat((prev_n_obs, obs), dim=0).view(1, self.c, self.h, self.w, self.n_screens)
    #         state_mu, state_logvar = self.vae.encode(x)
            
    #         # Determine a distribution over actions given the current observation:
    #         x = torch.cat((state_mu, torch.exp(state_logvar)), dim=1)
    #         policy = self.policy_net(x)
    #         return torch.multinomial(policy, 1)

# class VAE(nn.Module):
#     # In part taken from:
#     #   https://github.com/pytorch/examples/blob/master/vae/main.py

#     def __init__(self, n_screens, n_latent_states, lr=1e-5, device='cpu'):
#         super(VAE, self).__init__()
        
#         self.device = device
        
#         self.n_screens = n_screens
#         self.n_latent_states = n_latent_states
        
#         # The convolutional encoder
#         self.encoder = nn.Sequential(                
#                 nn.Conv3d(3, 16, (5,5,1), (2,2,1)),
#                 nn.BatchNorm3d(16),
#                 nn.ReLU(inplace=True),
                
#                 nn.Conv3d(16, 32, (5,5,1), (2,2,1)),
#                 nn.BatchNorm3d(32),
#                 nn.ReLU(inplace=True),
                
#                 nn.Conv3d(32, 32, (5,5,1), (2,2,1)),
#                 nn.BatchNorm3d(32),
#                 nn.ReLU(inplace=True)   
#                 ).to(self.device)
        
#         # The size of the encoder output
#         self.conv3d_shape_out = (32, 2, 8, self.n_screens)
#         self.conv3d_size_out = np.prod(self.conv3d_shape_out)
        
#         # The convolutional decoder
#         self.decoder = nn.Sequential(
#                 nn.ConvTranspose3d(32, 32, (5,5,1), (2,2,1)),
#                 nn.BatchNorm3d(32),
#                 nn.ReLU(inplace=True),
                
#                 nn.ConvTranspose3d(32, 16, (5,5,1), (2,2,1)),
#                 nn.BatchNorm3d(16),
#                 nn.ReLU(inplace=True),
                
#                 nn.ConvTranspose3d(16, 3, (5,5,1), (2,2,1)),
#                 nn.BatchNorm3d(3),
#                 nn.ReLU(inplace=True),
                
#                 nn.Sigmoid()
#                 ).to(self.device)
        
#         # Fully connected layers connected to encoder
#         self.fc1 = nn.Linear(self.conv3d_size_out, self.conv3d_size_out // 2)
#         self.fc2_mu = nn.Linear(self.conv3d_size_out // 2, self.n_latent_states)
#         self.fc2_logvar = nn.Linear(self.conv3d_size_out // 2, self.n_latent_states)
        
#         # Fully connected layers connected to decoder
#         self.fc3 = nn.Linear(self.n_latent_states, self.conv3d_size_out // 2)
#         self.fc4 = nn.Linear(self.conv3d_size_out // 2, self.conv3d_size_out)
        
#         self.optimizer = optim.Adam(self.parameters(), lr)
        
#         self.to(self.device)

#     def encode(self, x):
#         # Deconstruct input x into a distribution over latent states
#         conv = self.encoder(x)
#         h1 = F.relu(self.fc1(conv.view(conv.size(0), -1)))
#         mu, logvar = self.fc2_mu(h1), self.fc2_logvar(h1)
#         return mu, logvar

#     def reparameterize(self, mu, logvar):
#         # Apply reparameterization trick
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         return mu + eps*std

#     def decode(self, z, batch_size=1):
#         # Reconstruct original input x from the (reparameterized) latent states
#         h3 = F.relu(self.fc3(z))
#         deconv_input = self.fc4(h3)
#         deconv_input = deconv_input.view([batch_size] + [dim for dim in self.conv3d_shape_out])
#         y = self.decoder(deconv_input)
#         return y

#     def forward(self, x, batch_size=1):
#         # Deconstruct and then reconstruct input x
#         mu, logvar = self.encode(x)
#         z = self.reparameterize(mu, logvar)
#         recon = self.decode(z, batch_size)
#         return recon, mu, logvar

#     # Reconstruction + KL divergence losses summed over all elements and batch
#     def loss_function(self, recon_x, x, mu, logvar, batch=True):
#         '''
#         Returns the ELBO
#         '''
#         if batch:
#             BCE = F.binary_cross_entropy(recon_x, x, reduction='none')
#             BCE = torch.sum(BCE, dim=(1, 2, 3, 4))
            
#             KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

#         else:
#             BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
#             # see Appendix B from VAE paper:
#             # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#             # https://arxiv.org/abs/1312.6114
#             # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#             KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
#         return BCE + KLD