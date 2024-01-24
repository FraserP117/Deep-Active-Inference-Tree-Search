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
    
    def __init__(self, capacity, obs_shape, device='cuda:0'):
        
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
        self.inferred_hidden_state_mem = torch.empty([capacity]+[dim for dim in self.obs_shape], dtype=torch.float32, device=self.device)
        
        self.push_count = 0 # The number of times new data has been pushed to memory

    def push(self, obs, action, reward, terminated, truncated, inferred_hidden_state):
        # Store data in memory
        self.obs_mem[self.position()] = obs
        self.action_mem[self.position()] = action
        self.reward_mem[self.position()] = reward
        self.terminated_mem[self.position()] = terminated
        self.truncated_mem[self.position()] = truncated
        self.inferred_hidden_state_mem[self.position()] = inferred_hidden_state
        
        self.push_count += 1
    
    def position(self):
        # Returns the next position (index) to which data is pushed
        return self.push_count % self.capacity

    def sample(self, obs_indices, action_indices, reward_indices, terminated_indices, truncated_indices, hidden_state_indices, max_n_indices, batch_size):
        # Fine as long as max_n is not greater than the fewest number of time steps an episode can take
        
        # Pick indices at random
        end_indices = np.random.choice(min(self.push_count, self.capacity) - max_n_indices * 2, batch_size, replace=False) + max_n_indices
        
        # Correct for sampling near the position where data was last pushed
        for i in range(len(end_indices)):
            if end_indices[i] in range(self.position(), self.position() + max_n_indices):
                end_indices[i] += max_n_indices
        
        # Retrieve the specified indices that come before the end_indices
        obs_batch = self.obs_mem[np.array([index - obs_indices for index in end_indices])] # ValueError: operands could not be broadcast together with shapes (1,0) (3,)
        action_batch = self.action_mem[np.array([index - action_indices for index in end_indices])]
        reward_batch = self.reward_mem[np.array([index - reward_indices for index in end_indices])]
        terminated_batch = self.terminated_mem[np.array([index - terminated_indices for index in end_indices])]
        truncated_batch = self.truncated_mem[np.array([index - truncated_indices for index in end_indices])]
        hidden_state_batch = self.obs_mem[np.array([index - hidden_state_indices for index in end_indices])]

        # Correct for sampling over multiple episodes
        for i in range(len(end_indices)):
            index = end_indices[i]
            for j in range(1, max_n_indices):
                if self.terminated_mem[index - j] or self.truncated_mem[index - j]:
                    for k in range(len(obs_indices)):
                        if obs_indices[k] >= j:
                            obs_batch[i, k] = torch.zeros_like(self.obs_mem[0])
                    for k in range(len(action_indices)):
                        if action_indices[k] >= j:
                            action_batch[i, k] = torch.zeros_like(self.action_mem[0])
                    for k in range(len(reward_indices)):
                        if reward_indices[k] >= j:
                            reward_batch[i, k] = torch.zeros_like(self.reward_mem[0])
                    for k in range(len(terminated_indices)):
                        if terminated_indices[k] >= j:
                            terminated_batch[i, k] = torch.zeros_like(self.terminated_mem[0])
                    for k in range(len(truncated_indices)):
                        if truncated_indices[k] >= j:
                            truncated_batch[i, k] = torch.zeros_like(self.truncated_mem[0])
                    for k in range(len(hidden_state_indices)):
                        if hidden_state_indices[k] >= j:
                            hidden_state_batch[i, k] = torch.zeros_like(self.obs_mem[0])
                    break

        return obs_batch, action_batch, reward_batch, terminated_batch, truncated_batch, hidden_state_batch
    
class Model(nn.Module):
    
    def __init__(self, n_inputs, n_outputs, n_hidden=64, lr=1e-4, softmax=False, device='cuda:0'):
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

    def __init__(self, n_inputs, n_outputs, n_hidden, lr=1e-4, device='cpu', name = None):

        super(MVGaussianModel, self).__init__()

        self.name = name

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
        log_var = self.log_var(x_2) # 2 * log (sigma)

        return mean, log_var

    def rsample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(mu)

        return mu + (eps * std)


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
        
        # self.obs_shape = (4,) 
        self.obs_shape = self.env.observation_space.shape # The shape of observations
        self.obs_size = np.prod(self.obs_shape) # The size of the observation
        self.n_actions = self.env.action_space.n # The number of actions available to the agent
        self.all_actions = [0, 1]
        self.freeze_cntr = 0 # Keeps track of when to (un)freeze the target network

        # specify the goal prior distribution: p_C:
        self.goal_mu = torch.tensor([0, 0, 0, 0]).to(self.device)
        self.goal_var = torch.tensor([1, 1, 1, 1]).to(self.device)

        # Generative state prior: P_theta
        self.prior_transition_net_theta = MVGaussianModel(
            self.latent_state_dim + 1,      # inputs:  s_{t-1}, a_{t-1}
            self.latent_state_dim,          # outputs: mu(s_t), sigma(s_t)
            self.n_hidden_gen_trans,        # hidden layer
            lr = self.lr_gen_trans,         # learning rate
            device = self.device,
            name = 'prior_theta'
        )

        # Variational state posterior: Q_phi
        self.posterior_transition_net_phi = MVGaussianModel(
            2 * self.latent_state_dim + 1,  # inputs:  s_{t-1}, a_{t-1}, o_t
            self.latent_state_dim,          # outputs: mu(s_t), sigma(s_t)
            self.n_hidden_var_trans,        # hidden layer
            lr = self.lr_var_trans,         # learning rate
            device = self.device,
            name = 'posterior_phi'
        )

        # Generative observation likelihiid: P_xi
        self.generative_observation_net_xi = MVGaussianModel(
            self.latent_state_dim,          # input:   s_t (reparam'd state sample)
            self.latent_state_dim,          # outputs: mu(o_t), sigma(o_t)
            self.n_hidden_gen_obs,          # hidden layer
            lr = self.lr_gen_obs,           # learning rate
            device = self.device,
            name = 'obs_xi'
        )

        # Variational Policy prior: Q_nu
        self.policy_net_nu = Model(
            2 * self.latent_state_dim,      # inputs: mu(s_t), sigma(s_t)
            self.n_actions,                 # output: categorical action probs for all actions in action space
            self.n_hidden_pol,              # hidden layer
            lr = self.lr_pol,               # learning rate
            softmax = True, 
            device = self.device
        )

        # EFE bootstrap-estimate network: f_psi
        self.value_net_psi = Model(
            2 * self.latent_state_dim,      # input:  mu(s_t), sigma(s_t)
            self.n_actions,                 # output: Estimated EFE for all actions in actions space
            self.n_hidden_val,              # hidden layer
            lr = self.lr_val,               # learning rate
            device = self.device
        )

        # TARGET EFE bootstrap-estimate network: f_psi
        self.target_net = Model(
            2 * self.latent_state_dim,      # input:  mu(s_t), sigma(s_t)
            self.n_actions,                 # output: Estimated EFE for all actions in actions space
            self.n_hidden_val,              # hidden layer
            lr = self.lr_val,               # learnng rate
            device = self.device
        )

        self.target_net.load_state_dict(self.value_net_psi.state_dict())
            
        if self.load_network: # If true: load the networks given paths

            self.generative_transition_net.load_state_dict(torch.load(self.network_load_path.format("trans")))
            self.generative_transition_net.eval()

            self.generative_observation_net_xi.load_state_dict(torch.load(self.network_load_path.format("obs")))
            self.generative_observation_net_xi.eval()

            self.variational_transition_net.load_state_dict(torch.load(self.network_load_path.format("var_trans")))
            self.variational_transition_net.eval()

            self.policy_net_nu.load_state_dict(torch.load(self.network_load_path.format("pol"), map_location=self.device))
            self.policy_net_nu.eval()

            self.value_net_psi.load_state_dict(torch.load(self.network_load_path.format("val"), map_location=self.device))
            self.value_net_psi.eval()

        # Initialize the replay memory
        self.memory = ReplayMemory(self.memory_capacity, self.obs_shape, device=self.device)

        # When sampling from memory at index i, obs_indices indicates that we want observations with indices i-obs_indices, works the same for the others
        self.obs_indices = [3, 2, 1, 0] # the third most recent, second most recent and most recent observation
        self.hidden_state_indices = [3, 2, 1, 0] # the third most recent, second most recent and most recent inferred hidden state
        # self.action_indices = [2, 1]
        self.action_indices = [3, 2, 1, 0]
        self.reward_indices = [1]
        # self.done_indices = [0]
        self.terminated_indices = [0]
        self.truncated_indices = [0]

        self.max_n_indices = max(max(self.obs_indices, self.action_indices, self.reward_indices, self.terminated_indices, self.truncated_indices, self.hidden_state_indices)) + 1 


    def set_parameters(self, argv):
        
        # The default parameters
        default_parameters = {'run_id':"_rX", 'device':'cpu', # 'device':'cuda',
              'env':'CartPole-v1', 'n_episodes':2000, 
              'latent_state_dim':4, 'lr_vae':1e-5, 'alpha':25000, # latent_state_dim was OG used by th VAE
              'n_hidden_gen_trans':64, 'lr_gen_trans':1e-3,
              'n_hidden_var_trans':64, 'lr_var_trans':1e-5,
              'n_hidden_gen_obs':64, 'lr_gen_obs':1e-5,
              'n_hidden_pol':64, 'lr_pol':1e-3,
              'n_hidden_val':64, 'lr_val':1e-5,
              'memory_capacity':65536, 'batch_size':32, 'freeze_period':25, 
              'Beta':0.99, 'gamma':12.00,
              'print_timer':100,
              'keep_log':True, 'log_path':"logs/ai_pomdp_log{}.txt", 'log_save_timer':10,
              'save_results':True, 'results_path':"results/ai_pomdp_results{}.npz", 'results_save_timer':500,
              'save_network':True, 'network_save_path':"networks/ai_pomdp_{}net{}.pth", 'network_save_timer':500,
              'load_network':False, 'network_load_path':"networks/ai_pomdp_{}net_rX.pth",
              # 'pre_train_models':False, 'pt_n_episodes':500, 'pt_models_plot':False,
              'pre_train_models':True, 'pt_n_episodes':500, 'pt_models_plot':True,
              'load_pre_trained_vae':False, 'pt_vae_load_path':"networks/pre_trained_vae/vae_n{}_end.pth"}
        # Possible commands:
            # python ai_pomdp_agent.py device=cuda:0
            # python ai_pomdp_agent.py device=cuda:0 load_pre_trained_vae=False pre_train_models=True
        
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
        self.latent_state_dim = int(custom_parameters['latent_state_dim'])
        self.lr_vae = float(custom_parameters['lr_vae'])
        self.alpha = int(custom_parameters['alpha']) # Used to scale down the VAE's loss

        self.n_hidden_gen_trans = int(custom_parameters['n_hidden_gen_trans'])
        self.lr_gen_trans = float(custom_parameters['lr_gen_trans'])

        self.n_hidden_var_trans = int(custom_parameters['n_hidden_var_trans'])
        self.lr_var_trans = float(custom_parameters['lr_var_trans'])

        self.n_hidden_gen_obs = int(custom_parameters['n_hidden_gen_obs'])
        self.lr_gen_obs = float(custom_parameters['lr_gen_obs'])

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
        
        self.pre_train_models = interpret_boolean(custom_parameters['pre_train_models']) # If true pre trains the vae
        self.pt_n_episodes = custom_parameters['pt_n_episodes'] # The amount of episodes for which to pre train the vae
        self.pt_models_plot = interpret_boolean(custom_parameters['pt_models_plot']) # If true plots stuff while training the Models
        
        self.load_pre_trained_vae = interpret_boolean(custom_parameters['load_pre_trained_vae']) # If true loads a pre trained vae
        self.pt_vae_load_path = custom_parameters['pt_vae_load_path'].format(self.latent_state_dim) # The path from which to load the pre trained vae
        
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

    def monte_carlo_EFE(mu_theta, log_var_theta, N = 1):

        """
        Calculates the 'N' Monte Carlo approximation of the Expected Free Energy (EFE).
        mu_theta and log_var_theta parameterise the distribution W.R.T to which the state samples are drawn.
        This is the prior transition model: p_theta. 

        Parameters:
            mu_theta (torch.Tensor): batch of Mean vectors of the prior transition model.
            log_var_theta (torch.Tensor): batch of Diagonal standard deviations of the prior transition model.
            N: The number of Monte Carlo samples to use when approximating the expected entropy of the 
            observation likelihood. 

        Returns:
            torch.Tensor: Monte Carlo EFE approximation.

        1. calculate the kl div between p_theta and p_C
        2. calculate the Monte Carlo estimate of the expected log-evidence under p_theta:
            2.1. Gather 'N' reparameterised samples from p_theta
            2.2. Send these samples through the observation likelihod model: p_xi to get 'N' observation distributions.
            2.3. calcualte the log of each 
        """

        # calculate the KL divergenced:
        kl_div = self.gaussian_kl_div(
            mu_theta, 
            toch.exp(log_var_theta), 
            self.goal_mu, 
            self.goal_var
        )

        # Generate hidden state Samples from p_theta
        samples_theta = self.posterior_transition_net_phi.rsample(mu_theta, log_var_theta)

        # Generate the batch of predicted observation beliefs by feeding the State Samples into P_xi
        mu_xi, log_var_xi = self.generative_observation_net_xi(samples_theta)

        # Compute Monte Carlo Estimate
        mc_expected_entropy = self.Gaussian_entropy(torch.exp(log_var_xi))

        # compute the final approx EFE:
        mc_efe = kl_div - mc_expected_entropy

        return mc_efe

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

    def update_precision_AcT(self, depth, alpha, beta):

        per_depth_precision = stats.gamma.rvs(alpha, scale=beta)

        if depth > 0:
            per_depth_precision *= depth

        return per_depth_precision

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

    def variational_inference_AcT(self, node):

        # Compute the precision 𝛾_𝜏: for the current time
        precision_tau = self.update_precision_AcT(
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

    def select_action_and_infer_state(self, last_state_sample, last_action, curr_obs):

        if self.pre_train_models:

            # prepare the inputs to Q_phi:
            x = torch.cat((last_state_sample, last_action.float(), curr_obs), dim = 0)

            # state_mu, state_logvar = self.vae.encode(x) # Q_phi: posterior_transition_net_phi
            state_mu, state_logvar = self.posterior_transition_net_phi(x)

            # sample q_phi
            current_state_sample = self.posterior_transition_net_phi.rsample(state_mu, state_logvar)

            # sample a random action
            action = torch.tensor(self.env.action_space.sample(), dtype = torch.int64, device = self.device).unsqueeze(0)

            return action, current_state_sample

        else:

            with torch.no_grad():

                # prepare the inputs to Q_phi:
                x = torch.cat((last_state_sample, last_action.float(), curr_obs), dim = 0)

                # state_mu, state_logvar = self.vae.encode(x) # Q_phi: posterior_transition_net_phi
                state_mu, state_logvar = self.posterior_transition_net_phi(x)

                # sample q_phi
                current_state_sample = self.posterior_transition_net_phi.rsample(state_mu, state_logvar)

                # Determine a distribution over actions given the current observation:
                x = torch.cat((state_mu, torch.exp(state_logvar)), dim = 0)

                policy = self.policy_net_nu(x)

                return torch.multinomial(policy, 1), current_state_sample

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

    # # With Cholesky Decomposition
    # def Gaussian_entropy(sigma_sq):
    #     """
    #     Calculate entropy of the parameterized multivariate Gaussian distribution.

    #     Parameters:
    #         sigma_sq (torch.Tensor): Diagonal variances of the distribution.

    #     Returns:
    #         torch.Tensor: Gaussian Entropy.
    #     """
    #     k = sigma_sq.size(-1)  # Dimensionality of the distribution

    #     L = torch.diag(torch.sqrt(sigma_sq))
        
    #     entropy = 0.5 * (k + k * torch.log(2 * torch.tensor(np.pi)) + torch.sum(torch.log(L)))

    #     return entropy

    def Gaussian_entropy(sigma_sq):
        """
        Calculate entropy of the parameterised multivariate Gaussian distribution.

        Parameters:
            sigma_sq (torch.Tensor): Diagonal variances of the distribution.

        Returns:
            torch.Tensor: Gaussian Entropy.
        """
        k = sigma_sq.size(-1)  # Dimensionality of the distribution

        log_term = torch.sum(torch.log(sigma_sq))

        entropy = 0.5 * (k + k*torch.log(2*torch.tensor(np.pi)) + log_term)

        return entropy

    def get_mini_batches(self):

        all_obs_batch, all_actions_batch, reward_batch_t1, terminated_batch_t2, truncated_batch_t2, hidden_state_batch = self.memory.sample(
            self.obs_indices, self.action_indices, self.reward_indices, 
            self.terminated_indices, self.truncated_indices, self.hidden_state_indices, 
            self.max_n_indices, self.batch_size
        )

        # Retrieve a batch of inferred hidden states for 3 consecutive points in time
        inferred_state_batch_t0 = hidden_state_batch[:, 0].view([self.batch_size] + [dim for dim in self.obs_shape])
        inferred_state_batch_t1 = hidden_state_batch[:, 1].view([self.batch_size] + [dim for dim in self.obs_shape])
        inferred_state_batch_t2 = hidden_state_batch[:, 2].view([self.batch_size] + [dim for dim in self.obs_shape])

        # Retrieve a batch of observations, hidden states, and actions for consecutive points in time
        # obs_batch_t0 = all_obs_batch[:, 0, :].view(self.batch_size, -1)  # Most recent observation
        obs_batch_t1 = all_obs_batch[:, 1, :].view(self.batch_size, -1)  # Second most recent observation
        obs_batch_t2 = all_obs_batch[:, 2, :].view(self.batch_size, -1)  # Third most recent observation
        obs_batch_t3 = all_obs_batch[:, 3, :].view(self.batch_size, -1)  # Fourth most recent observation 

        # Retrieve the agent's action history for time t0, t1, t2 and t3
        action_batch_t0 = all_actions_batch[:, 0].unsqueeze(1)
        action_batch_t1 = all_actions_batch[:, 1].unsqueeze(1)
        action_batch_t2 = all_actions_batch[:, 2].unsqueeze(1)
        action_batch_t3 = all_actions_batch[:, 3].unsqueeze(1)

        q_phi_inputs_t0 = torch.cat((inferred_state_batch_t0, action_batch_t1, obs_batch_t1), dim = 1) # s_t, a_{t + 1}, o_{t + 1}
        q_phi_inputs_t1 = torch.cat((inferred_state_batch_t1, action_batch_t2, obs_batch_t2), dim = 1) # s_{t + 1}, a_{t + 2}, o_{t + 2}
        q_phi_inputs_t2 = torch.cat((inferred_state_batch_t2, action_batch_t3, obs_batch_t3), dim = 1) # s_{t + 2}, a_{t + 3}, o_{t + 3}

        # Retrieve a batch of distributions over states for n_screens consecutive points in time
        state_mu_batch_t0, state_logvar_batch_t0 = self.posterior_transition_net_phi(q_phi_inputs_t0) # \mu{s_t}, \log{\Sigma^2(s_t)}
        state_mu_batch_t1, state_logvar_batch_t1 = self.posterior_transition_net_phi(q_phi_inputs_t1) # \mu{s_{t + 1}}, \log{\Sigma^2(s_{t + 1})}
        state_mu_batch_t2, state_logvar_batch_t2 = self.posterior_transition_net_phi(q_phi_inputs_t2) # \mu{s_{t + 2}}, \log{\Sigma^2(s_{t + 2})}

        # Reparameterize the distribution over states for time t0 and t1
        z_batch_t0 = self.posterior_transition_net_phi.rsample(state_mu_batch_t0, state_logvar_batch_t0)
        z_batch_t1 = self.posterior_transition_net_phi.rsample(state_mu_batch_t1, state_logvar_batch_t1)

        # At time t0 predict the state at time t1:
        X = torch.cat((state_mu_batch_t0.detach(), action_batch_t0.float()), dim = 1)
        pred_batch_mean_t0t1, pred_batch_logvar_t0t1 = self.prior_transition_net_theta(X)

        # Determine the prediction error wrt time t0-t1 using state KL Divergence:
        pred_error_batch_t0t1 = torch.sum(
            self.gaussian_kl_div(
                pred_batch_mean_t0t1, torch.exp(pred_batch_logvar_t0t1),
                state_mu_batch_t1, torch.exp(state_logvar_batch_t1)
            ), dim=1
        ).unsqueeze(1)

        # print(f"\n\nget_mini_batches - state_mu_batch_t0:\n{state_mu_batch_t0}")
        # print(f"get_mini_batches - state_logvar_batch_t0:\n{state_logvar_batch_t0}")
        # print(f"get_mini_batches - state_mu_batch_t1:\n{state_mu_batch_t1}")
        # print(f"get_mini_batches - state_logvar_batch_t1:\n{state_logvar_batch_t1}\n\n")

        return (
            state_mu_batch_t1, state_logvar_batch_t1, 
            state_mu_batch_t2, state_logvar_batch_t2, 
            action_batch_t1, reward_batch_t1, 
            terminated_batch_t2, truncated_batch_t2, pred_error_batch_t0t1,
            obs_batch_t1, state_mu_batch_t1,
            state_logvar_batch_t1, z_batch_t0, z_batch_t1
        )

    # def mc_expected_log_evidence(self, inputs_phi):
    def mc_expected_log_evidence(self, samples_phi):

        # Generate the batch of predicted observation beliefs
        mu_xi, log_var_xi = self.generative_observation_net_xi(samples_phi)

        var_xi = torch.diag_embed(torch.exp(log_var_xi))

        # Reparameterize Observation Samples
        samples_xi = self.generative_observation_net_xi.rsample(mu_xi, log_var_xi)

        samples_xi_clamped = torch.clamp(samples_xi, min=-1e9, max=1e9)

        multivariate_normal_p = torch.distributions.MultivariateNormal(
            loc = mu_xi,
            covariance_matrix = var_xi
        )

        # log_likelihood_values_p = multivariate_normal_p.log_prob(samples_xi)
        log_likelihood_values_p = multivariate_normal_p.log_prob(samples_xi_clamped) # here

        log_likelihood_values_p_clamped = torch.clamp(log_likelihood_values_p, min=-100.0, max=100.0)

        # Compute Monte Carlo Estimate
        mc_log_likelihood = torch.mean(log_likelihood_values_p_clamped)

        return mc_log_likelihood

    def compute_VFE(
        self, 
        expected_log_ev, 
        state_mu_batch_t1, state_logvar_batch_t1,
        pred_error_batch_t0t1
    ):

        # Determine the action distribution for time t1:
        state_batch_t1 = torch.cat((state_mu_batch_t1, state_logvar_batch_t1), dim = 1)
        policy_batch_t1 = self.policy_net_nu(state_batch_t1)
        
        # Determine the EFEs for time t1:
        EFEs_batch_t1 = self.value_net_psi(state_batch_t1) # ONLY REPLACE THIS WITH ACTS PLANNING

        # Take a gamma-weighted Boltzmann distribution over the EFEs:
        boltzmann_EFEs_batch_t1 = torch.softmax(-self.gamma * EFEs_batch_t1, dim=1).clamp(min=1e-9, max=1-1e-9)
        
        # Weigh them according to the action distribution:
        energy_term_batch = -(policy_batch_t1 * torch.log(boltzmann_EFEs_batch_t1)).sum(-1).unsqueeze(1)
        
        # Determine the entropy of the action distribution
        entropy_batch = -(policy_batch_t1 * torch.log(policy_batch_t1)).sum(-1).unsqueeze(1)
        
        # Determine the VFE, then take the mean over all batch samples:
        VFE_batch = expected_log_ev + pred_error_batch_t0t1 + (energy_term_batch - entropy_batch)

        VFE = torch.mean(VFE_batch)
        
        return VFE

    def compute_value_net_psi_loss(
        self,
        state_mu_batch_t1, state_logvar_batch_t1, 
        state_mu_batch_t2, state_logvar_batch_t2,
        action_batch_t1, reward_batch_t1,
        terminated_batch_t2, truncated_batch_t2, pred_error_batch_t0t1
    ):

        with torch.no_grad():

            # Determine the action distribution for time t2:
            state_batch_t2 = torch.cat((state_mu_batch_t2.detach(), state_logvar_batch_t2.detach()), dim = 1)

            policy_batch_t2 = self.policy_net_nu(state_batch_t2)
            
            # Determine the target EFEs for time t2:
            target_EFEs_batch_t2 = self.target_net(state_batch_t2)
            
            # Weigh the target EFEs according to the action distribution:
            weighted_targets = ((1-(terminated_batch_t2 | truncated_batch_t2)) * policy_batch_t2 *
                                target_EFEs_batch_t2).sum(-1).unsqueeze(1)
            
            # Determine the batch of bootstrapped estimates of the EFEs:
            EFE_estimate_batch = -reward_batch_t1 + pred_error_batch_t0t1 + self.Beta * weighted_targets
        
        # Determine the EFE at time t1 according to the value network:
        state_batch_t1 = torch.cat((state_mu_batch_t1.detach(), state_logvar_batch_t1.detach()), dim = 1)

        EFE_batch_t1 = self.value_net_psi(state_batch_t1).gather(1, action_batch_t1)
        
        # Determine the MSE loss between the EFE estimates and the value network output:
        value_net_psi_loss = F.mse_loss(EFE_estimate_batch, EFE_batch_t1)
        
        return value_net_psi_loss
    
    def learn(self):
        '''
        Train on mini batches of training data from memory.

        Calculate the VFE/EFE and perform Gradient Descent W.R.T to the 
        params of the neural networks on the VFE loss. 

        Compute the EFE-lookahead via the bootstrapped neural network prediction. 
        Perform Gradient Descent on the EFE-value net loss. 
        '''
        
        # If there are not enough transitions stored in memory, return
        if self.memory.push_count - self.max_n_indices*2 < self.batch_size:
            return
        
        # After every freeze_period time steps, update the target network
        if self.freeze_cntr % self.freeze_period == 0:
            self.target_net.load_state_dict(self.value_net_psi.state_dict())
        self.freeze_cntr += 1
        
        # Retrieve mini-batches of data from memory
        (
            state_mu_batch_t1, state_logvar_batch_t1,
            state_mu_batch_t2, state_logvar_batch_t2,
            action_batch_t1, reward_batch_t1, 
            terminated_batch_t2, truncated_batch_t2, pred_error_batch_t0t1,
            obs_batch_t1, state_mu_batch_t1,
            state_logvar_batch_t1, z_batch_t0, z_batch_t1
        ) = self.get_mini_batches()

        # Determine the reconstruction loss for time t1
        expected_log_ev = self.mc_expected_log_evidence(z_batch_t1)
        
        # Compute the value network loss:
        value_net_psi_loss = self.compute_value_net_psi_loss(
            state_mu_batch_t1, state_logvar_batch_t1, 
            state_mu_batch_t2, state_logvar_batch_t2,
            action_batch_t1, reward_batch_t1,
            terminated_batch_t2, truncated_batch_t2, pred_error_batch_t0t1
        )

        # Compute the variational free energy:
        VFE = self.compute_VFE(
            expected_log_ev, 
            state_mu_batch_t1.detach(), state_logvar_batch_t1.detach(),
            pred_error_batch_t0t1
        )
        
        # Reset the gradients:
        self.policy_net_nu.optimizer.zero_grad()
        self.prior_transition_net_theta.optimizer.zero_grad()
        self.posterior_transition_net_phi.zero_grad()
        self.generative_observation_net_xi.zero_grad()
        self.value_net_psi.optimizer.zero_grad()
        
        # Compute the gradients:
        VFE.backward(retain_graph=True)
        value_net_psi_loss.backward()


        # # Print gradients for each network
        # print("\n\nlearn - Policy Net Gradients:")
        # for name, param in self.policy_net_nu.named_parameters():
        #     if param.grad is not None:
        #         print(f"learn - {name}: {param.grad.norm().item()}")

        # print("\nlearn - Prior Transition Net Gradients:")
        # for name, param in self.prior_transition_net_theta.named_parameters():
        #     if param.grad is not None:
        #         print(f"learn - {name}: {param.grad.norm().item()}")

        # print("\nlearn - Posterior Transition Net Gradients:")
        # for name, param in self.posterior_transition_net_phi.named_parameters():
        #     if param.grad is not None:
        #         print(f"learn - {name}: {param.grad.norm().item()}")

        # print("\nlearn - Generative Observation Net Gradients:")
        # for name, param in self.generative_observation_net_xi.named_parameters():
        #     if param.grad is not None:
        #         print(f"learn - {name}: {param.grad.norm().item()}")

        # print("\nlearn - Value Net Gradients:")
        # for name, param in self.value_net_psi.named_parameters():
        #     if param.grad is not None:
        #         print(f"learn - {name}: {param.grad.norm().item()}")
        # print(f"\n\n")
        

        # Perform gradient descent:
        self.policy_net_nu.optimizer.step()
        self.prior_transition_net_theta.optimizer.step()
        self.posterior_transition_net_phi.optimizer.step()
        self.generative_observation_net_xi.optimizer.step()
        self.value_net_psi.optimizer.step()

        return VFE, value_net_psi_loss, expected_log_ev

    def train_models(self):
        """ Train the models on a random policy. """
        
        batch_size = 256

        noise_std = 0.1
        
        losses = []

        for ith_episode in range(self.pt_n_episodes):
            
            obs, _ = self.env.reset()
            obs = torch.tensor(obs, dtype = torch.float32, device = self.device)
            noisy_obs = obs.cpu() + noise_std * np.random.randn(*obs.shape)
            noisy_obs = torch.tensor(noisy_obs, dtype = torch.float32, device = self.device)

            terminated = False
            truncated = False

            state_sample = torch.zeros(self.obs_shape)  # initial hidden state is the zero vector
            action = torch.tensor(np.random.choice(self.all_actions), dtype = torch.int64).unsqueeze(0) # initial action is random

            while not (terminated or truncated):

                # this function now handles the case of self.pre_train_models:
                action, state_sample = self.select_action_and_infer_state(state_sample, action, noisy_obs)

                self.memory.push(obs, action, -99, terminated, truncated, state_sample)
                
                obs, _, terminated, truncated, _ = self.env.step(action.item())
                obs = torch.tensor(obs, dtype = torch.float32, device = self.device)
                noisy_obs = obs.cpu() + noise_std * np.random.randn(*obs.shape)
                noisy_obs = torch.tensor(noisy_obs, dtype = torch.float32, device = self.device)
                
                if self.memory.push_count > batch_size: # + self.n_screens*2:

                    VFE, value_net_psi_loss, expected_log_ev = self.learn()

                    print(f"episode: {ith_episode}, VFE = {VFE}, value_net_psi_loss = {value_net_psi_loss}, expected_log_ev = {expected_log_ev}")
                
                if (terminated or truncated):

                    self.memory.push(noisy_obs, action, -99, terminated, truncated, state_sample)

                    # # UNCOMMENT THIS LATER, WILL BE NECESSARY !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    # if ith_episode > 0 and ith_episode % 100 == 0:
                    #     torch.save(
                    #         self.vae.state_dict(), "networks/pre_trained_vae/vae_n{}_{:d}.pth".format(
                    #         self.latent_state_dim, ith_episode)
                    #     )
            
        self.memory.push_count = 0
        # torch.save(self.vae.state_dict(), "networks/pre_trained_vae/vae_n{}_end.pth".format(self.latent_state_dim)) # UNCOMMENT THIS LATER


    def train(self):

        # THE VECTOR BASED VERSION !!!!!!!!!!!!

        filename = f"Deep_AIF_MDP_Cart_Pole_v1"
        figure_file = f"plots/{filename}.png"
        
        if self.pre_train_models: # If True: pre-train the VAE
            msg = "Environment is: {}\nPre-training models. Starting at {}\n".format(self.env.unwrapped.spec.id, datetime.datetime.now())
            print(msg)
            if self.keep_log:
                self.record.write(msg+"\n")
            self.train_models()
            
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
            noisy_obs = torch.tensor(noisy_obs, dtype = torch.float32, device = self.device) # initial observation
            state_sample = torch.zeros(self.obs_shape)  # initial hidden state is the zero vector
            action = torch.tensor(np.random.choice(self.all_actions), dtype = torch.int64).unsqueeze(0) # initial action is random

            terminated = False
            truncated = False
            reward = 0

            while not (terminated or truncated):
                
                # action = self.select_action_and_infer_state(obs)
                # action = self.select_action_and_infer_state(state, action, noisy_obs) # return the inferred state too ??????????????????????????
                action, state_sample = self.select_action_and_infer_state(state_sample, action, noisy_obs)

                self.memory.push(noisy_obs, action, reward, terminated, truncated, state_sample)
                
                # _, reward, done, _ = self.env.step(action[0].item())
                # _, reward, terminated, truncated, _ = self.env.step(action[0].item())
                # obs = self.get_screen(self.env, self.device)
                obs, reward, terminated, truncated, _ = self.env.step(action[0].item())
                noisy_obs = obs + noise_std * np.random.randn(*obs.shape)
                noisy_obs = torch.tensor(noisy_obs, dtype = torch.float32, device = self.device) 

                total_reward += reward
                
                self.learn()

                # self.observation = noisy_obs
                # self.selected_action = action
                
                if (terminated or truncated):
                    # self.memory.push(obs, -99, -99, terminated, truncated)
                    self.memory.push(noisy_obs, -99, -99, terminated, truncated, state_sample)

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
                torch.save(self.value_net_psi.state_dict(), "networks/intermediary/intermediary_networks{}_{:d}.pth".format(self.run_id, ith_episode))
        
        self.env.close()
        
        # If enabled, save the results and the network (state_dict)
        if self.save_results:
            np.savez("results/intermediary/intermediary_results{}_end".format(self.run_id), np.array(results))
            np.savez(self.results_path, np.array(results))
        if self.save_network:
            torch.save(self.value_net_psi.state_dict(), "networks/intermediary/intermediary_networks{}_end.pth".format(self.run_id))
            torch.save(self.value_net_psi.state_dict(), self.network_save_path)
        
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
    # agent.train()
    agent.train_models()

    # x = [i + 1 for i in range(agent.n_episodes)]
    # plot_learning_curve(x, agent.results, figure_file, "AcT Action Selection")