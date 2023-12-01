# __author__ = "Otto van der Himst"
# __credits__ = "Otto van der Himst, Pablo Lanillos"
# __version__ = "1.0"
# __email__ = "o.vanderhimst@student.ru.nl"

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
from scipy.special import softmax


'''

* exploration parameter 𝜅𝑝 = 1

NEURAL NETOWRKS - THE MDP AGENT:

* EFE-value network: f_{psi}(s_t)
    - Used to create the bootstrapped EFE estimate
* Transition network: f_{phi}(s_{t-1}, a_{t-1})
    - Yields the predicted next state, given the previous state and action
    - implements p(s_{t+1} | s_t, a_t) = B matrix
    - lowercase s denotes the state.
* Policy network: g_{xi}(a_t | s_t)
    - The distribution over actions from which to sample. 


NEURAL NETOWRKS - THE POMDP AGENT

* Encoder network: q_{theta}(s_t, o_{t-3:t}) = S_{mu, t}, ln(S_{Sigma, t})
    - Encodes hight dimensional observations into a low/erdimensional latent space.
* Decoder  network: p_{nu}(o_{t-3:T} | Z_t)
    - Reconstructs OG high-dimensional observations from the low/er dimensional latent space encoding.
    - The latemt space encoding is further encoded into "Z_t" where:
        * sigma = exp((ln(S_{Sigma}))/(2))
        * epsilon ~ N(0, 1)
        * Z = S_{mu} + epsilon * sigma 

* EFE-value network: f_{psi}(S_{mu, t}, S_{Sigma, t})
    - Used to create the bootstrapped EFE estimate
* State-transition network: f_{phi}(S_{mu, t-1}, a_{t-1})
    - Yields the predicted next state, given the previous state and action
    - implements p(s_{t+1} | s_t, a_t) = B matrix
* Policy network: g_{xi}(a_t | S_{mu, t}, S_{Sigma})
    - The distribution over actions from which to sample. 


QUESTIONS and THINGS TO DO:

* Don't really understand E very well yet.
* Updating the action posterior - how do we do this exactly? with E? with a simple ratio of visits?
* W.R.T the unused_actions list for each node:
    - what happends when a node has used all possible actions to transition from itself to the next node?
* The predictive EFE - how is that caluclated again?
* Do I always need to use get_mini_batches when I want to use the transitionm network, or can I just directly 
pass inputs to the transition network?
* a_prime = what for initial root node?
* review the path_integration_AcT function


* Still need to implement:
    - ?????? 
* Is there a difference between a node being "fully expanded" as opposed to just "expanded"?
    - I don't think so, however:
        - From page 16, Section 3.1.2 Second stage: Expansion: "Expansion of a leaf node 𝑣𝜏 corresponds to instantiating a
        new child node 𝑣′ by implementing a random action 𝑢′ among those previously unused."
            - perhaps then a "fully expanded node" is one for which all possible actions have been used to 
              effect a transition from itself to one if its child nodes?

CURRENT PROBLEMS:

* IndexError on line 631

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
    
    def __init__(self, capacity, obs_shape, device='cpu'):
        
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
    

class Model(nn.Module):
    
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
        h_relu = F.relu(self.fc1(x))
        y = self.fc2(h_relu)
        
        if self.softmax:
            y = F.softmax(self.fc2(h_relu), dim=-1).clamp(min=1e-9, max=1-1e-9)
        
        return y


class Node:

    def __init__(self, state_belief = None, action_at_tau_minus_one = None, parent = None):
        '''
        This is a FULLY OBSERVED NODE. The corresponding Node class in the POMDP version of the problem
        will not store the state at time t. Here a STATE = an OBSERVATION.

        state_belief = prediction from transition network
        action_at_tau_minus_one = previous action
        parent = the parent node (a "Node" instance)
        '''

        # predictive_EFE_tau is used to score the quality of a generic policy with respect to the future 
        # outcomes and states that are expected under such policies.
        self.action_space = np.array([0, 1]) # the action space for CartPole-v1
        self.predictive_EFE_tau = 0 # The EFE computed at time tau: delta^tau * G(*, v_tau).
        self.state_belief = state_belief # x_tau = the predicted state in the MDP case
        self.visit_count = 0 # the number of times this node has been visited
        self.depth = 0 # the depth od the node from the root
        self.parent = parent # this node's parent node
        self.children = [] # this node's children
        self.action_at_tau_minus_one = action_at_tau_minus_one # the action that led to the visitation of the present node
        self.action_posterior_belief = np.ones(len(self.action_space)) / len(self.action_space) # this is a distribution over all possible actions

        # self.unused_actions = [] # a list of all actions that have not been used to transition from this node to a subsequent node.
        self.used_actions = [] # a list of all actions that HAVE been used to transition from this node to a subsequent node.

    def update_action_posterior_belief(self):
        """
        Update the action posterior belief based on visit counts and exploration factor.
        exploration_factor: The exploration factor (𝜅𝑝) that modulates the prior belief about policies.

        Called upon each visit to the present node.
        """
        total_visits = self.visit_count

        if total_visits > 0:
            visit_counts = np.array([child.visit_count for child in self.children])
            values = np.array([child.predictive_EFE_tau for child in self.children])

            # Compute action probabilities using softmax function
            action_probabilities = softmax(values)

            self.action_posterior_belief = action_probabilities
        else:
            self.action_posterior_belief = np.zeros(len(self.children))


class Agent():
    
    def __init__(self, argv):
        
        self.set_parameters(argv) # Set parameters
        
        self.obs_shape = self.env.observation_space.shape # The shape of observations
        self.obs_size = np.prod(self.obs_shape) # The size of the observation
        self.n_actions = self.env.action_space.n # The number of actions available to the agent
        self.all_actions = np.array([0, 1]) # ADDED 
        self.freeze_cntr = 0 # Keeps track of when to (un)freeze the target network
        
        # Initialize the networks:
        self.transition_net = Model(self.obs_size+1, self.obs_size, self.n_hidden_trans, lr=self.lr_trans, device=self.device)
        self.policy_net = Model(self.obs_size, self.n_actions, self.n_hidden_pol, lr=self.lr_pol, softmax=True, device=self.device)
        self.value_net = Model(self.obs_size, self.n_actions, self.n_hidden_val, lr=self.lr_val, device=self.device)
        
        if self.load_network: # If true: load the networks given paths
            self.transition_net.load_state_dict(torch.load(self.network_load_path.format("trans")))
            self.transition_net.eval()
            self.policy_net.load_state_dict(torch.load(self.network_load_path.format("pol")))
            self.policy_net.eval()
            self.value_net.load_state_dict(torch.load(self.network_load_path.format("val")))
            self.value_net.eval()

        # self.target_net = Model(self.obs_size, self.n_actions, self.n_hidden_val, lr=self.lr_val, device=self.device) # OG
        self.target_net = Model(int(self.obs_size), int(self.n_actions), self.n_hidden_val, lr=self.lr_val, device=self.device)

        self.target_net.load_state_dict(self.value_net.state_dict())
        
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
            'run_id':"_rX", 'device':torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            # 'env':'LunarLander-v2', 'n_episodes':50, 
            'env':'CartPole-v1', 'n_episodes':500, # OG VERSION FOR CARTPOLE-V1
            'n_hidden_trans':64, 'lr_trans':1e-3, 
            'n_hidden_pol':64, 'lr_pol':1e-3, 
            'n_hidden_val':64, 'lr_val':1e-4,
            'memory_capacity':65536, 'batch_size':64, 'freeze_period':25,
            'Beta':0.99, 'gamma':1.00, 
            'print_timer':100,
            'keep_log':True, 'log_path':"logs/ai_mdp_log{}.txt", 'log_save_timer':10,
            'save_results':True, 'results_path':"results/ai_mdp_results{}.npz", 'results_save_timer':500,
            'save_network':True, 'network_save_path':"networks/ai_mdp_{}net{}.pth", 'network_save_timer':500,
            'load_network':False, 'network_load_path':"networks/ai_mdp_{}net_rX.pth",
            'delta':0.95, 'd': 5, 'epsilon': 0.4, 'k_p': 1.0
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
        self.n_hidden_trans = int(custom_parameters['n_hidden_trans'])
        self.lr_trans = float(custom_parameters['lr_trans'])
        self.n_hidden_pol = int(custom_parameters['n_hidden_pol'])
        self.lr_pol = float(custom_parameters['lr_pol'])
        self.n_hidden_val = int(custom_parameters['n_hidden_val'])
        self.lr_val = float(custom_parameters['lr_val'])
        
        self.memory_capacity = int(custom_parameters['memory_capacity']) # The maximum number of items to be stored in memory
        self.batch_size = int(custom_parameters['batch_size']) # The mini-batch size
        self.freeze_period = int(custom_parameters['freeze_period']) # The number of time-steps the target network is frozen
        
        self.gamma = float(custom_parameters['gamma']) # A precision parameter
        self.Beta = float(custom_parameters['Beta']) # The discount rate
        
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
    
    def get_mini_batches(self):
        # Retrieve transition data in mini batches
        all_obs_batch, all_actions_batch, reward_batch_t1, terminated_batch_t2, truncated_batch_t2 = self.memory.sample(
            self.obs_indices, self.action_indices, 
            self.reward_indices, self.terminated_indicies, self.truncated_indicies, 
            self.max_n_indices, self.batch_size
        )
        
        # Retrieve a batch of observations for 3 consecutive points in time
        obs_batch_t0 = all_obs_batch[:, 0].view([self.batch_size] + [dim for dim in self.obs_shape])
        obs_batch_t1 = all_obs_batch[:, 1].view([self.batch_size] + [dim for dim in self.obs_shape])
        obs_batch_t2 = all_obs_batch[:, 2].view([self.batch_size] + [dim for dim in self.obs_shape])
        
        # Retrieve the agent's action history for time t0 and time t1
        action_batch_t0 = all_actions_batch[:, 0].unsqueeze(1)
        action_batch_t1 = all_actions_batch[:, 1].unsqueeze(1)
        
        # At time t0 predict the state at time t1:
        X = torch.cat((obs_batch_t0, action_batch_t0.float()), dim=1)

        pred_batch_t0t1 = self.transition_net(X)

        # Determine the prediction error wrt time t0-t1:
        pred_error_batch_t0t1 = torch.mean(
            F.mse_loss(
                pred_batch_t0t1, obs_batch_t1, reduction='none'
            ), dim=1
        ).unsqueeze(1)

        return (
            obs_batch_t0, obs_batch_t1, obs_batch_t2, action_batch_t0,
            action_batch_t1, reward_batch_t1, terminated_batch_t2, truncated_batch_t2, 
            pred_error_batch_t0t1
        )

    def select_action_AcT(self, root_node):
        action_probabilities = root_node.action_posterior_belief
        action_indices = np.arange(len(action_probabilities))
        chosen_action_index = np.random.choice(action_indices, p = action_probabilities)
        chosen_action = self.all_actions[chosen_action_index]

        return chosen_action

    # def tree_policy_AcT(self, node, B): 
    def tree_policy_AcT(self, node): 
        '''
        self.transition_net = B
        '''
        while not self.node_is_terminal_leaf_AcT(node):
            if not self.node_fully_expanded_AcT(node):
                # return self.expand_AcT(node, B)
                return self.expand_AcT(node)
            else:
                node = self.variational_inference_AcT(node)
        v_tau = node

        return v_tau

    # def expand_AcT(self, node, B):
    def expand_AcT(self, node):
        '''
        Performs the node expansion step.
        '''
        unused_actions = self.get_unused_actions_AcT(node)
        a_prime = random.choice(unused_actions)
        node.used_actions.append(a_prime) # ADDED 

        # At time t0 predict the state at time t1:
        state_transition_t0t1 = torch.cat((node.state, a_prime), dim = 1) # For MDP, obs_t0 = node.state
        pred_state_t0t1 = self.transition_net(state_transition_t0t1)

        child = Node(state_belief=pred_state_t0t1, action=a_prime, parent=node)
        node.children.append(child)

        return child

    def variational_inference_AcT(self, node):
        exploration_factor = math.sqrt(2 * math.log(node.parent.visit_count) / node.visit_count)
        probabilities = [math.exp(exploration_factor * child.predictive_EFE_tau) for child in node.parent.children]
        probabilities_sum = sum(probabilities)
        action_probs = [prob / probabilities_sum for prob in probabilities]
        a_prime = random.choices(node.parent.children, weights=action_probs)[0].action
        return self.find_child_node_AcT(node, a_prime)

    # def evaluate_AcT(self, node, A, B, C, delta):
    def evaluate_AcT(self, node, delta):
        '''
        returns: the predictive EFE, G_{Delta}
        '''
      
        # Determine the EFE for time t1:
        g = self.value_net(node.state).detach()

        g_delta = delta ** node.visit_count * g

        return g_delta

    def path_integration_AcT(self, node, g_delta):
        while node != None:
            node.visit_count += 1
            node.predictive_EFE_tau += (1 / node.visit_count) * (g_delta - node.predictive_EFE_tau)

            # node.update_action_posterior_belief() # UPDATE APPROX ACTION POSTERIOR HERE?

            node = node.parent

    def halting_conditions_satisfied_AcT(self, t):
        # simply test to see if within alllowable time horizon - can make more sophisticated
        return t > 25

    def node_is_terminal_leaf_AcT(self, node):
        return node.children == []

    def node_fully_expanded_AcT(self, node):
        '''
        Returns True iff the input node has exhausted the action space in making transitions between itself and its children. 
        '''
        return len(node.children) == self.all_actions

    def get_unused_actions_AcT(self, node):
        '''
        THIS IS INSUFFICIENT. MUST ACTUALLY HANDLE THE FULLY EXPANDED CASE.
        '''
        unused_actions = []

        for action in self.all_actions:
            if action not in node.used_actions:
                unused_actions.append(action)
            else:
                print(f"\nNode: {node} has exhausted all available actions\n")

        return unused_actions

    def find_child_node_AcT(self, node, action):
        '''
        Return the child node to the input: "node", which will be transitioned 
        to, as a conseqence of performing the input action. 
        '''
        for child in node.children:
            if child.action_at_tau_minus_one == action:

                return child

    def active_inferece_tree_search(self, initial_state_belief, delta, epsilon):
        '''
        Iteratively performs Active Inference Tree Search.
        initial_state_belief is the prior distribution over on hidden state 
        beliefs (POMDP), or the actual initial state (MDP).
        '''

        # Initialize the planning tree with the root node
        root_node = Node(
            state_belief = initial_state_belief, action_at_tau_minus_one = None, parent = None # probably need to look at action_at_tau_minus_one = ???
        )

        # the planning time-horizon 
        t = 0

        # while within planning time-horizon and any other halting condition not satisfied:
        while not self.halting_conditions_satisfied_AcT(t):

            # update expected state belief/value x_{t-1} using s_t, o_t, a_{t-1}, A, B, D
            initial_random_action = torch.tensor(np.random.choice(self.all_actions)).float().unsqueeze(0)
            root_node_state_belief = root_node.state_belief # .unsqueeze(0)

            state_transition = torch.cat((root_node_state_belief, initial_random_action), dim = 0)
            pred_state = self.transition_net(state_transition)

            root_node.state_belief = pred_state

            # build the planning tree
            while delta ** root_node.visit_count < epsilon:

                # Perform Stage 1 and 2: Variational Inference and Expansion
                selected_node = self.tree_policy_AcT(root_node)
                new_node = self.expand_AcT(selected_node)
                new_node.depth += selected_node.depth
                new_node.parent = selected_node
                selected_node.children.append(new_node)


                # Stage 3: Evaluation
                value = self.evaluate_AcT(new_node, delta)

                # Stage 4: Path Integration
                self.path_integration_AcT(new_node, value)

                # Update the action posterior belief for the selected node 
                new_node.update_action_posterior_belief() # do this here?

            # # Update time step and other variables
            # s_t_plus_1, o_t_plus_1, pred_state, a_t = self.extract_information(root_node)

            t += 1

        # return the root node: containing all necessary information
        return root_node


    def select_action_dAIF(self, obs):
        '''
        Standard dAIF action-selection
        '''
        with torch.no_grad():
            # Determine the action distribution given the current observation:
            policy = self.policy_net(obs)

            return torch.multinomial(policy, 1)
    
    def compute_EFE_value_net_loss(
        self, obs_batch_t1, obs_batch_t2, action_batch_t1, reward_batch_t1,
        terminated_batch_t2, truncated_batch_t2, pred_error_batch_t0t1
    ):
        
        with torch.no_grad():

            # Determine the action distribution for time t2:
            policy_batch_t2 = self.policy_net(obs_batch_t2)
            
            # Determine the target EFEs for time t2:
            target_EFEs_batch_t2 = self.target_net(obs_batch_t2)
            
            # Weigh the target EFEs according to the action distribution:
            weighted_EFE_value_targets = ((1-(terminated_batch_t2 | truncated_batch_t2)) * policy_batch_t2 *
                                target_EFEs_batch_t2).sum(-1).unsqueeze(1)
                
            # Determine the batch of bootstrapped estimates of the EFEs:
            EFE_estimate_batch = -reward_batch_t1 + pred_error_batch_t0t1 + self.Beta * weighted_EFE_value_targets
        
        # Determine the EFE at time t1 according to the value network:
        EFE_batch_t1 = self.value_net(obs_batch_t1).gather(1, action_batch_t1)
            
        # Determine the MSE loss between the EFE estimates and the value network output:
        EFE_value_net_loss = F.mse_loss(EFE_estimate_batch, EFE_batch_t1)
        
        return EFE_value_net_loss
    
    def compute_VFE(self, obs_batch_t1, pred_error_batch_t0t1):
        
        # Determine the action distribution for time t1:
        policy_batch_t1 = self.policy_net(obs_batch_t1)
        
        # Determine the EFEs for time t1:
        EFEs_batch_t1 = self.value_net(obs_batch_t1).detach()

        # Take a gamma-weighted Boltzmann distribution over the EFEs:
        boltzmann_EFEs_batch_t1 = torch.softmax(-self.gamma * EFEs_batch_t1, dim=1).clamp(min=1e-9, max=1-1e-9)
        
        # Weigh them according to the action distribution:
        energy_batch = -(policy_batch_t1 * torch.log(boltzmann_EFEs_batch_t1)).sum(-1).view(self.batch_size, 1)
        
        # Determine the entropy of the action distribution
        entropy_batch = -(policy_batch_t1 * torch.log(policy_batch_t1)).sum(-1).view(self.batch_size, 1)
        
        # Determine the VFE, then take the mean over all batch samples:
        VFE_batch = pred_error_batch_t0t1 + (energy_batch - entropy_batch)
        VFE = torch.mean(VFE_batch)
        
        return VFE
        
    def learn(self):
        
        # If there are not enough transitions stored in memory, return:
        if self.memory.push_count - self.max_n_indices*2 < self.batch_size:
            return
        
        # After every freeze_period time steps, update the target network:
        if self.freeze_cntr % self.freeze_period == 0:
            self.target_net.load_state_dict(self.value_net.state_dict())
        self.freeze_cntr += 1
        
        # Retrieve transition data in mini batches:
        (
            obs_batch_t0, obs_batch_t1, obs_batch_t2, action_batch_t0,
            action_batch_t1, reward_batch_t1, terminated_batch_t2, truncated_batch_t2, 
            pred_error_batch_t0t1
        ) = self.get_mini_batches()

        # Compute the value network loss:
        value_net_loss = self.compute_EFE_value_net_loss(
            obs_batch_t1, obs_batch_t2, 
            action_batch_t1, reward_batch_t1,
            terminated_batch_t2, truncated_batch_t2, pred_error_batch_t0t1
        )
        
        # Compute the variational free energy:
        VFE = self.compute_VFE(obs_batch_t1, pred_error_batch_t0t1)
        
        # Reset the gradients:
        self.transition_net.optimizer.zero_grad()
        self.policy_net.optimizer.zero_grad()
        self.value_net.optimizer.zero_grad()
        
        # Compute the gradients:
        VFE.backward()
        value_net_loss.backward()
        
        # Perform gradient descent:
        self.transition_net.optimizer.step()
        self.policy_net.optimizer.step()
        self.value_net.optimizer.step()
        
    def train(self):

        filename = f"Deep_AIF_MDP_Lunar_Lander_v2"
        figure_file = f"plots/{filename}.png"

        msg = "Environment is: {}\nTraining started at {}".format(self.env.unwrapped.spec.id, datetime.datetime.now())
        print(msg)
        print(f"\n----------------- Deep (EFE-Bootstrapping) Active Inference -----------------\n")
        if self.keep_log:
            self.record.write(msg+"\n")
        
        results = []

        for ith_episode in range(self.n_episodes):
            
            total_reward = 0
            obs, info = self.env.reset()
            obs = torch.tensor(obs, dtype=torch.float32, device=self.device) # OG

            terminated = False
            truncated = False
            done = terminated or truncated

            reward = 0

            while not done:

                #######################################################################################################
                # # ACTIVE INFERENCE TREE SEARCH (MDP)

                # # NEW CODE: Do Active Inference Tree Search:
                # root_node = self.active_inferece_tree_search(initial_state_belief = obs, delta = 0.95, epsilon = 0.4)

                # # NEW CODE: Select the best action based on the updated tree
                # action = self.select_action_AcT(root_node) # self.select_action(obs)
                #######################################################################################################

                # DEEP ACIVE INFERENCE: Select an action - dAIF:
                action = self.select_action_dAIF(obs)

                # remember the transition:
                self.memory.push(obs, action, reward, terminated, truncated)

                # # ACTIVE INFERENCE TREE SEARCH: Execute the selected action and transition to a new state - AcT:
                # obs, reward, terminated, truncated, _  = self.env.step(action)

                # DEEP ACTIVE INFERENCE: Execute the selected action and transition to a new state - dAIF:
                obs, reward, terminated, truncated, _  = self.env.step(action[0].item())

                # cast the observation to the correct dtype:
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)

                # increment the per-episode reward:
                total_reward += reward
                
                # learn:
                self.learn()

                if terminated or truncated:
                    done = True
                    self.memory.push(obs, -99, -99, terminated, truncated)

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
                torch.save(self.transition_net.state_dict(), "networks/intermediary/intermediary_transnet{}_{:d}.pth".format(self.run_id, ith_episode))
                torch.save(self.policy_net.state_dict(), "networks/intermediary/intermediary_polnet{}_{:d}.pth".format(self.run_id, ith_episode))
                torch.save(self.value_net.state_dict(), "networks/intermediary/intermediary_valnet{}_{:d}.pth".format(self.run_id, ith_episode))
        
        self.env.close()
        
        # If enabled, save the results and the network (state_dict)
        if self.save_results:
            np.savez("results/intermediary/intermediary_results{}_end".format(self.run_id), np.array(results))
            np.savez(self.results_path, np.array(results))
        if self.save_network:
            torch.save(self.transition_net.state_dict(), "networks/intermediary/intermediary_transnet{}_end.pth".format(self.run_id))
            torch.save(self.policy_net.state_dict(), "networks/intermediary/intermediary_polnet{}_end.pth".format(self.run_id))
            torch.save(self.value_net.state_dict(), "networks/intermediary/intermediary_valnet{}_end.pth".format(self.run_id))
            
            torch.save(self.transition_net.state_dict(), self.network_save_path.format("trans"))
            torch.save(self.policy_net.state_dict(), self.network_save_path.format("pol"))
            torch.save(self.value_net.state_dict(), self.network_save_path.format("val"))
        
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



