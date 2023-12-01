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



'''
ACTIVE INFERENCE TREE SEARCH:

* The goal of Active Inference Tree Search (AcT) is to estimate the posterior over control states p(u),
from which the best action a_t can be sampled.

* This estimate is obtained from simulations, starting from the current state s_t,
with observation o_t, and proceed forward through specific branches of the decision tree,
corresponding to a series of paths/histories h_tau = {(v_t, u_t), (v_{t+1}, u_{t+1}), ..., (v_tau, *)}

* These simulations - the histories: h_tau - approximate, statistically the EFE value G_{pi} of the 
policy pi = {u_t, u_{t+1}, u_{t+2}, ...}. The larger the number of simulations - the more h_tau - the 
more reliable the approximation of G_{pi}. 

* This plainning process is iterated until one or more halting conditions are satisfied.
    - the depth: d of the planning tree depends on two control params:
        - the discont factor: delta and the discount horizon: epsilon.
    - the maximum depth of a planning tree - and hence, 
    the maximum number of simulations employed to create it - is fixed by imposing: delta^d < epsilon.

* Note: AcT is an algorithm to build a planning tree, not to select actions. 
It is assumed that after the planning tree has been built, the agent selects an action by sampling 
from the distribution of control states infered at the root node; and executes this action. The agent then
makes a transition to a new hidden state and recieves a new observation. The planing can then begin anew.  



THE FOUR STAGES OF AcT:

* AcT has 4 successive stages:
    - Variational Inference
    - Expansion
    - Evaluation 
    - Path Integration

* The four stages are repeated iteratively - at each timestep t - until a criterion is met to provide 
estimates of the G values of a tree node subnet.


1. VARIATIONAL INFERENCE

* The goal of this stage is to select the next non-terminal leaf node: v_tau of the tree to expand.
* From the root node - recursively, until reaching an expandable node of the planning tree - this stage
samples an action over a Boltzmann distribution 𝜎 (𝜅𝑝 * ln(𝐄) − 𝛾*𝜏 * G(𝜋_𝜏 , 𝑣_𝜏 )) that depends on three terms:
    - The EFE: G(pi_tau, v_tau)
    - The precision gamma_tau, computed at each depth of the tree 
    - The the prior belief about the policy: E
* These three terms define the estimated quality of a policy. They consider:
    - The divergence between preferences encoded in C and the expected outcomes A*x_tau and expected entropy of observations.
    - A modulation of the policy quality distribution that controls the stochasticity of action selection.
    - A confidence bound that regulates exploration.
* E is modulated by factor: k_p called the "explorarion factor", closely related to the factor c_p in UCB1 fo rmulti-armed bandits.
* E ~ sqrt(2*ln(v)/N(v')) takes the form of a probabilistic distribution, where v' denotes a child node and N(v) denotes the number 
of visits to node v. 


2. EXPANSION:

* The goal of this stage is to expand the non-terminal leaf node v_tau selected during the variational inference stage.
* This stage instantiates a new child node v' by implementing a random action u' among those previously unused
    - This means that u' should not have been used for any other child node of v_tau.
* Each child node v' stands for a future state x' an agent can visit, according to the transitions encoded in B. 
* The Variational Inference and Expansion stages both return a node, however the former selects a node and the latter creates a node.
* It's possible to merge stages 1 and 2 into a unique stage such as TreePolicy.


3. EVALUATION:

* The goal of this stage is to assign a value to leaf node v_tau expanded in stage 2. 
* The evaluation considers the EFE: G(*, v_tau), which is a function of the state and the observation associated with v_tau.
* G(*, v_tau) scores the EFE of node v_tau, not the sum of all EFEs from the root to v_tau.
* EFE then weighted by its "temporal precision": a discount factor, equal to delta^tau, that depends on an arbitrary param: delta
and the depth tau of the node v_tau.
* Resulting G_Delta = delta^tau * G(*, v_tau) called "predictive EFE" is finally assigned to v_tau.
* Evaluating the quality of a node does not require a random policy or "rollout".


4. PATH INTEGRATION:

* The goal of this stage is to adjust the G values of the tree nodes up to the root node v_tau by considering the new values obtained 
during stage 3. 
* The value estimated in stage 3 is used to update the quality of G and the number of visits N of the nodes on the tree path: v_tau, ..., v_t
obtained during stage 1 and 2 (TreePolicy).



TREE NODES:

* nu_tau is a tree node data structure at planning time tau.
nu_tau collects:
    * hidden state belief: x_tau = P(s_{t+1} | s_t, a_t, Θ) or s_{t+1}^hat = f_theta(s_t, a_t)
    * the EFE computed at time tau
    * the number of node visits
    * the depth of the node
    * a pointer to this node's parent node. 
    * a pointer - list of? - to its child node/s

* x_tau is computed as soon as a new node is generated in the expand step. 

* theta = parameters

* A = p(o_t | s_t, theta), likelihood of observation given hidden state
* B(u_t = pi^t) = p(s_{t+1} | s_t, pi, theta) for t > 0, hidden state transition dynamics
* C = p(o_t | theta)? prior distribution of observations 
* D = p(s_0 | theta), prior for the initial state 
* E = prior expectation of each policy
* G_{pi} = score of the "quality" of a generic policy "pi". 
    - G_{pi} can be viewed as the log prior of a given policy, conditioned 
    on the future state[s?] and observations, together with preferred outcomes. 

* p(pi | gamma, theta) = sigmoid(ln(E) - gamma * G_{pi} | theta)
* p(gamma | theta) ~ Gamma(alpha, beta)



POSSIBLE HALTING CONDITIONS:

1. delta^d < epsilon is DEFINITIVELY one halting condition.

2. Maximum Planning Iterations: Set a maximum number of planning iterations and halt the algorithm after reaching that limit.

3. Convergence of EFE Values: Monitor the convergence of the estimated EFE (Expected Free Energy) values and halt the algorithm 
when the change falls below a predefined threshold.

4. Time-based Halt: Specify a maximum allowable runtime for the algorithm and halt the execution if it exceeds this limit.

5. Quality Improvement Threshold: Monitor the improvement in the estimated policy quality during the planning iterations and 
halt the algorithm when the improvement falls below a threshold.


QUESTIONS and THINGS TO DO:

* Don't really understand E very well yet.
* Updating the action posterior - how do we do this exactly? with E? with a simple ratio of visits?
* W.R.T the unused_actions list for each node:
    - what happends when a node has used all possible actions to transition from itself to the next node?
* The predictive EFE - how is that caluclated again?
* Do I always need to use get_mini_batches when I want to use the transitionm network, or can I just directly 
pass inputs to the transition network?
* a_prime = what for initial root node?
* review the path_integration function



* Still need to implement:
    - node_is_terminal_leaf
    - node_fully_expanded
* Is there a difference between a node being "fully expanded" as opposed to just "expanded"?

'''

class Node:

    def __init__(self, hidden_state_belief = None, action_at_tau_minus_one = None, hidden_state_at_tau = None, parent = None):
        self.hidden_state_belief = hidden_state_belief # x_tau = B?
        self.predictive_EFE_tau = 0 # The EFE computed at time tau: delta^tau * G(*, v_tau) 
        self.visit_count = 0 # the number of times this node has been visited
        self.depth = 0 # the depth od the node from the root
        self.parent = parent # this node's parent node
        self.children = [] # this node's children

        self.action_at_tau_minus_one = action_at_tau_minus_one

        self.hidden_state_at_tau = hidden_state_at_tau # should I keep this? I think so...

        # self.action = action
        # self.cumulative_value = 0
        # self.precision_parameter = 0

    def compute_action_distribution(self):
        total_visits = sum(child.visit_count for child in self.children)
        action_probs = []
        for child in self.children:
            action_prob = child.visit_count / total_visits
            action_probs.append((child.action, action_prob))
        return action_probs


def sample_action(action_distribution):
    actions, probabilities = zip(*action_distribution)
    sampled_action = random.choices(actions, probabilities)[0]
    return sampled_action

def tree_policy(node, B):
    while not node_is_terminal_leaf(node):
        if not node_fully_expanded(node):
            return expand(node, B) # use a NEURAL NETOWRK??
        else:
            node = variational_inference(node)
    v_tau = node
    return v_tau

def expand(node, B):
    unused_actions = get_unused_actions(node)
    a_prime = random.choice(unused_actions)
    x_prime = B(a_prime) * node.state
    child = Node(x_prime, action=a_prime)
    node.children.append(child)
    return child

def variational_inference(node):
    exploration_factor = math.sqrt(2 * math.log(node.parent.visits) / node.visits)
    probabilities = [math.exp(exploration_factor * child.quality) for child in node.parent.children]
    probabilities_sum = sum(probabilities)
    action_probs = [prob / probabilities_sum for prob in probabilities]
    a_prime = random.choices(node.parent.children, weights=action_probs)[0].action
    return find_child_node(node, a_prime)

def evaluate(node, A, B, C, delta):
    g = expected_free_energy(node.state, node.observation, node.action, A, B, C) # NEURAL NETWORK APPROXIMATION
    g_delta = delta ** node.visits * g
    return g_delta

def path_integration(node, g_delta):
    while node != None:
        node.visits += 1
        node.quality += (1 / node.visits) * (g_delta - node.quality)
        node = node.parent

# Helper functions

def halting_conditions_satisfied(t)
    # simply test to see if within alllowable runtime - can make more sophisticated
    return t < 50000

def node_is_terminal_leaf(node):
    # Implement your own condition to check if the node is a terminal leaf
    return False

def node_fully_expanded(node):
    # Implement your own condition to check if the node is fully expanded
    return len(node.children) == num_possible_actions

def get_unused_actions(node):
    # Implement your own logic to retrieve the unused actions from the node
    return []

def find_child_node(node, action):
    for child in node.children:
        if child.action == action:
            return child

def update_expected_state_belief()
    # Implement your own function to update the expected state belief
    return x_t

def expected_free_energy(state, observation, action, A, B, C):
    # Implement your own function to calculate the expected free energy
    return g

def extract_information(node):

    # Extract information from the given node

    # s_t_minus_1 = node.hidden_state_belief  # Previous state
    a_t_minus_1 = node.action_at_tau_minus_one  # Previous action
    s_t = node.hidden_state_belief

    s_t_plus_one = transition_dynamics(s_t, a_t_minus_1)  # NEURAL NETWORK APPROXIMATION
    o_t_plus_one = observe(s_t_plus_one)  # NEURAL NETWORK APPROXIMATION
    x_t = node.hidden_state_belief
    a_t = node.action_at_tau_minus_one

    return s_t_plus_one, o_t_plus_one, x_t, a_t


##### ACTIVE INFERENCE TREE SEARCH - BEST SO FAR (INTEGRATE WITH ACTUAL CODE NOW) #####
def active_inferece_tree_search(initial_hidden_state_belief, delta, epsilon):
    '''
    initial_hidden_state_belief is the prior on hidden state beliefs.
    '''

    # Initialize the planning tree with the root node
    root_node = Node(
        hidden_state_belief = initial_hidden_state_belief, action_at_tau_minus_one = None, hidden_state_at_tau = None
    )

    # Perform Active Inference Tree Search iteratively
    t = 0

    while not halting_conditions_satisfied():

        # update expected state belief x_{t-1} using s_t, o_t, a_{t-1}, A, B, D
        x_t = update_expected_state_belief() # NEURAL NETWORK APPROXIMATION

        root_node.hidden_state_belief = x_t
        root_node.action_at_tau_minus_one = a_t_minus_one

        while delta ** root_node.visit_count < epsilon:

            # Perform Variational Inference and Expansion
            selected_node = tree_policy(root_node, B)
            new_node = expansion(selected_node)
            new_node.depth += selected_node.depth
            new_node.parent = selected_node
            selected_node.children.append(new_node)

            # Stage 3: Evaluation
            value = evaluate(root_node, A, B, C, delta)

            # Stage 4: Path Integration
            path_integration(root_node, value)

        # Update time step and other variables
        s_t_plus_1, o_t_plus_1, x_t, a_t = extract_information(root_node)

        # Not sure about the following:
        # root_node = Node(
        #     hidden_state_belief = x_t, action_at_tau_minus_one = a_t_minus_one, hidden_state_at_tau = s_t_plus_1
        # )

        t += 1

    # not sure about this
    return s_t_plus_1, o_t_plus_1, x_t, a_t

    # end active_inferece_tree_search

    # not sure about this
    # return planning_tree


if __name__ == '__main__':

    # set the initial hidden state belief: x_0:
    initial_hidden_state_belief = None

    # do AcT to build the search tree for optimal action calculation:
    s_t_plus_1, o_t_plus_1, x_t, a_t = active_inferece_tree_search(initial_hidden_state_belief)

    # Select the best action based on the updated tree
    best_action = select_action(root_node)

    # Execute the selected action and transition to a new state
    transition(best_action) # NEURAL NETWORK APPROXIMATION

    # Update the observation and continue with the next planning iteration
    update_observation() # NEURAL NETWORK APPROXIMATION







## CREATING A PLANNING TREE
# # Create root node
# root_node = Node(hidden_state_belief=root_state, action_at_tau_minus_one=None, hidden_state_at_tau=root_state)

# # Create child nodes
# child_node = Node(hidden_state_belief=state1, action_at_tau_minus_one=action1, hidden_state_at_tau=state1)

# # Connect child node to the root node
# root_node.children.append(child_node)




# # AcT From Appendix A: PSeudocode
# def do_aif_tree_search(self, A, B, C, D, delta, epsilon):
#     t = 0
#     while not halting_conditions_satisfied():
#         x_t = update_expected_state_belief() # update expected state belief x_{t_1} by using s_t, o_t, a_{t-1}, A, B, D
#         root = Node(x_t, a_t_minus_one) # 
#         while delta ** root.visits < epsilon:
#             v_t = tree_policy(root, B) # implements variational_inference and expansion
#             g_delta = evaluate(v_t, A, B, C, delta)
#             path_integration(v_t, g_delta)
#         s_t_plus_1, o_t_plus_1, x_t, a_t = extract_information(root)
#         t += 1

# # ACTIVE INFERENCE TREE SEARCH

# # Initialize the planning tree with the root node
# root_node = Node(hidden_state_belief=initial_state)

# # Perform Active Inference Tree Search iteratively
# while not halting_condition():

#     # Stage 1: Variational Inference
#     selected_node = variational_inference(root_node)

#     # Stage 2: Expansion
#     new_node = expansion(selected_node)

#     # Stage 3: Evaluation
#     value = evaluation(new_node)

#     # Stage 4: Path Integration
#     path_integration(new_node, value)

# # Select the best action based on the updated tree
# best_action = select_action(root_node)

# # Execute the selected action and transition to a new state
# transition(best_action)

# # Update the observation and continue with the next planning iteration
# update_observation()


#######################################################################################


# ##### ACTIVE INFERENCE TREE SEARCH - BEST SO FAR (INTEGRATE WITH ACTUAL CODE NOW) #####

# # Initialize the planning tree with the root node
# root_node = Node(
#     hidden_state_belief = initial_hidden_state_belief, action_at_tau_minus_one=None, hidden_state_at_tau = None
# )

# # Perform Active Inference Tree Search iteratively
# t = 0

# while not halting_conditions_satisfied():

#     # update expected state belief x_{t-1} using s_t, o_t, a_{t-1}, A, B, D
#     x_t = update_expected_state_belief()

#     root_node.hidden_state_belief = x_t
#     root_node.action_at_tau_minus_one = a_t_minus_one

#     while delta ** root_node.visit_count < epsilon:

#         # Perform Variational Inference and Expansion
#         selected_node = tree_policy(root_node, B)
#         new_node = expansion(selected_node)
#         new_node.depth += selected_node.depth
#         new_node.parent = selected_node
#         selected_node.children.append(new_node)

#         # Stage 3: Evaluation
#         value = evaluate(root_node, A, B, C, delta)

#         # Stage 4: Path Integration
#         path_integration(root_node, value)

#     # Update time step and other variables
#     s_t_plus_1, o_t_plus_1, x_t, a_t = extract_information(root_node)
#     t += 1

# # Select the best action based on the updated tree
# best_action = select_action(root_node)

# # Execute the selected action and transition to a new state
# transition(best_action)

# # Update the observation and continue with the next planning iteration
# update_observation()


# ##### ACTIVE INFERENCE TREE SEARCH - BEST SO FAR (INTEGRATE WITH ACTUAL CODE NOW) #####

# def active_inferece_tree_search(initial_hidden_state_belief):
#     '''
#     initial_hidden_state_belief is the prior on hidden state beliefs.
#     '''

#     # Initialize the planning tree with the root node
#     root_node = Node(
#         hidden_state_belief = initial_hidden_state_belief, action_at_tau_minus_one = None, hidden_state_at_tau = None
#     )

#     # Perform Active Inference Tree Search iteratively
#     t = 0

#     while not halting_conditions_satisfied():

#         # update expected state belief x_{t-1} using s_t, o_t, a_{t-1}, A, B, D
#         x_t = update_expected_state_belief()

#         root_node.hidden_state_belief = x_t
#         root_node.action_at_tau_minus_one = a_t_minus_one

#         while delta ** root_node.visit_count < epsilon:

#             # Perform Variational Inference and Expansion
#             selected_node = tree_policy(root_node, B)
#             new_node = expansion(selected_node)
#             new_node.depth += selected_node.depth
#             new_node.parent = selected_node
#             selected_node.children.append(new_node)

#             # Stage 3: Evaluation
#             value = evaluate(root_node, A, B, C, delta)

#             # Stage 4: Path Integration
#             path_integration(root_node, value)

#         # Update time step and other variables
#         s_t_plus_1, o_t_plus_1, x_t, a_t = extract_information(root_node)



#         t += 1

#         # end active_inferece_tree_search

#     return planning_tree

# # Select the best action based on the updated tree
# best_action = select_action(root_node)

# # Execute the selected action and transition to a new state
# transition(best_action)

# # Update the observation and continue with the next planning iteration
# update_observation()




# def do_aif_tree_search(A, B, C, D, delta, epsilon): 
#     '''
#     EVENTUALLY REPLACE THIS WITH THE CODE IN MAIN EXEC
#     '''
#     t = 0
#     root = Node(initial_state)  # Create the root node
#     while not halting_conditions_satisfied():
#         x_t = update_expected_state_belief()  # Implement your own function for updating the expected state belief
#         root = Node(x_t)
#         while delta ** root.visits < epsilon:
#             v_t = tree_policy(root, B)
#             g_delta = evaluate(v_t, A, B, C, delta)
#             path_integration(v_t, g_delta)
#         s_t_plus_1, o_t_plus_1, x_t, a_t = extract_information(root)
#         t += 1

#     action_distribution = root.compute_action_distribution()
#     sampled_action = sample_action(action_distribution)
#     # The sampled_action is the action the agent can take