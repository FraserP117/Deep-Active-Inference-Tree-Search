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
import torch.distributions.multivariate_normal as mvn
from networkx.drawing.nx_pydot import graphviz_layout
import networkx as nx
import copy

import pdb


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
        return f"id = {str(self.name)},\nprev_action = {str(self.action_at_tau_minus_one)},\nused_actions = {str(self.used_actions)}"

def get_actions_AcT(node):

    unused_actions = []

    all_actions = np.array([0, 1])

    for action in all_actions:

        # print(f"action: {action}") # only ever "0""

        if action not in node.used_actions: # NOT CORRECT !!!!!
            unused_actions.append(action)

            return unused_actions

        else:
            # All actions have been used, so we simply return the best known action
            sorted_children = sorted(node.children, key = lambda child: child.predictive_EFE)
            action_with_minimum_EFE = sorted_children[0].action_at_tau_minus_one

            return action_with_minimum_EFE


if __name__ == '__main__':

	global node_label_counter

	node_label_counter = 0

	env = gym.make('CartPole-v1')

	noise_std = 0.1

	obs, info = env.reset()
	noisy_obs = obs + noise_std * np.random.randn(*obs.shape)

	state_belief_mean = torch.tensor([noisy_obs[0], noisy_obs[1], noisy_obs[2], noisy_obs[3]], dtype=torch.float32)
	state_belief_var = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.float32)

	# Initializing the planning tree
	root_node = Node(
		pred_mean_state = state_belief_mean,
		pred_var_state = state_belief_var,
		action_at_tau_minus_one = None, 
		parent = None,
		name = str(node_label_counter)
	)

	action_0 = torch.tensor([0])

	root_node