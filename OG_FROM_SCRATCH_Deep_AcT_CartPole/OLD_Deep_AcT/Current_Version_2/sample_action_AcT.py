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




def sample_action(k_rho, E, gamma, expected_free_energy):
    # Calculate the argument for the Boltzmann distribution
    argument = k_rho * np.log(E) - gamma * expected_free_energy
    
    # Apply softmax to obtain action probabilities
    action_probabilities = np.exp(argument - np.max(argument))  # Subtract max value for numerical stability
    action_probabilities /= np.sum(action_probabilities)
    
    # Sample an action based on the probabilities
    sampled_action = np.random.choice(len(action_probabilities), p = action_probabilities)
    
    return sampled_action


if __name__ == '__main__':

	# Example values
	k_rho = 1.0
	E = np.array([0.3, 0.7])  # Example action prior probabilities for actions [0, 1]
	gamma = 0.5
	expected_free_energy = 0.2  # Replace with the actual expected free energy

	# Sample an action
	action = sample_action(k_rho, E, gamma, expected_free_energy)
	print("Sampled Action:", action)

