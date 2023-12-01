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


def kl_mvn(m0, S0, m1, S1):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    

    From wikipedia
    KL( (m0, S0) || (m1, S1))
         = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| + 
                  (m1 - m0)^T S1^{-1} (m1 - m0) - N )
    """

    print()

    dim = S0.shape

    S1 = np.transpose(S1)
    S0 = np.transpose(S0)

    S1 = np.dot(S1, np.identity(dim))

    print(f"S1:\n{S1}")

    # store inv diag covariance of S1 and diff between means
    N = m0.shape[0]
    iS1 = np.linalg.inv(S1)
    diff = m1 - m0

    # kl is made of three terms
    tr_term   = np.trace(iS1 @ S0)
    det_term  = np.log(np.linalg.det(S1)/np.linalg.det(S0)) #np.sum(np.log(S1)) - np.sum(np.log(S0))
    quad_term = diff.T @ np.linalg.inv(S1) @ diff #np.sum( (diff*diff) * iS1, axis=1)
    #print(tr_term,det_term,quad_term)
    return .5 * (tr_term + det_term + quad_term - N) 

def kl_divergence_diag_cov_gaussian(mu1, sigma1_sq, mu2, sigma2_sq):
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

    avg_kl_div = avg_kl_div / len(mu1)

    return avg_kl_div


if __name__ == '__main__':

    m_p = torch.tensor([1.0, 2.0, 3.0, 4.0])
    s_p = torch.tensor([0.1, 0.1, 0.1, 0.1])

    m_q = torch.tensor([2.0, 3.0, 4.0, 5.0])
    s_q = torch.tensor([0.1, 0.1, 0.1, 0.1])


    print(f"\nm_p: {m_p}")
    print(f"s_p: {s_p}")
    print(f"m_q: {m_q}")
    print(f"s_q: {s_q}")

    kl_1 = kl_divergence_diag_cov_gaussian(m_p, s_p, m_q, s_q)

    print(f"kl_1: {kl_1}\n")

