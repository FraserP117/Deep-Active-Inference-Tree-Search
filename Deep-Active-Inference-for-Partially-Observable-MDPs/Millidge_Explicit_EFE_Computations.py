def immediate_efe(s_t, o_t, r, Q_prior, Q_posterior):
    """
    Compute the immediate EFE for a given state s_t and observation o_t.
    
    Parameters:
        s_t: Current state
        o_t: Current observation
        r: Reward function for the observation o_t
        Q_prior: Prior distribution over s_t
        Q_posterior: Posterior distribution over s_t given o_t
        
    Returns:
        G(s_t, o_t): The immediate expected free energy (EFE) at time step t.
    """
    # Equation 15: immediate EFE
    efe = -r(o_t) + Q_prior.log_prob(s_t) - Q_posterior.log_prob(s_t)
    return efe


def total_efe(s_t, o_t, r, Q_prior, Q_posterior, action_space, depth=2):
    """
    Compute the total EFE over a ply-2 binary tree (depth=2).
    
    Parameters:
        s_t: Current state at time step t
        o_t: Current observation at time step t
        r: Reward function for observations
        Q_prior: Prior distribution over s_t
        Q_posterior: Posterior distribution over s_t given o_t
        action_space: Set of possible actions
        depth: Current depth of the binary tree (ply-2 in this case)
        
    Returns:
        G_total: Total expected free energy (EFE) for the policy.
    """
    if depth == 0:
        # Base case: return the immediate EFE at the leaf node
        return immediate_efe(s_t, o_t, r, Q_prior, Q_posterior)
    
    # Compute the immediate EFE at the current node (time t or t+1)
    G_current = immediate_efe(s_t, o_t, r, Q_prior, Q_posterior)
    
    # Recursively compute the EFE for future states (for actions a_t^1 and a_t^2)
    G_future = 0
    for action in action_space:
        # Simulate the next state and observation based on the action
        s_next, o_next, r_next = transition_model(s_t, action)  # Simulate the state transition and observation
        
        # Recursively calculate the EFE for the next depth (t+1 or t+2)
        G_future += total_efe(s_next, o_next, r_next, Q_prior, Q_posterior, action_space, depth-1)
    
    # Average the future EFEs over the actions (assumes uniform probability over actions)
    G_future /= len(action_space)
    
    # Return the total EFE: current EFE + expected future EFE
    return G_current + G_future


def compute_policy_efe(s_0, o_0, r, Q_prior, Q_posterior, action_space):
    """
    Compute the EFE for all policies in a binary tree with ply-2.
    
    Parameters:
        s_0: Initial state at time t
        o_0: Initial observation at time t
        r: Reward function for observations
        Q_prior: Prior distribution over states
        Q_posterior: Posterior distribution over states given observations
        action_space: Set of possible actions (binary tree, so 2 actions at each step)
    
    Returns:
        policy_efes: A dictionary mapping policies to their EFEs.
    """
    policy_efes = {}
    
    # There are 2 actions at each node in a binary tree
    for action_1 in action_space:
        # First action at time t
        s_1, o_1 = transition_model(s_0, action_1)
        
        for action_2 in action_space:
            # Second action at time t+1
            s_2, o_2 = transition_model(s_1, action_2)
            
            # Calculate the total EFE for this policy (action_1 -> action_2)
            policy = (action_1, action_2)  # Represents the policy (sequence of actions)
            G_total = total_efe(s_0, o_0, r, Q_prior, Q_posterior, action_space, depth=2)
            
            # Store the EFE for this policy
            policy_efes[policy] = G_total
    
    return policy_efes


# Example usage:
s_0 = initial_state()  # Initial state at time t
o_0 = initial_observation()  # Initial observation at time t
r = reward_function  # Reward function
Q_prior = prior_distribution  # Prior over states
Q_posterior = posterior_distribution  # Posterior over states given observations
action_space = [a1, a2]  # Binary action space

policy_efes = compute_policy_efe(s_0, o_0, r, Q_prior, Q_posterior, action_space)

# Output the EFE for each policy
for policy, efe in policy_efes.items():
    print(f"Policy {policy}: EFE = {efe}")
