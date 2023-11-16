# def variational_inference_AcT(self, node):
#     '''
#     Sample an action from a Boltzmann distribution:
#         a_𝜏 ~ 𝜎(𝜅𝑝 ∙ ln(𝐄) − 𝛾_𝜏 ∙ G(𝜋_𝜏, 𝑣_𝜏))
#     where:
#         * 𝐄 is the prior belief about the policy.
#         * 𝛾_𝜏 is the precision, computed at each depth of the tree visit
#         * G(𝜋_𝜏, 𝑣_𝜏) = (𝐀 ∙ 𝐱_𝜏) ∙ (ln(𝐀 ∙ 𝐱_𝜏) − ln 𝐂) + 𝐱_𝜏 ∙ (∑_𝑖 𝐀𝑖𝑗 ∙ ln(𝐀𝑖𝑗))_𝑗 is the EFE
#
#     The latter term (𝐄) is modulated by a factor 𝜅𝑝 named the "exploration factor" and is closely related to the
#     factor 𝑐𝑝 in the UCB1 algorithm for the Multi-armed Bandits.
#
#     𝐄 takes the form of probabilistic distribution 𝐄 ~ sqrt(2 ∙ ln 𝑁(𝑣) ⁄ 𝑁(𝑣′)), where 𝑣′ denotes a child
#     node, 𝑁(𝑣′) denotes the number of visits of 𝑣′, and 𝑁(𝑣) denotes the number of visits of the parent
#     node 𝑣.
#
#     The goal of the first (variational inference) stage is to select the next non-terminal leaf node 𝑣𝜏 of the
#     tree to expand. From the root and recursively until reaching an expandable node of the planning
#     tree—this stage samples an action over a Boltzmann distribution 𝜎(𝜅𝑝 ln 𝐄 − 𝛾𝜏 ∙ G(𝜋𝜏, 𝑣𝜏))...
#
#     Conducts approximate Bayesian infernce over policy beliefs - replaces the Node method?
#     '''
#
#     # Compute 𝐄:
#     prior_belief_about_policy = np.array([math.sqrt(2 * math.log(node.visit_count) / child_node.visit_count) for child_node in node.children])
#
#     # Compute 𝛾_𝜏:
#     print(f"node.depth: {node.depth}")
#     precision_tau = self.update_precision(depth = node.depth, alpha = 1, beta = 1)
#
#     # Compute G(𝜋_𝜏, 𝑣_𝜏):
#     EFE_tau = self.value_net(node.state_belief).detach()
#     # EFE_tau = self.predictive_EFE_tau # ?????????????????????????
#
#     ###############################################################################################################################################
#     # # compute the argument to the Boltzmann distribution over actions:
#     # action_dist_arg = [(self.exploration_factor * np.log(policy_prior) - precision_tau * EFE_tau) for policy_prior in prior_belief_about_policy]
#     #
#     # # Construct the Boltzmann distribution over actions - posterior belief about actions (posterior action probabilities):
#     # action_probs = np.exp(action_dist_arg) / np.sum(np.exp(action_dist_arg))
#     ###############################################################################################################################################
#
#     # Construct the Boltzmann distribution over actions - posterior belief about actions (posterior action probabilities):
#     action_probs = self.update_action_posterior(node, prior_belief_about_policy, precision_tau, EFE_tau)
#     # action_probs = np.exp(prior_belief_about_policy) / np.sum(np.exp(prior_belief_about_policy)) # OG
#
#     # sample this distribution:
#     selected_child_node = random.choices(node.children, weights=action_probs)[0]
#
#     return selected_child_node



    # def path_integration_AcT(self, node, g_delta):
    #
    #     # print("\tINSIDE path_integration_AcT")
    #
    #     while node != None:
    #         node.visit_count += 1
    #         node.predictive_EFE_tau += (1 / node.visit_count) * (g_delta - node.predictive_EFE_tau)
    #
    #         # node.update_action_posterior_belief() # UPDATE APPROX ACTION POSTERIOR HERE?
    #
    #         node = node.parent

    # def variational_inference_AcT(self, node):
    #
    #     prior_belief_about_policy = []
    #
    #     if node.parent is None:
    #         # since the root_node has no parent, add a dummy value for its "parent"
    #         prior_belief_about_policy.append(1.0)
    #     else:
    #         for child_node in node.parent.children:
    #             prior_prob = math.sqrt(2 * math.log(node.parent.visit_count) / child_node.visit_count)
    #             prior_belief_about_policy.append(prior_prob)
    #
    #     action_probs = np.exp(prior_belief_about_policy) / np.sum(np.exp(prior_belief_about_policy))
    #
    #     selected_child_node = random.choices(node.parent.children, weights=action_probs)[0]
    #
    #     return selected_child_node



    # def variational_inference_AcT(self, node):
    #     '''
    #     POTENTIAL METHODS FOR HANDELING THE ROOT NODE'S "PARENT":
    #
    #     1. Add a "dummy parent" to the root node, with visit_count = 1?
    #     2. Introduce a conditional expression here - variational_inference_AcT - that handles this?
    #     '''
    #
    #     # if the input node is the root node
    #     if node.parent == None:
    #         # copy the root node and make it its own parent?
    #         # node.parent = node
    #         node.parent = Node(
    #             state_belief = torch.randn(4, device='cuda:0'), action_at_tau_minus_one = 0, parent = None
    #         )
    #         node.parent.visit_count += 1
    #         node.parent.children = node
    #         # since the node is its own parent, the probabilities list calculation won't work.
    #
    #     # calcualte the "E" prior:
    #     prior_belief_about_policy = []
    #     for child_node in node.parent.children:
    #         prior_prob = math.sqrt(2 * math.log(node.parent.visit_count) / child.visit_count)
    #         # prior_prob = math.sqrt(2 * math.log(node.visit_count) / child.visit_count)
    #         prior_belief_about_policy.append(prior_prob)
    #     prior_belief_sum = sum(prior_belief_about_policy)
    #     action_probs = [prior_prob / prior_belief_sum for prior_prob in prior_belief_sum]
    #     selected_child_node = random.choices(node.parent.children, weights = action_probs)[0]
    #     # selected_child_node = random.choices(node.children, weights = action_probs)[0]
    #
    #
    #     # exploration_factor = math.sqrt((2 * math.log(node.parent.visit_count)) / node.visit_count)
    #     # # exploration_factor = [math.sqrt((2 * math.log(node.visit_count)) / child.visit_count) for child in node.children] # the actual specification)
    #     # probabilities = [torch.exp(exploration_factor * child.predictive_EFE_tau) for child in node.parent.children]
    #     # probabilities_sum = sum(probabilities)
    #     # action_probs = [prob / probabilities_sum for prob in probabilities]
    #     # a_prime = random.choices(node.parent.children, weights=action_probs)[0].action
    #     #
    #     # return self.find_child_node_AcT(node, a_prime)
    #
    #     return selected_child_node






    # def variational_inference_AcT(self, node):
    #     '''
    #     POTENTIAL METHODS FOR HANDELING THE ROOT NODE'S "PARENT":
    #
    #     1. Add a "dummy parent" to the root node, with visit_count = 1?
    #     2. Introduce a conditional expression here - variational_inference_AcT - that handles this?
    #     '''
    #
    #     # if the input node is the root node
    #     if node.parent == None:
    #         # copy the root node and make it its own parent?
    #         # node.parent = node
    #         node.parent = Node(
    #             state_belief = torch.randn(4, device='cuda:0'), action_at_tau_minus_one = 0, parent = None
    #         )
    #         node.parent.visit_count += 1
    #         node.parent.children = node
    #         # since the node is its own parent, the probabilities list calculation won't work.
    #
    #     print(f"\n\ntype(node.parent): {node.parent}") # node.parent is "None" initially !!!
    #     print(f"node.parent.visit_count: {node.parent.visit_count}")
    #     print(f"math.log(node.parent.visit_count): {math.log(node.parent.visit_count)}")
    #     print(f"(2 * math.log(node.parent.visit_count)): {(2 * math.log(node.parent.visit_count))}")
    #     print(f"node.visit_count: {node.visit_count}\n\n")
    #
    #     exploration_factor = math.sqrt((2 * math.log(node.parent.visit_count)) / node.visit_count) # AttributeError: 'NoneType' object has no attribute 'visit_count'
    #
    #     print(f"\n\nexploration_factor: {exploration_factor}")
    #     print(f"node.parent.children: {node.parent.children}")
    #     print(f"[torch.exp(exploration_factor * child.predictive_EFE_tau) for child in node.parent.children]: {[torch.exp(exploration_factor * child.predictive_EFE_tau) for child in node.parent.children]}")
    #     print(f"node.parent.children: {node.parent.children}\n\n")
    #
    #     # probabilities = [math.exp(exploration_factor * child.predictive_EFE_tau) for child in node.parent.children]
    #     probabilities = [torch.exp(exploration_factor * child.predictive_EFE_tau) for child in node.parent.children]
    #     probabilities_sum = sum(probabilities)
    #     action_probs = [prob / probabilities_sum for prob in probabilities]
    #
    #     print(f"\n\nnode.parent.children: {node.parent.children}")
    #     print(f"action_probs: {action_probs}\n\n")
    #
    #     print(f"\n\nrandom.choices(node.parent.children, weights=action_probs): {random.choices(node.parent.children, weights=action_probs)}")
    #     print(f"random.choices(node.parent.children, weights=action_probs)[0]: {random.choices(node.parent.children, weights=action_probs)[0]}")
    #
    #     a_prime = random.choices(node.parent.children, weights=action_probs)[0].action
    #
    #     return self.find_child_node_AcT(node, a_prime)


    # def variational_inference_AcT(self, node):
    #
    #     # prior_belief_about_policy = np.array([])
    #     # prior_belief_about_policy = [(math.sqrt(2 * math.log(node.visit_count)) / child_node.visit_count) for child_node in node.children]
    #
    #     # # if the input node is the root node:
    #     # if node.depth == 0:
    #     #     prior_belief_about_policy = np.array([math.sqrt(2 * math.log(node.visit_count) / child_node.visit_count) for child_node in node.children])
    #     # else:
    #     #     # prior_belief_about_policy = np.array([math.sqrt(2 * math.log(node.parent.visit_count) / child_node.visit_count) for child_node in node.children])
    #     #     prior_belief_about_policy = np.array([math.sqrt(2 * math.log(node.visit_count) / child_node.visit_count) for child_node in node.children])
    #
    #     prior_belief_about_policy = np.array([math.sqrt(2 * math.log(node.visit_count) / child_node.visit_count) for child_node in node.children])
    #
    #     # if node.depth == 0:
    #     #     prior_prob = math.sqrt(2 * math.log(node.visit_count) / node.visit_count)
    #     #     prior_belief_about_policy.append(prior_prob)
    #     # else:
    #     #     for child_node in node.children:
    #     #         prior_prob = math.sqrt(2 * math.log(node.visit_count) / child_node.visit_count)
    #     #         prior_belief_about_policy.append(prior_prob)
    #
    #     action_probs = np.exp(prior_belief_about_policy) / np.sum(np.exp(prior_belief_about_policy))
    #
    #     selected_child_node = random.choices(node.children, weights=action_probs)[0]
    #
    #     return selected_child_node

    # def variational_inference_AcT(self, node):
    #     '''
    #     This function is always returning the input node as the selected_child_node.
    #     '''
    #
    #     # print("INSIDE expand_AcT")
    #
    #     prior_belief_about_policy = [] # INVESTIGATE THIS
    #
    #     # if the input node is the root node:
    #     if node.depth == 0:
    #         print(f"node.depth: {node.depth}")
    #         prior_prob = math.sqrt(2 * math.log(node.parent.visit_count) / node.visit_count)
    #         prior_belief_about_policy.append(prior_prob)
    #     else:
    #         print(f"node.depth: {node.depth}")
    #         for child_node in node.parent.children:
    #             prior_prob = math.sqrt(2 * math.log(node.parent.visit_count) / child_node.visit_count)
    #             prior_belief_about_policy.append(prior_prob)
    #
    #     # Compute action probabilities using softmax function
    #     action_probs = np.exp(prior_belief_about_policy) / np.sum(np.exp(prior_belief_about_policy))
    #
    #     print(f"\n\nprior_belief_about_policy: {prior_belief_about_policy}\n\n")
    #
    #     if node.depth == 0:
    #         # print(f"\n\nnode.children: {node.children}")
    #         # print(f"node.parent.children: {node.parent.children}")
    #         # print(f"action_probs: {action_probs}")
    #         # print(f"random.choices(node.children, weights=action_probs): {random.choices(node.children, weights=action_probs)}\n\n")
    #         selected_child_node = random.choices(node.children, weights=action_probs)[0]
    #     else:
    #         selected_child_node = random.choices(node.parent.children, weights=action_probs)[0]
    #
    #     print(f"\nnode == selected_child_node: {node == selected_child_node}\n")
    #
    #     return selected_child_node


    # def variational_inference_AcT(self, node):
    #     '''
    #     This function is always returning the input node as the selected_child_node.
    #     '''
    #
    #     print("INSIDE expand_AcT")
    #
    #     prior_belief_about_policy = []
    #
    #     # POSSIBLY NEED TO DO THIS IMMIDIATELY UPON MAKING THE ROOT NODE
    #     if node.parent is None:
    #         # Create a "dummy parent" node for the root_node
    #         print("node.parent is the root node")
    #         dummy_parent = Node()
    #         dummy_parent.children = [node]
    #         dummy_parent.visit_count = 1
    #         prior_belief_about_policy.append(1.0)
    #         node.parent = dummy_parent
    #     else:
    #         print("node.parent is not the root node")
    #         for child_node in node.parent.children:
    #             prior_prob = math.sqrt(2 * math.log(node.parent.visit_count) / child_node.visit_count)
    #             prior_belief_about_policy.append(prior_prob)
    #
    #     # Compute action probabilities using softmax function
    #     action_probs = np.exp(prior_belief_about_policy) / np.sum(np.exp(prior_belief_about_policy))
    #
    #     selected_child_node = random.choices(node.parent.children, weights=action_probs)[0]
    #
    #     print(f"\nselected_child_node:\n{selected_child_node}")
    #     print(f"node == selected_child_node: {node == selected_child_node}\n")
    #
    #     return selected_child_node
