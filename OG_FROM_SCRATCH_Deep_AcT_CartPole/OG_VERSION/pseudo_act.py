def do_aif_tree_search(A, B, C, D, delta):
    t = 0
    while not (halting conditions satisifed):
        x_t <- update expected state belief x_{t-1}, using s_t, o_t, a_{t-1}, A, B, D
        create a node V(x_t, a_{t-1})
        while delta ** tau < epsilon:
            v_t = TreePolicy(V, B)
            G_delta = Eval(v_tau, A, B, C, delta)
            PathIntegration(v_t, G_delta)
        s_{t+1}, o_{t+1}, x_t, a_t <- extract information saved in V
        t += 1

def TreePolicy(V):
    while not (V.is_terminal_leaf):
        if V not fully expanded:
            return Expansion(v, B)
        else:
            V <- VariationalInference(v)
    v_tau = V
    return v_tau

def Expansion(V, B):
    randomly draw an unused action a_prime on V
    for parent V, generate new child V_prime(x_prime, a_prime) with x_prime = B(a_prime) * x
    return V_prime

def VariationalInference(V):
    build the distribution E via the probability mas function sqrt((2*ln(N(V)) / (N(V_prime)))
    sample a_prime ~ Boltzmann(k_p * ln(E) = gamma * G_Delta(V_prime))
    return V_prime(a_prime)

def Eval(v_tau, A, B, C, delta):
    evaluate the expected fre energy: G(*, v_tau) through A, B, C
    return G_delta = delta ** tau * G(*, v_tau)

def PathIntegration(v_tau, G_delta):
    while v_tau not V:
        N(v_tau) += 1
        G(v_tau) = G(v_tau) + (1/(N(v_tau))) * (G_delta - G(v_tau))
        v_tau = parent of v_tau 