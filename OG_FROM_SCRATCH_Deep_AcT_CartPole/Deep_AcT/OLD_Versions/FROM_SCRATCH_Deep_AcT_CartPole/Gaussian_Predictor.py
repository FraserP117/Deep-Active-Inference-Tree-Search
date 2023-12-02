import torch
import torch.nn as nn

'''
Input and Output Size: The input to the neural network will be 
the state vector [x, x_dot, theta, theta_dot], which is a 
4-dimensional vector representing the environment state. The output 
of the network will be two vectors: the mean vector 
(also of size 4, one value for each dimension of the state), and the 
diagonal entries of the covariance matrix (also of size 4, representing 
the variances for each dimension).
'''

class GaussianPredictor(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(GaussianPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.fc3 = nn.Linear(hidden_size, hidden_size) # added

        self.mean_fc = nn.Linear(hidden_size, output_size)
        self.var_fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        x = torch.relu(self.fc3(x)) # added

        mean = self.mean_fc(x)
        var = torch.exp(self.var_fc(x)) # exp to ensure positivity

        return mean, var

def sample_from_gaussian(mean, var):
    noise = torch.randn_like(mean)
    sampled_state = mean + torch.sqrt(var) * noise
    return sampled_state


if __name__ == '__main__':
    gauss_predictor = GaussianPredictor(input_size = 4, hidden_size = 64, output_size = 4)