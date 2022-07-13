import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, linear_input_size, latent_space, outputs, device):
        super(DQN, self).__init__()
        self.device = device
        self.network = nn.Sequential(
          nn.Linear(linear_input_size, latent_space),
          nn.ReLU(),
          nn.Linear(latent_space, outputs),
        )

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(self.device)
        return self.network(x)
