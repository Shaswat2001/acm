from typing import Optional
import torch
from torch import nn 
from typing import Sequence

class ContinuousMLP(nn.Module):

    def __init__(self, hidden_dim : Sequence[int], n_action: int, bound: Sequence[float]):

        super(ContinuousMLP,self).__init__()
        self.bound = torch.tensor(bound,dtype=torch.float32)

        layers = []
        for i in range(len(hidden_dim)-1):

            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim[-1],n_action))
        layers.append(nn.Tanh())

        self.network = nn.Sequential(*layers)

    def forward(self,state : torch.Tensor):

        output = self.network(state)
        output = output*self.bound

        return output

