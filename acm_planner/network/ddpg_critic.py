from typing import Optional
import torch
from torch import nn 
from typing import Sequence

class DDPGCritic(nn.Module):

    def __init__(self, hidden_dim : Sequence[int]):

        super(DDPGCritic,self).__init__()

        layers = []
        for i in range(len(hidden_dim)-1):

            layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim[-1],1))

        self.network = nn.Sequential(*layers)

    def forward(self,state : torch.Tensor, action : torch.Tensor):

        ipt = torch.cat((state,action),dim=1)
        output = self.network(ipt)

        return output

