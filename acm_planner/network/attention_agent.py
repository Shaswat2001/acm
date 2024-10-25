import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import chain
from typing import Sequence

class AttentionAgent(nn.Module):
    """
    Attention network, used as critic for all agents. Each agent gets its own
    observation and action, and can also attend over the other agents' encoded
    observations and actions.
    """
    def __init__(self, n_traj: int, action_dim: int, traj_dim: int, hidden_dim: int,state_dim: int, attention_head: int, max_bound: Sequence[float], min_bound: Sequence[float],traj_bound: Sequence[float]):
        """
        Inputs:
            sa_sizes (list of (int, int)): Size of state and action spaces per
                                          agent
            hidden_dim (int): Number of hidden dimensions
            norm_in (bool): Whether to apply BatchNorm to input
            attend_heads (int): Number of attention heads to use (use a number
                                that hidden_dim is divisible by)
        """
        super(AttentionAgent, self).__init__()
        assert (hidden_dim % attention_head) == 0

        self.n_traj = n_traj
        self.time_steps = 3
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

        self.max_bound = torch.tensor(max_bound,dtype=torch.float32)
        self.min_bound = torch.tensor(min_bound,dtype=torch.float32)
        self.traj_bound = torch.tensor(traj_bound,dtype=torch.float32)
        self.trajnet = nn.Sequential(
            nn.Linear(n_traj,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,traj_dim),
            nn.Tanh(),
        )

        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()

        self.state_encoders = nn.ModuleList()

        for _ in range(n_traj):
        # iterate over agents
            encoder = nn.Sequential(
                nn.Linear(traj_dim, hidden_dim),
                nn.LeakyReLU()
                )
            
            critic = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, 1)
            )

            state_encoder = nn.Sequential(
                nn.Linear(traj_dim,hidden_dim),
                nn.LeakyReLU()
                )
            
            self.state_encoders.append(state_encoder)
            self.critic_encoders.append(encoder)
            self.critics.append(critic)

        attend_dim = hidden_dim // attention_head
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()

        for i in range(attention_head):
            self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(nn.Sequential(nn.Linear(hidden_dim,
                                                                attend_dim),
                                                       nn.LeakyReLU()))

        self.shared_modules = [self.key_extractors, self.selector_extractors,
                               self.value_extractors, self.critic_encoders]

    def shared_parameters(self):
        """
        Parameters shared across agents and reward heads
        """
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        """
        Scale gradients for parameters that are shared since they accumulate
        gradients from the critic loss function multiple times
        """
        for p in self.shared_parameters():
            p.grad.data.mul_(1. / self.nagents)

    def forward(self, state):
        """
        Inputs:
            inps (list of PyTorch Matrices): Inputs to each agents' encoder
                                             (batch of obs + ac)
            agents (int): indices of agents to return Q for
            return_q (bool): return Q-value
            return_all_q (bool): return Q-value for all actions
            regularize (bool): returns values to add to loss function for
                               regularization
            return_attend (bool): return attention weights per agent
            logger (TensorboardX SummaryWriter): If passed in, important values
                                                 are logged
        """
        action = self.network(state)
        action = action*self.max_bound
        action = torch.clip(action,self.min_bound,self.max_bound)
        trajectories = []

        mean = action[:,:3*self.time_steps]
        mean[:,3:6] += mean[:,0:3]
        mean[:,6:9] += mean[:,3:6]
        std = action[:,3*self.time_steps:]

        for _ in range(self.n_traj):

            trajectory = torch.normal(mean,std)
            trajectories.append(trajectory)

        inps_state = torch.stack(trajectories).permute(1,0,2)
        policy_input = []
        # extract state-action encoding for each agent
        ta_encodings = [self.critic_encoders[i](inps_state[:,i,:]) for i in range(inps_state.shape[1])]
        # ta_encodings = [encoder(inp) for encoder, inp in zip(self.critic_encoders, inps)]
        # extract state encoding for each agent that we're returning Q for
        s_encodings = [self.state_encoders[n_t](inps_state[:,n_t,:]) for n_t in range(self.n_traj)]
        # extract keys for each head for each agent
        all_head_keys = [[k_ext(enc) for enc in ta_encodings] for k_ext in self.key_extractors]
        # extract sa values for each head for each agent
        all_head_values = [[v_ext(enc) for enc in ta_encodings] for v_ext in self.value_extractors]
        # extract selectors for each head for each agent that we're returning Q for
        all_head_selectors = [[sel_ext(enc) for i, enc in enumerate(s_encodings) if i in range(self.n_traj)]
                            for sel_ext in self.selector_extractors]

        other_all_values = [[] for _ in range(self.n_traj)]
        all_attend_logits = [[] for _ in range(self.n_traj)]
        all_attend_probs = [[] for _ in range(self.n_traj)]
        # calculate attention per head
        for curr_head_keys, curr_head_values, curr_head_selectors in zip(
                all_head_keys, all_head_values, all_head_selectors):
            # iterate over agents
            for i, selector in zip(range(self.n_traj), curr_head_selectors):
                keys = [k for j, k in enumerate(curr_head_keys) if j != i]
                values = [v for j, v in enumerate(curr_head_values) if j != i]
                # calculate attention across agents
                key_stacked = torch.stack(keys).reshape(selector.shape[0],-1,self.n_traj-1)
                attend_logits = torch.matmul(selector.view(selector.shape[0], 1, -1),
                                            key_stacked)
                # scale dot-products by size of key (from Attention is All You Need)
                scaled_attend_logits = attend_logits / np.sqrt(keys[0].shape[1])
                attend_weights = F.softmax(scaled_attend_logits, dim=2)
                val_stacked = torch.stack(values).reshape(selector.shape[0],-1,self.n_traj-1)
                other_values = (val_stacked *
                                attend_weights).sum(dim=2)
                other_all_values[i].append(other_values)
                all_attend_logits[i].append(attend_logits)
                all_attend_probs[i].append(attend_weights)
        # calculate Q per agent
        all_rets = []

        for i in range(self.n_traj):
            critic_in = torch.cat((s_encodings[i], *other_all_values[i]), dim=1)
            all_q = self.critics[i](critic_in)
            all_rets.append(all_q)

        policy_input = torch.stack(all_rets).reshape(state.shape[0],-1)
        trajectory = self.trajnet(policy_input)

        trajectory = trajectory*self.traj_bound
        return trajectory

if __name__ == "__main__":

    agent = AttentionAgent(n_traj=100,action_dim = 18, traj_dim=9,hidden_dim=64,attention_head=4, state_dim=3, min_bound=np.array([-0.01]*9+[0.0]*9), max_bound=np.array([0.01]*9+[0.0]*9),traj_bound = np.array([0.1]*9))

    inp = torch.Tensor(np.random.uniform(size=(8,3)))

    print(agent(inp).shape)

