from typing import Any
import numpy as np
import argparse
import torch
from torch import nn 
import os

from acm_planner.network.ddpg_critic import DDPGCritic
from acm_planner.common.exploration import OUActionNoise
from acm_planner.common.replay_buffer import ReplayBuffer
from acm_planner.common.utils import soft_update,hard_update

class DDPG:

    def __init__(self, args: argparse.ArgumentParser, policy: nn.Module) -> None:

        self.args = args
        self.policy = policy
        self.reset()
    
    def reset(self) -> None:
        
        self.PolicyNetwork = self.policy(n_traj=100,action_dim = self.args.n_action, traj_dim=9,hidden_dim=64,attention_head=4, state_dim=self.args.input_shape, min_bound=self.args.max_action, max_bound=self.args.min_action,traj_bound = np.array([0.1]*9))
        self.PolicyOptimizer = torch.optim.Adam(self.PolicyNetwork.parameters(),lr=self.args.actor_lr)
        self.TargetPolicyNetwork = self.policy(n_traj=100,action_dim = self.args.n_action, traj_dim=9,hidden_dim=64,attention_head=4, state_dim=self.args.input_shape, min_bound=self.args.max_action, max_bound=self.args.min_action,traj_bound = np.array([0.1]*9))

        self.QNetwork = DDPGCritic(hidden_dim = self.args.hidden_dim)
        self.QOptimizer = torch.optim.Adam(self.QNetwork.parameters(),lr=self.args.critic_lr)
        self.TargetQNetwork = DDPGCritic(hidden_dim = self.args.hidden_dim)

        self.exploration = OUActionNoise(mean=np.zeros(9), std_deviation=float(0.01) * np.ones(9))
        self.buffer = ReplayBuffer(input_shape = self.args.input_shape, mem_size = self.args.mem_size, n_actions = 9)

        hard_update(self.TargetPolicyNetwork,self.PolicyNetwork)
        hard_update(self.TargetQNetwork,self.QNetwork)

        self.learning_step = 0

    def choose_action(self,state: np.ndarray,training_state: str = "training") -> np.ndarray:

        state = torch.Tensor(state)

        if training_state == "training":
            action = self.PolicyNetwork(state).detach().numpy()
            action += self.exploration()
        else:
            action = self.TargetPolicyNetwork(state).detach().numpy()

        return action
    
    def learn(self) -> None:
        
        self.learning_step+=1
        
        if self.learning_step<self.args.batch_size:
            return
        
        state,action,reward,next_state,done = self.buffer.shuffle()

        state = torch.Tensor(state)
        next_state = torch.Tensor(next_state)       
        action  = torch.Tensor(action)
        reward = torch.Tensor(reward)
        next_state = torch.Tensor(next_state)
        done = torch.Tensor(done)
        
        target_critic_action = self.TargetPolicyNetwork(next_state)
        target = self.TargetQNetwork(next_state,target_critic_action)
        y = reward + self.args.gamma*target*(1-done)
        critic_value = self.QNetwork(state,action)
        critic_loss = torch.mean(torch.square(y - critic_value),dim=1)
        self.QOptimizer.zero_grad()
        critic_loss.mean().backward()
        self.QOptimizer.step()

        actions = self.PolicyNetwork(state)
        critic_value = self.QNetwork(state,actions)
        actor_loss = -critic_value.mean()
        self.PolicyOptimizer.zero_grad()
        actor_loss.mean().backward()
        self.PolicyOptimizer.step()

        if self.learning_step%self.args.target_update == 0:                
            soft_update(self.TargetPolicyNetwork,self.PolicyNetwork,self.args.tau)
            soft_update(self.TargetQNetwork,self.QNetwork,self.args.tau)
    
    
    def add(self,s,action,rwd,next_state,done) -> None:
        self.buffer.store(s,action,rwd,next_state,done)

    def save(self,env) -> None:
        print("-------SAVING NETWORK -------")

        os.makedirs("config/saves/training_weights/"+ env + "/ddpg_weights", exist_ok=True)
        torch.save(self.PolicyNetwork.state_dict(),"config/saves/training_weights/"+ env + "/ddpg_weights/actorWeights.pth")
        torch.save(self.QNetwork.state_dict(),"config/saves/training_weights/"+ env + "/ddpg_weights/QWeights.pth")
        torch.save(self.TargetPolicyNetwork.state_dict(),"config/saves/training_weights/"+ env + "/ddpg_weights/TargetactorWeights.pth")
        torch.save(self.TargetQNetwork.state_dict(),"config/saves/training_weights/"+ env + "/ddpg_weights/TargetQWeights.pth")

    def load(self,env) -> None:

        self.PolicyNetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/ddpg_weights/actorWeights.pth",map_location=torch.device('cpu')))
        self.QNetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/ddpg_weights/QWeights.pth",map_location=torch.device('cpu')))
        self.TargetPolicyNetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/ddpg_weights/TargetactorWeights.pth",map_location=torch.device('cpu')))
        self.TargetQNetwork.load_state_dict(torch.load("config/saves/training_weights/"+ env + "/ddpg_weights/TargetQWeights.pth",map_location=torch.device('cpu')))