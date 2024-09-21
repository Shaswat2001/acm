from typing import Any
import jax
import argparse
from jax.random import split
import flax.linen as nn

from acm_planner.network.ddpg_critic import DDPGCritic
from acm_planner.common.utils import create_train_state,init_params

class DDPG:

    def __init__(self, key: jax.Array, args: argparse.ArgumentParser, policy: nn.Module) -> None:

        self.args = args
        self.policy = policy
        self.key = key
        
        self.reset()
        self.initialize_params()
    
    def reset(self) -> None:

        self.PolicyNetwork = self.policy(hidden_dim = self.args.act_hidden_dim, n_action = self.args.n_action, bound = self.args.max_action)
        self.TargetPolicyNetwork = self.policy(hidden_dim = self.args.act_hidden_dim, n_action = self.args.n_action, bound = self.args.max_action)

        self.QNetwork = DDPGCritic(hidden_dim=self.args.hidden_dim)
        self.TargetQNetwork = DDPGCritic(hidden_dim=self.args.hidden_dim)

    def initialize_params(self) -> None:

        policy_key,qnet_key,tpolicy_key,tqnet_key = split(self.key,num=4)
        policy_input = self.args.input_shape
        qnet_input = self.args.input_shape +  self.args.n_action

        policy_param = init_params(policy_key,input_dims=policy_input,model=self.PolicyNetwork)
        tpolicy_param = init_params(tpolicy_key,input_dims=policy_input,model=self.TargetPolicyNetwork)
        qnet_param = init_params(qnet_key,input_dims=qnet_input,model=self.QNetwork)
        tqnet_param = init_params(tqnet_key,input_dims=qnet_input,model=self.TargetQNetwork)

        self.PolicyState = create_train_state(policy_param, self.PolicyNetwork)
        self.TargetPolicyState = create_train_state(tpolicy_param, self.TargetPolicyNetwork)
        self.QState = create_train_state(qnet_param, self.QNetwork)
        self.TargetQState = create_train_state(tqnet_param, self.TargetQNetwork)

