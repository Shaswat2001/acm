#!/usr/bin/env python3

import os
import pickle
import argparse
import sys
import numpy as np
import rclpy
import copy
import matplotlib.pyplot as plt
# sys.path.insert(0, '/Users/shaswatgarg/Documents/WaterlooMASc/StateSpaceUAV')
from acm_planner.common.arguments import *
from acm_planner.agent import DDPG
from acm_planner.network.base_net import ContinuousMLP
import gym

def train(args,env,trainer):

    best_reward = -np.inf
    total_reward = []
    avg_reward_list = []
    os.makedirs("config/saves/rl_rewards/" +args.Environment, exist_ok=True)
    os.makedirs("config/saves/images/" +args.Environment, exist_ok=True)
    
    for i in range(args.n_episodes):
        s = env.reset()
        reward = 0
        
        while True:

            action = trainer.choose_action(s)
            next_state,rwd,done,info = env.step(action)
            trainer.add(s,action,rwd,next_state,done)
            trainer.learn()
            reward+=rwd

            if done:
                break
                
            s = next_state

        total_reward.append(reward)
        avg_reward = np.mean(total_reward[-40:])
        if avg_reward>best_reward and i > 10:
            best_reward=avg_reward
            if args.save_rl_weights:
                print("Weights Saved !!!")
                trainer.save(args.Environment)

        print("Episode * {} * Avg Reward is ==> {}".format(i, avg_reward))
        avg_reward_list.append(avg_reward)

    if args.save_results:
        list_cont_rwd = [avg_reward_list]
        f = open("config/saves/rl_rewards/" +args.Environment + "/" + args.Algorithm + ".pkl","wb")
        pickle.dump(list_cont_rwd,f)
        f.close()

    plt.subplot(212)
    plt.title(f"Reward values - {args.Algorithm}")
    plt.xlabel("Iterations")
    plt.ylabel("Reward")
    plt.plot(avg_reward_list)
    plt.show()



if __name__=="__main__":

    rclpy.init(args=None)

    args = build_parse()

    env = gym.make("CartPole-v1")
    env.reset()

    args = get_env_parameters(args,env)
    
    if args.Algorithm == "DDPG":
        args = get_ddpg_args(args)
        trainer = DDPG.DDPG(args = args,policy = ContinuousMLP)

    train(args,env,trainer)