#!/usr/bin/env python3

import os
import pickle
import argparse
import sys
import numpy as np
import rclpy
import copy
import jax
import matplotlib.pyplot as plt
from acm_planner.environment.GazeboEnv.BaseGazeboACMEnv import BaseGazeboACMEnv
from acm_planner.common.arguments import *
from acm_planner.agent import DDPG
from acm_planner.network.base_net import ContinuousMLP
import gymnasium as gym
from scipy.interpolate import splrep, splev

def train(args,env,trainer):


    best_reward = -np.inf
    total_reward = []
    trainer.load(args.Environment)    
    s = env.reset(pose = np.array([0.0,0.0,2.0]))
    reward = 0
    poses_x = []
    poses_y = []
    while True:

        action = trainer.choose_action(s)
        # print(action)
        next_state,rwd,done,_,_ = env.step(action)
        poses_x.append(env.pose[0])
        poses_y.append(env.pose[1])
        trainer.add(s,action,rwd,next_state,done)
        trainer.learn()
        reward+=rwd

        if done:
            break
            
        s = next_state

    total_reward.append(reward)
    print("Avg Reward is ==> {}".format(total_reward))
    poses_x = np.array(poses_x)
    poses_y = np.array(poses_y)

    sorted_indices = np.argsort(poses_x)
    poses_x_sorted = poses_x[sorted_indices]
    poses_y_sorted = poses_y[sorted_indices]

    spline_params = splrep(poses_x_sorted, poses_y_sorted, s=0, k=4)

    # Generate dense x values for plotting the smooth spline
    x_dense = np.linspace(poses_x_sorted.min(), poses_x_sorted.max(), 300)
    y_spline = splev(x_dense, spline_params)

    # Plot the original data and the spline fit
    plt.plot(poses_x, poses_y, color='red', label='Data Points')
    plt.plot(x_dense, y_spline, color='green', label='Spline Fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Spline Fitting Example')
    plt.legend()
    plt.grid(True)
    plt.show()
if __name__=="__main__":

    rclpy.init(args=None)

    args = build_parse()

    env = BaseGazeboACMEnv()

    args = get_env_parameters(args,env)
    
    if args.Algorithm == "DDPG":
        args = get_ddpg_args(args)
        trainer = DDPG.DDPG(args = args,policy = ContinuousMLP)

    train(args,env,trainer)