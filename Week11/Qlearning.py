#!/usr/bin/python3
# -*- coding: utf-8 -*-
#參考（複製）：https://medium.com/pyladies-taiwan/reinforcement-learning-%E5%81%A5%E8%BA%AB%E6%88%BF-openai-gym-e2ad99311efc

from random import randint, random, shuffle, choice
import gym
import math
import numpy as np
from numpy import argmax
import time
from time import sleep
e_greedy = 0.3
lr = 0.99
Render = False


def choose_action(state, q_table, action_space, e_greedy):
    if  random() < e_greedy:
        return action_space.sample()
    else :
        return argmax(q_table[state])
def get_state(observation, n_buckets, state_bounds):
    state = [0]*len(observation)
    for i,s in enumerate(observation):
        l, u = state_bounds[i][0], state_bounds[i][1] # 每個 feature 值的範圍上下限
        if s <= l: # 低於下限，分配為 0
            state[i] = 0
        elif s >= u: # 高於上限，分配為最大值
            state[i] = n_buckets[i] - 1
        else: # 範圍內，依比例分配
            state[i] = int(((s - l) / (u - l)) * n_buckets[i])

    return tuple(state)        

env = gym.make('CartPole-v0')
env._max_episode_steps = 4000


get_epsilon = lambda i: max(0.01, min(1, 1.0 - math.log10((i+1)/25)))  # epsilon-greedy; 隨時間遞減
get_lr = lambda i: max(0.01, min(0.5, 1.0 - math.log10((i+1)/25))) # learning rate; 隨時間遞減 
gamma = 0.99 # reward discount factor

n_buckets = (1,1,6,3) 
n_actions = env.action_space.n


state_bounds = list(zip(env.observation_space.low, env.observation_space.high))
state_bounds[1] = [-0.5, 0.5]
state_bounds[3] = [-math.radians(50), math.radians(50)]

q_table = np.zeros(n_buckets + (n_actions,))

for i_episode in range(400):
    if i_episode >395: Render = True
    e_greedy = get_epsilon(i_episode)
    lr = get_lr(i_episode)
    
    
    observation = env.reset()
    reward = 0
    state = get_state(observation, n_buckets, state_bounds)
    for t in range(50000):
        if Render: env.render()
        #print(observation)
        action = choose_action(state, q_table, env.action_space, e_greedy)
        observation, reward, done, info = env.step(action)

        reward += reward
        next_state = get_state(observation, n_buckets, state_bounds)
        #print(observation.shape)
        q_next_max = np.amax(q_table[next_state]) # 進入下一個 state 後，預期得到最大總 reward
        q_table[state + (action,)] += lr * (reward + gamma * q_next_max - q_table[state + (action,)]) # 就是那個公式
        
        state = next_state
        if Render : sleep(1/15)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
