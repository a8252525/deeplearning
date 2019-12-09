#!/home/tedbest2/pytorch_g/bin/python

import gym
import logging
import torch
import numpy as np
from network import *
import torchvision
from torchvision import transforms

def convert_gym_state(state):
    x = torch.tensor(state)

    return x

def choose_action(state):
    if np.random < epsilon:
        return state.action_space.sample()
    else:
        #
        #tmp = network.evaluation_network(state)
        #return action = argmax(tmp[0])
        #havn't finish 


action_space
action_spacein__':
action_spacefig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
action_spacefig of this game
    env = gym.make('Breakout-v0')
    # observation_space: Box(210,160,3)
    NUM_STATES = env.observation_space.shape
    # action_space: Discrete(3)
    NUM_ACTIONS = env.action_space.n
    NUM_EPOSIDES = 4000
    NUM_BATCHES = 32
    INITIAL_EPSILON = 0.4
    #FINAL_EPSILON = 0.05
    #EPSILON_DECAY = 1000000
    TRAINING_CYCLE = 2000
    TARGET_UPDATE_CYCLE = 100
    epsilon = INITIAL_EPSILON

    get_epsilon = lambda i: max(0.01, min(1, 1.0 - math.log10((i+1)/25)))  # epsilon-greedy; 隨時間遞減
    state = env.reset()
    
    
    outdir = './results'
    network = DeepQNetwork(NUM_STATES,NUM_BATCHES,NUM_ACTIONS,TRAINING_CYCLE,TARGET_UPDATE_CYCLE,False)
    for eposide in (NUM_EPOSIDES):
        epsilon = get_epsilon(eposide)
        lr = get_lr(eposide)
        #let lr and epsilon change with time

        episode_reward = 0
        state = env.reset()
        state = convert_gym_state(state)
        t = 0
        while True:
            action = choose_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = convert_gym_state(next_state)

            #remember experience until out of memory_capacity
            network.append_experience({'state':state,
            'action':[action],'reward':[reward],'next_state': next_state})
            #
            episode_reward += reward
            
            #out of memory_capacity then learn
            if memory_buffer > memory_capacity:
                network.train()
                network.delete_experience()
            if done:
                logging.info('Episode {} finished after {} timesteps, total rewards {}'.format(episode, t+1, episode_reward))
                break
            t+=1

        if episode % 5 == 0:
            #chage save weight to pytorch
            # network.save_weights("rl.h5")
    env.close()





