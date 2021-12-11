#!/usr/bin/env python
# coding: utf-8

# In[30]:


import gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch as T
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pylab

from Prioritized_Replay import Replay_Agent_Memory
from Agent import Agent
from Network import DQN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env = gym.make('SpaceInvaders-v0')
env.reset()

height, width, channels = env.observation_space.shape
print(env.observation_space.shape)
print(env.action_space)
actions = env.action_space
print(env.unwrapped.get_action_meanings())
observation, reward, done, info = env.step(15)
#imgplot = plt.imshow(observation)


# In[31]:


print("Enter 'e' for Experience Replay and 'p' for Prioritized Experience Replay:")
ans = input()


# In[32]:


if ans == 'e':
    agent = Agent(ans,gamma=0.99,lr=0.001,epsilon=0.1,input_dims= env.observation_space.shape,n_actions=env.action_space.n,batch_size=1,max_mem_size=1000,eps_end=0.01) 
    Q_eval = DQN(ans,lr = 0.001, input_dims= [100800], fc1_dims = 300, fc2_dims = 100, n_actions = 6)
    
    x1 = observation[:,:,0]
    x2 = observation[:,:,1]
    x3 = observation[:,:,2]
    observation_new = np.zeros((3,210,160), dtype = np.float32)
    observation_new[0] = x1 
    observation_new[1] = x2
    observation_new[2] = x3

    state = T.tensor([observation_new]).to(Q_eval.device)
    actions = Q_eval.forward(state)
    action = T.argmax(actions).item()

    action_space = [0,1,2,3,4,5]
    action = np.random.choice(action_space)

    env.reset()

    agent = Agent(ans, gamma=0.99,lr=0.001,epsilon=0.1,input_dims= observation_new.shape ,n_actions=env.action_space.n,batch_size=1,max_mem_size=1000,eps_end=0.01) 

    scores = []
    n_games = 10
    losses = []
    episodes = []

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        x1 = observation[:,:,0]
        x2 = observation[:,:,1]
        x3 = observation[:,:,2]
        observation_new = np.zeros((3,210,160), dtype = np.float32)
        observation_new[0] = x1 
        observation_new[1] = x2
        observation_new[2] = x3


        while not done:

            action = agent.choose_action(observation_new)
            observation_, reward, done, info = env.step(action)
            x1 = observation_[:,:,0]
            x2 = observation_[:,:,1]
            x3 = observation_[:,:,2]
            observation_new_ = np.zeros((3,210,160), dtype = np.float32)
            observation_new_[0] = x1 
            observation_new_[1] = x2
            observation_new_[2] = x3
            #print(reward)

            #env.render()
            score += reward
            agent.store_transition(observation_new, action, reward, observation_new_, done)
            agent.learn()
            observation_new = observation_new_

        scores.append(score)

        avg_score = np.mean(scores[-100:])
        episodes.append(i)

        print('episode ', i, 'score%.2f' % score,'average score %.2f' % avg_score)
    pylab.plot(episodes, scores, 'b')


# In[33]:


if ans == 'p':
    agent = Agent(ans, gamma=0.99,lr=0.001,epsilon=0.1,input_dims= env.observation_space.shape,n_actions=env.action_space.n,batch_size=1,max_mem_size=1000,eps_end=0.01) 
    
    Q_eval = DQN(ans,lr = 0.001, input_dims= env.observation_space.shape, fc1_dims = 300, fc2_dims = 100, n_actions = env.action_space.n)
    Q_next = DQN(ans,lr = 0.001, input_dims= env.observation_space.shape, fc1_dims = 300, fc2_dims = 100, n_actions = env.action_space.n)
    
    x1 = observation[:,:,0]
    x2 = observation[:,:,1]
    x3 = observation[:,:,2]
    observation_new = np.zeros((3,210,160), dtype = np.float32)
    observation_new[0] = x1 
    observation_new[1] = x2
    observation_new[2] = x3

    state = T.tensor([observation_new]).to(Q_eval.device)
    actions = Q_eval.forward(state)

    env.reset()
    action = T.argmax(actions[1]).item()

    action_space = np.arange(0,18,1)
    action = np.random.choice(action_space)
    
    agent = Agent(ans, gamma=0.99,lr=0.001,epsilon=0.1,input_dims= observation_new.shape ,n_actions=env.action_space.n,batch_size=1,max_mem_size=1000,eps_end=0.01) 

    scores = []
    n_games = 10
    losses = []
    episodes = []

    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        x1 = observation[:,:,0]
        x2 = observation[:,:,1]
        x3 = observation[:,:,2]
        observation_new = np.zeros((3,210,160), dtype = np.float32)
        observation_new[0] = x1 
        observation_new[1] = x2
        observation_new[2] = x3


        while not done:

            #action = agent.choose_action(observation_new)
            observation_, reward, done, info = env.step(15)
            x1 = observation_[:,:,0]
            x2 = observation_[:,:,1]
            x3 = observation_[:,:,2]
            observation_new_ = np.zeros((3,210,160), dtype = np.float32)
            observation_new_[0] = x1 
            observation_new_[1] = x2
            observation_new_[2] = x3
            #print(reward)

            #env.render()
            score += reward
            agent.store_transition(observation_new, action, reward, observation_new_, done)
            agent.learn()
            observation_new = observation_new_

        scores.append(score)

        avg_score = np.mean(scores[-100:])
        episodes.append(i)
        print('episode ', i, 'score%.2f' % score,'average score %.2f' % avg_score)
    pylab.plot(episodes, scores, 'b')


# In[ ]:





