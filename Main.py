#!/usr/bin/env python
# coding: utf-8

# In[5]:


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
actions = env.action_space
observation, reward, done, info = env.step(3)
#imgplot = plt.imshow(observation)


# In[6]:


print("Enter 'e' for Experience Replay and 'p' for Prioritized Experience Replay:")
ans = input()


# In[7]:


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
    rewards = []

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

        rewards.append(reward)
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        episodes.append(i)

        print('episode ', i, 'score%.2f' % score,'average score %.2f' % avg_score)
    pylab.plot(episodes, scores, 'b')


# In[8]:


if ans == 'p':
    best_score = -np.inf
    prev_mean_score = -np.inf
    load_checkpoint = False
    n_games = 100
    
    agent = Agent(ans, gamma=0.99,lr=0.001,epsilon=0.1,input_dims= env.observation_space.shape,n_actions=env.action_space.n,batch_size=1,max_mem_size=1000,eps_end=0.01) 
    
    Q_eval = DQN(ans,lr = 0.001, input_dims= env.observation_space.shape, fc1_dims = 300, fc2_dims = 100, n_actions = env.action_space.n)
    Q_next = DQN(ans,lr = 0.001, input_dims= env.observation_space.shape, fc1_dims = 300, fc2_dims = 100, n_actions = env.action_space.n)

    if load_checkpoint: 
        agent.load()

    count = 0
    scores, eps_history, count_array, losses = [], [], [], []
    
    for i in range(n_games):
        done = False
        observation = env.reset()
        prev_life_ = 0
        store_when_complete = True
        frames = []
        score = 0

        while not done:

            if i  < 275:
                if i % 25 == 0:
                    frames.append(env.render(mode='rgb_array'))
            else:
                if i % 5 == 0:
                    frames.append(env.render(mode='rgb_array'))

            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            if info['ale.lives'] >= prev_life_:
                store_when_complete = done
            else:
                store_when_complete = True
            prev_life_ = info['ale.lives']
            score += reward
            reward = clip_reward(reward)

            if not load_checkpoint:
                agent.store(observation, action, reward, observation_, int(store_when_complete))
                agent.learn()

            observation = observation_
            count += 1
          
        if i  < 275:
            if i % 25 == 0:
                save_frames_as_gif(frames,i)
        else:
            if i % 5 == 0:
                save_frames_as_gif(frames,i)
        
        
        scores.append(score)
        count_array.append(count)
        avg_score = np.mean(scores[-25:])
        print('episode', i, 'last score %.0f, average score %.2f, best score %.2,' %
            (score, avg_score, best_score)
    
        if avg_score > prev_mean_score:
            if not load_checkpoint:
                agent.save()
                prev_mean_score = avg_score              

        if score > best_score:    
            best_score = score


# In[ ]:




