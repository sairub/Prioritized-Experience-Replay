#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from network import DQN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Agent():
    def __init__(self,ans,gamma,lr,epsilon,input_dims,n_actions,batch_size,max_mem_size=100,eps_end=0.01):
        self.input_dims = input_dims
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.lr = lr
        self.action_space = [0,1,2,3,4,5] #Change this to n_actions for generalised application
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.n_actions = n_actions
        self.beta = 0.4
        self.ans = ans
        
        if self.ans == 'e':
            self.Q_eval = DQN(self.ans, lr, input_dims= [100800], fc1_dims = 300, fc2_dims = 100, n_actions = 6)
            self.state_memory = np.zeros((self.mem_size,*self.input_dims), dtype = np.float32)
            self.new_state_memory = np.zeros((self.mem_size,*self.input_dims), dtype = np.float32)
            self.action_memory = np.zeros(self.mem_size, dtype = np.int32) 
            self.reward_memory = np.zeros(self.mem_size, dtype = np.float32)
            self.terminal_memory = np.zeros(self.mem_size,dtype= bool)
        
        if self.ans == 'p':
            env = gym.make('SpaceInvaders-v0')
            env.reset()
            observation, reward, done, info = env.step(3)
            self.memory = Replay_Agent_Memory(self.mem_size, self.input_dims, self.n_actions)
            self.Q_eval = DQN(self.ans, self.lr, input_dims= env.observation_space.shape, fc1_dims = 300, fc2_dims = 100, n_actions = env.action_space.n)
            self.Q_next = DQN(self.ans, self.lr, input_dims= env.observation_space.shape, fc1_dims = 300, fc2_dims = 100, n_actions = env.action_space.n)

    
    def store_transition(self,state,action,reward,state_,done):
        if self.ans == 'e':
            index = self.mem_cntr % self.mem_size #For only last 10,000 entries
            self.state_memory[index] = state
            self.new_state_memory[index] = state_ #Value for next state = state_
            self.action_memory[index] = action
            self.reward_memory[index] = reward
            self.terminal_memory[index] = done
                    
            self.mem_cntr += 1
            
        if self.ans == 'p':
            self.memory.store_transition(state, action, reward, state_, done)

        
    def choose_action(self,state):
        if np.random.random() > self.epsilon:
            state = T.tensor([state]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space) #Here action_space = [0,1]
        
        return action
    
    def sample(self):
        self.beta = np.min([1., self.beta+10e-3])
        state, action, reward, observation_, done, imp, info = self.memory.sample(self.batch_size, self.beta)  

        state = T.tensor(state).to(self.Q_eval.device)
        action = T.tensor(action).to(self.Q_eval.device)
        reward = T.tensor(reward).to(self.Q_eval.device)
        observation_ = T.tensor(observation_).to(self.Q_eval.device)
        complete = T.tensor(done).to(self.Q_eval.device)
        imp = T.tensor(imp, dtype= T.float32).to(self.Q_eval.device)
        info = T.tensor(info).to(self.Q_eval.device)
        return state, action, reward, observation_, complete, imp, info
    
    
    def save(self):
        self.Q_eval.save_checkpoint()
        self.Q_next.save_checkpoint()

    def load(self):
        self.Q_eval.load_checkpoint()
        self.Q_next.load_checkpoint()
        
    def learn(self):
        if self.ans == 'e':
            if self.mem_cntr < self.batch_size:
                return

            self.Q_eval.optimizer.zero_grad()

            max_mem = min(self.mem_cntr, self.mem_size)

            batch = np.random.choice(max_mem, self.batch_size, replace=False)
            batch_index = np.arange(self.batch_size, dtype=np.int32)

            state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
            new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)

            action_batch = self.action_memory[batch]

            reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
            terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

            q_eval = self.Q_eval.forward(state_batch)
            q_next = self.Q_eval.forward(new_state_batch)

            q_target = reward_batch + self.gamma*T.max(q_next, dim=0)[0]

            loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
            loss.backward()
            self.Q_eval.optimizer.step()
        
        if self.ans == 'p':
            if self.memory.mem_cntr < self.batch_size:
                return

            self.Q_eval.optimizer.zero_grad()

            states, actions, rewards, next_states, complete, imp, info = self.sample()  
            i = np.arange(self.batch_size)

            states = T.tensor(self.states[info]).to(self.Q_eval.device)
            next_states = T.tensor(self.next_states[info]).to(self.Q_next.device)

            v, a = self.Q_eval.forward(states)
            v_next, a_next = self.Q_next.forward(next_states)
            v_next_eval, a_next_eval = self.Q_eval(next_states)

            q_pred = T.tensor(v, (a - a.mean(dim=1, keepdim=True)))[i, actions]
            q_next = T.tensor(v_next,(a_next - a_next.mean(dim = 1, keepdim = True)))
            q_eval = T.tensor(v_next_eval, (a_next_eval-a_next_eval.mean(dim = 1, keepdim = True)))
            actionmax = T.argmax(q_eval, dim=1)

            q_next[complete] = 0.0
            
            #RL Main Equation - Bellman's 
            q_target = rewards + self.gamma * q_next[i, actionmax]

            #TD-Error
            diff = T.abs(q_pred - q_target)

            for i in range(self.batch_size):
                idx = info[i]
                self.memory.prop_priority(idx, diff[i].cpu().detach().numpy())

            loss = (T.cuda.FloatTensor(imp) * F.smooth_l1_loss(q_pred, q_target)).mean().to(self.Q_eval.device)
            loss.backward()

            losses.append(loss.item())

            self.Q_eval.optimizer.step()

