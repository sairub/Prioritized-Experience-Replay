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
import collections
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pylab

from Network import DQN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Agent():
    def __init__(self,ans,gamma,lr,epsilon,input_dims,n_actions,batch_size,max_mem_size=100,eps_end=0.01):
        self.input_dims = input_dims
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.lr = lr
        self.action_space = n_actions #Change this to n_actions for generalised application
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
            self.Q_eval = DQN(self.ans, lr, input_dims= [100800], fc1_dims = 300, fc2_dims = 100, n_actions = 6)
            
            self.priorities = collections.deque(maxlen=self.mem_size)

            self.state_memory = np.zeros((self.mem_size,*self.input_dims), dtype = np.float32)
            self.new_state_memory = np.zeros((self.mem_size,*self.input_dims), dtype = np.float32)
            self.action_memory = np.zeros(self.mem_size, dtype = np.int32) 
            self.reward_memory = np.zeros(self.mem_size, dtype = np.float32)
            self.terminal_memory = np.zeros(self.mem_size,dtype= bool)
            
    
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
            index = self.mem_cntr % self.mem_size 
            self.state_memory[index] = state
            self.action_memory[index] = action
            self.reward_memory[index] = reward
            self.new_state_memory[index] = state_
            self.terminal_memory[index] = done
            
            self.priorities.append(max(self.priorities, default=1))
            self.mem_cntr += 1
            
    def scaled_prob(self):
        P = np.array(self.priorities, dtype = np.float64)
        P /= P.sum() 
        return P
    
    def prob_imp(self, prob, beta):
        self.beta = beta
        i =  (1/self.mem_size * 1/prob)**(-self.beta)
        i /= max(i)
        return i
    
    def choose_action(self,state):
        if np.random.random() > self.epsilon:
            state = T.tensor([state]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions[1]).item()
        else:
            action = np.random.choice(self.action_space) #Here action_space = [0,1]
        
        return action
    
    def sample(self):
        self.beta = np.min([1., 0.001 + self.beta])
        
        max_mem = min(self.mem_cntr, self.mem_size)
        probability = self.scaled_prob()
        info = np.random.choice(max_mem, self.batch_size, replace=False, p = probability)
        imp = self.prob_imp(probability[info], self.beta)
   
        return imp, info
    
    def prop_priority(self, i, err, c = 1.1, alpha_value = 0.7): #for proportional priority
        self.priorities[i] = (np.abs(err) + c)** alpha_value
        
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
            if self.mem_cntr < self.batch_size:
                return
            
            self.Q_eval.optimizer.zero_grad()
            imp, info = self.sample()
            
            state_batch = T.tensor(self.state_memory[info]).to(self.Q_eval.device)
            new_state_batch = T.tensor(self.new_state_memory[info]).to(self.Q_eval.device)

            action_batch = self.action_memory[info]

            reward_batch = T.tensor(self.reward_memory[info]).to(self.Q_eval.device)
            terminal_batch = T.tensor(self.terminal_memory[info]).to(self.Q_eval.device)

            q_eval = self.Q_eval.forward(state_batch)
            q_next = self.Q_eval.forward(new_state_batch)

            q_target = reward_batch + self.gamma*T.max(q_next, dim=0)[0]
            
            i = np.arange(self.batch_size)
            diff = T.abs(q_eval - q_target)
            for i in range(self.batch_size):
                idx = info[i]
                self.prop_priority(idx, diff[i].detach().numpy())
                
            #loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
            loss = (T.FloatTensor(imp) * F.smooth_l1_loss(q_eval, q_target)).mean().to(self.Q_eval.device)
            
            loss.backward()
            self.Q_eval.optimizer.step()
