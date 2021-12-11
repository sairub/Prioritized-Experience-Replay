#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

class DQN(nn.Module):
    def __init__(self, ans, lr, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(DQN, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.ans = ans
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels= 32, kernel_size= (5,5), stride= (4,4))
        self.conv2 = nn.Conv2d(in_channels= 32, out_channels= 64, kernel_size= (3,3), stride= (2,2))
        self.conv3 = nn.Conv2d(in_channels= 64, out_channels= 64, kernel_size= (3,3), stride= (2,2))

        self.fc1 = nn.Linear(128, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        
        self.loss = nn.MSELoss()

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.conv1(state)
        x = F.relu(x)
        x = F.max_pool2d(x,(2, 2))
        
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,(2, 2))
        
        x = self.conv3(x)
        x = F.relu(x)
        
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions

