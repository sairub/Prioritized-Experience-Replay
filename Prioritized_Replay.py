#!/usr/bin/env python
# coding: utf-8

# In[1]:


import collections
import numpy as np

class Replay_Agent_Memory():
    def __init__(self, max_size, input_shape, n_actions):
        self.size = max_size
        self.priorities = collections.deque(maxlen=self.size)
        self.inp_shape = input_shape
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.size,*self.inp_shape), dtype=np.float32)
        self.next_memory =np.zeros((self.size,*self.inp_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.size,dtype=np.int64)
        self.reward_memory = np.zeros(self.size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.size, dtype=np.bool)

    def store_transition(self, state, action, reward, next_state, done):
        index = self.mem_cntr % self.size 
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.next_memory[index] = next_state
        self.final_memory[index] = done
        self.priorities.append(max(self.priorities, default=1))
        self.mem_cntr += 1
    
    def scaled_prob(self):
        P = np.array(self.priorities)
        P /= P.sum() #Check formula for more information
        return P
    
    def prob_imp(self, prob, beta):
        self.beta = beta
        i =  (1/self.size * 1/prob)**(-self.beta)
        i /= max(i)
        return i

    def sample(self, batch_size, beta):
        max_mem = min(self.mem_cntr, self.size)
        probability = self.scaled_prob()
        info = np.random.choice(max_mem, batch_size, replace=False, p = prob)
        states = self.state_memory[info]
        actions = self.action_memory[info]
        rewards = self.reward_memory[info]
        observations_ = self.next_memory[info]
        complete = self.terminal_memory[info]
        imp = self.prob_imp(sample_probs[info], beta)

        return states, actions, rewards, observations_, complete, imp, info
    
    def prop_priority(self, i, err, c = 1.1, alpha_value = 0.7): #for proportional priority
            self.priorities[i] = (np.abs(err) + c)** alpha_value


