changesimport tensorflow as T
import gym
import time
import numpy as np
import tensorflow.keras.layers as kl
import tensorflow.keras.optimizers as ko


class DQN(T.keras.Model):
    def __init__(self, num_actions):
        super().__init__(name='basic_prddqn')
        self.fc1 = kl.Dense(32, activation='relu', kernel_initializer='he_uniform')
        self.fc2 = kl.Dense(32, activation='relu', kernel_initializer='he_uniform')
        self.final = kl.Dense(num_actions, name='q_values')

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.final(x)
        return x

    def q_action(self, observation):
        q_values = self.predict(observation)
        choose_action = np.argmax(q_values, axis=-1)
        if choose_action.shape[0] > 1:
            return choose_action
        else:
            return choose_action[0], q_values[0]


def test_dqn():
    env = gym.make('CartPole-v0')
    dqn = DQN(env.action_space.n)

    observation = env.reset()

    choose_action, q_values = dqn.q_action(observation[None])
    print('Test DQN: ', choose_action, q_values)


class Prioritization:
    def __init__(self, total):
        self.total = total
        self.memory_tree = np.zeros(2 * total - 1)
        self.changes = np.empty(total, dtype=object)
        self.next_index = 0

    @property
    def total_p(self):
        return self.memory_tree[0]

    def appenddd(self, priority, transition):
        index = self.next_index + self.total - 1
        self.changes[self.next_index] = transition
        self.change(index, priority)
        self.next_index = (self.next_index + 1) % self.total

    def change(self, index, priority):
        change = priority - self.memory_tree[index]
        self.memory_tree[index] = priority
        self.yo(index, change)

    def yo(self, index, change):
        branch_parent = (index - 1) // 2
        self.memory_tree[branch_parent] += change
        if branch_parent != 0:
            self.yo(branch_parent, change)

    def get_data(self, s):
        index = self.find(0, s)
        trans_index = index - self.total + 1
        return index, self.memory_tree[index], self.changes[trans_index]

    def find(self, index, s):
        left = 2 * index + 1
        right = left + 1
        if left >= len(self.memory_tree):
            return index
        if s <= self.memory_tree[left]:
            return self.find(left, s)
        else:
            return self.find(right, s - self.memory_tree[left])


class Agent:
    def __init__(self, dqn, target_dqn, env, lr=.0012, e=.1, e_dacay=0.995, e_min=.01,
                 gamma=.9, size=8, target_update_iter=400, train_nums=5000, total_size=200, replay_time=20,
                 alpha=0.4, beta=0.4, per_sample_beta=0.001):
        self.dqn = dqn
        self.target_dqn = target_dqn
        opt = ko.Adam(learning_rate=lr)
        self.dqn.compile(optimizer=opt, loss=self.loss)


        self.env = env
        self.lr = lr
        self.e = e
        self.e_decay = e_dacay
        self.e_min = e_min
        self.gamma = gamma
        self.size = size
        self.target_update_iter = target_update_iter
        self.train_nums = train_nums

        self.obs_b = np.empty((self.size,) + self.env.reset().shape)
        self.act_b = np.empty(self.size, dtype=np.int8)
        self.rewards_b = np.empty(self.size, dtype=np.float32)
        self.next_states_b = np.empty((self.size,) + self.env.reset().shape)
        self.b_dones = np.empty(self.size, dtype=np.bool)

        self.replay_memory = Prioritization(total_size)
        self.total_size = total_size
        self.replay_time = replay_time
        self.alpha = alpha
        self.beta = beta
        self.per_sample_beta = per_sample_beta
        self.total_number = 0
        self.margin = 0.01
        self.p1 = 1
        self.wts = np.power(self.total_size, -self.beta)
        self.abs_error_upper = 1

    def loss(self, y_target, y_pred):
        return T.reduce_mean(self.wts * T.math.squared_difference(y_target, y_pred))

    def train(self):
        observation = self.env.reset()
        for t in range(1, self.train_nums):
            choose_action, q_values = self.dqn.q_action(observation[None])
            action = self.action_model(choose_action)
            next_observation, reward, done, info = self.env.step(action)
            if t == 1:
                p = self.p1
            else:
                p = np.max(self.replay_memory.memory_tree[-self.replay_memory.total:])
            self.store_transition(p, observation, action, reward, next_observation, done)
            self.total_number = min(self.total_number + 1, self.total_size)

            if t > self.total_size:
                losses = self.trainings()
                if t % 1000 == 0:
                    print('losses each 1000 steps: ', losses)

            if t % self.target_update_iter == 0:
                self.update_target_dqn()
            if done:
                observation = self.env.reset()
            else:
                observation = next_observation

    def trainings(self):
        index, self.wts = self.Sampling_Priority(self.size)
        choose_action_index, _ = self.dqn.q_action(self.next_states_b)
        target_q = self.get_data_part_2(self.next_states_b)
        td_target = self.rewards_b + self.gamma * target_q[np.arange(target_q.shape[0]), choose_action_index] * (1 - self.b_dones)
        predict_q = self.dqn.predict(self.obs_b)
        td_predict = predict_q[np.arange(predict_q.shape[0]), self.act_b]
        abs_td_error = np.abs(td_target - td_predict) + self.margin
        clipped_error = np.where(abs_td_error < self.abs_error_upper, abs_td_error, self.abs_error_upper)
        ps = clipped_error ** self.alpha
        for index, p in zip(index, ps):
            self.replay_memory.change(index, p)

        for i, val in enumerate(self.act_b):
            predict_q[i][val] = td_target[i]

        target_q = predict_q
        losses = self.dqn.train_on_batch(self.obs_b, target_q)

        return losses

    def Sampling_Priority(self, k):
        id_list = []
        wtss = np.empty((k, 1))
        self.beta = min(1., self.beta + self.per_sample_beta)
        min_prob = np.min(self.replay_memory.memory_tree[-self.replay_memory.total:]) / self.replay_memory.total_p
        max_weight = np.power(self.total_size * min_prob, -self.beta)
        segment = self.replay_memory.total_p / k
        for i in range(k):
            s = np.random.uniform(segment * i, segment * (i + 1))
            index, p, t = self.replay_memory.get_data(s)
            id_list.append(index)
            self.obs_b[i], self.act_b[i], self.rewards_b[i], self.next_states_b[i], self.b_dones[i] = t
            # P(j)
            sampling_probabilities = p / self.replay_memory.total_p     # where p = p ** self.alpha
            wtss[i, 0] = np.power(self.total_size * sampling_probabilities, -self.beta) / max_weight
        return id_list, wtss

    def calculate(self, env, render=True):
        observation, done, ep_reward = env.reset(), False, 0
        while not done:
            action, q_values = self.dqn.q_action(observation[None])  #[None] extendends the dimension from (4,) to (1,4)
            observation, reward, done, info = env.step(action)
            ep_reward += reward
            if render:
                env.render()
            time.sleep(0.05)
        env.close()
        return ep_reward

    def store_transition(self, priority, observation, action, reward, next_state, done):
        transition = [observation, action, reward, next_state, done]
        self.replay_memory.appenddd(priority, transition)

    def rand_based_sample(self, k):
        pass

    def action_model(self, choose_action):
        if np.random.rand() < self.e:
            return self.env.action_space.sample()
        return choose_action

    def update_target_dqn(self):
        self.target_dqn.set_weights(self.dqn.get_data())

    def get_data_part_2(self, observation):
        return self.target_dqn.predict(observation)

    def e_decay(self):
        self.e *= self.e_decay


if __name__ == '__main__':
    test_dqn()

    env = gym.make("CartPole-v0")
    num_actions = env.action_space.n
    dqn = DQN(num_actions)
    target_dqn = DQN(num_actions)
    agent = Agent(dqn, target_dqn, env)
    rewards_sum = agent.calculate(env)
    print("Before Training: %d out of 200" % rewards_sum)

    agent.train()
    rewards_sum = agent.calculate(env)
    print("After Training: %d out of 200" % rewards_sum)


# In[ ]:
