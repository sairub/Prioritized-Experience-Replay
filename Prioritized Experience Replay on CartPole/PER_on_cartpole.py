import tensorflow as T
import gym
import tensorflow.keras.optimizers as ko
import numpy as np
import tensorflow.keras.layers as kl


class Model(T.keras.Model):
    def __init__(self, num_actions):
        super().__init__(name='basic_prddqn')
        self.fc1 = kl.Dense(32, activation='relu', kernel_initializer='he_uniform')
        self.fc2 = kl.Dense(32, activation='relu', kernel_initializer='he_uniform')
        self.final = kl.Dense(num_actions, name='q_values')

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.final(x)
        return x

    def action_value(self, observations):
        q_values = self.predict(observations)
        choose_action = np.argmax(q_values, axis=-1)
        return choose_action if choose_action.shape[0] > 1 else choose_action[0], q_values[0]


def test_dqn():
    env = gym.make('CartPole-v0')
    dqn = Model(env.action_space.n)

    observations = env.reset()

    choose_action, q_values = dqn.action_value(observations[None])


class ReplayMemory:
    def __init__(self, count_total):
        self.count_total = count_total
        self.mem_tree = np.zeros(2 * count_total - 1)
        self.transitions = np.empty(count_total, dtype=object)
        self.next_indices = 0

    @property
    def total_p(self):
        return self.mem_tree[0]

    def add(self, priority, transition):
        i = self.next_indices + self.count_total - 1
        self.transitions[self.next_indices] = transition
        self.change_it(i, priority)
        self.next_indices = (self.next_indices + 1) % self.count_total

    def change_it(self, i, priority):
        change = priority - self.mem_tree[i]
        self.mem_tree[i] = priority
        self._propagate(i, change)

    def _propagate(self, i, change):
        parent = (i - 1) // 2
        self.mem_tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def getdata(self, s):
        i = self.find(0, s)
        trans_i = i - self.count_total + 1
        return i, self.mem_tree[i], self.transitions[trans_i]

    def find(self, i, s):
        left = 2 * i + 1
        right = left + 1
        if left >= len(self.mem_tree):
            return i
        if s <= self.mem_tree[left]:
            return self.find(left, s)
        else:
            return self.find(right, s - self.mem_tree[left])


class Agent:
    def __init__(self, dqn, target_dqn, env, learning_rate=.0012, e=.1, e_dacay=0.995, min_e=.01,
                 gamma=.9, size=8, target_update_iter=400, count_trainings=5000, mem_size=200, replay_time=20,
                 alpha=0.4, beta=0.4, per_beta=0.001):
        self.dqn = dqn
        self.target_dqn = target_dqn

        opt = ko.Adam(learning_rate=learning_rate) #Gradient
        self.dqn.compile(optimizer=opt, loss=self.loss)

        # Hyper parameters
        self.env = env
        self.lr = learning_rate
        self.e = e
        self.e_decay = e_dacay
        self.min_e = min_e
        self.gamma = gamma
        self.size = size
        self.target_update_iter = target_update_iter
        self.count_trainings = count_trainings

        self.b_observations = np.empty((self.size,) + self.env.reset().shape)
        self.act_b = np.empty(self.size, dtype=np.int8)
        self.rewards_b = np.empty(self.size, dtype=np.float32)
        self.next_states_b = np.empty((self.size,) + self.env.reset().shape)
        self.completed = np.empty(self.size, dtype=np.bool)

        self.replay_memory = ReplayMemory(mem_size)
        self.mem_size = mem_size
        self.replay_time = replay_time
        self.alpha = alpha
        self.beta = beta
        self.per_beta = per_beta
        self.number_ = 0
        self.margin = 0.01
        self.prio = 1
        self.wt = np.power(self.mem_size, -self.beta)
        self.abs_error_upper = 1

    def loss(self, y_target, y_pred):
        return T.reduce_mean(self.wt * T.math.squared_difference(y_target, y_pred))

    def train(self):
        observations = self.env.reset()
        for t in range(1, self.count_trainings):
            choose_action, q_values = self.dqn.action_value(observations[None])
            action = self.act(choose_action)
            next_observations, reward, done, info = self.env.step(action)
            if t == 1:
                p = self.prio
            else:
                p = np.max(self.replay_memory.mem_tree[-self.replay_memory.count_total:])
            self.store_transition(p, observations, action, reward, next_observations, done)
            self.number_ = min(self.number_ + 1, self.mem_size)

            if t > self.mem_size:
                l = self.training()
                if t % 1000 == 0:
                    print('Loss at every 1000 steps = ', l)

            if t % self.target_update_iter == 0:
                self.change_target_dqn()
            if done:
                observations = self.env.reset()
            else:
                observations = next_observations

    def training(self):
        index, self.wt = self.Sampling_Priority(self.size)
        choose_action_index, _ = self.dqn.action_value(self.next_states_b)
        target_predicition = self.get_target_value(self.next_states_b)
        err_target = self.rewards_b + \
            self.gamma * target_predicition[np.arange(target_predicition.shape[0]), choose_action_index] * (1 - self.completed)
        make_predicition = self.dqn.predict(self.b_observations)
        err = make_predicition[np.arange(make_predicition.shape[0]), self.act_b]
        temporal_err = np.abs(err_target - err) + self.margin
        c = np.where(temporal_err < self.abs_error_upper, temporal_err, self.abs_error_upper)
        ps = np.power(c, self.alpha)
        for i, p in zip(index, ps):
            self.replay_memory.change_it(i, p)

        for i, val in enumerate(self.act_b):
            make_predicition[i][val] = err_target[i]

        target_predicition = make_predicition
        l = self.dqn.train_on_batch(self.b_observations, target_predicition)

        return l

    def Sampling_Priority(self, k):
        index_list = []
        wts = np.empty((k, 1))
        self.beta = min(1., self.beta + self.per_beta)
        min_prob = np.min(self.replay_memory.mem_tree[-self.replay_memory.count_total:]) / self.replay_memory.total_p
        max_weight = np.power(self.mem_size * min_prob, -self.beta)
        segment = self.replay_memory.total_p / k
        for i in range(k):
            s = np.random.uniform(segment * i, segment * (i + 1))
            i, p, t = self.replay_memory.getdata(s)
            index_list.append(i)
            self.b_observations[i], self.act_b[i], self.rewards_b[i], self.next_states_b[i], self.completed[i] = t
            sampling_probabilities = p / self.replay_memory.total_
            wts[i, 0] = np.power(self.mem_size * sampling_probabilities, -self.beta) / max_weight
        return index_list, wts

    def calculate(self, env, render=True):
        observations, done, reward_per_ep = env.reset(), False, 0
        while not done:
            action, q_values = self.dqn.action_value(observations[None])
            observations, reward, done, info = env.step(action)
            reward_per_ep += reward
            if render:
                env.render()
        env.close()
        return reward_per_ep

    def store_transition(self, priority, observations, action, reward, next_state, done):
        transition = [observations, action, reward, next_state, done]
        self.replay_memory.add(priority, transition)

    def rand_based_sample(self, k):
        pass

    def act(self, choose_action):
        if np.random.rand() < self.e:
            return self.env.action_space.sample()
        return choose_action

    def change_target_dqn(self):
        self.target_dqn.set_weights(self.dqn.get_weights())

    def get_target_value(self, observations):
        return self.target_dqn.predict(observations)

    def e_decay(self):
        self.e *= self.e_decay


test_dqn()

env = gym.make("CartPole-v0")
num_actions = env.action_space.n
dqn = Model(num_actions)
target_dqn = Model(num_actions)
agent = Agent(dqn, target_dqn, env)
rewards_sum = agent.calculate(env)
print("Prior Training Reward_Sum: %d out of 200" % rewards_sum)

agent.train()
rewards_sum = agent.calculate(env)
print("Training Result Reward_Sum: %d out of 200" % rewards_sum)
