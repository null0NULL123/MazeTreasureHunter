import random
import time
import tkinter as tk
import pandas as pd
import numpy as np
import copy
import time
from package import states_list
from package import r_list
from package import fake_list
from package import greed_rewards
from package import color_list
import matplotlib.pyplot as plt
import itertools
class Environment:
    pass
class RL:
    def __init__(
        self,
        r,
        learning_rate=0.1,
        reward_decay=0.9,
        e_greedy=0,
        train_time=10000,
        batch_size=1000,
        pic_error=0.1,
        action_error=0.05,
        plot=True,
        sarsa=True
    ):
        self.attribute = random.randint(0, 15)
        self.state = states_list[self.attribute]
        self.action = 0
        self.scores = 0
        self.fake = fake_list[self.attribute]
        self.actions = []
        self.sets = set()
        self.alpha = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.train_time = train_time
        self.batch_size = batch_size
        self.pic_error = pic_error
        self.action_error = action_error
        self.steps = []
        self.destination = 9
        # self.q = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.q_table = np.matrix(np.zeros((9,self.destination)))
        self.isTest = False
        self.isPLT = plot
        self.plot_list = []
        self.test_time = 500
        self.isSarsa = sarsa
        self.time = 0
        self.isGreed = False
        self.r_table = np.matrix(r)
        self.info = 0
        self.findFake=False
        
    def reset(self):
        self.plot_list = []
        self.q_table = np.matrix(np.zeros((9,self.destination)))

    def q_best(self, next_action):
        return (
            np.max(self.q_table[next_action, self.actions])
            if len(self.actions) > 0
            else self.r_table[next_action,self.destination]
        )

    def update(self, action):
        self.sets.discard(action)
        # if self.pic_error > np.random.random():
        #     return
        if self.state[action - 1]:
            self.scores += 1

        if action ==self.destination - self.fake:
            self.sets.discard(9 - action)
            self.findFake=True
        if not self.state[action - 1] and action != self.fake: 
            self.sets.discard(8 - action) if action % 2 == 1 else self.sets.discard(
                10 - action
            )
        else:
            self.sets.discard(action+1) if action % 2 == 1 else self.sets.discard(
                action-1
            )
            self.sets.discard(9 - action)
        self.actions = list(self.sets)
        if self.info==0:
            self.info = 1 if abs(action-4.5)>2 else 2
        elif (self.info==1 and abs(action-4.5)<2)  or (self.info==2 and abs(action-4.5)>2):
            self.info = 3

    def take_action(self):
        # if np.random.random() < self.action_error:
        #     pass
        if self.isGreed:
            action = self.actions[np.argmax(self.r_table[self.action, self.actions])]        
        elif self.isTest or np.random.random() > self.epsilon:
            action = self.actions[np.argmax(self.q_table[self.action, self.actions])]
        else:
            action = np.random.choice(self.actions)
        self.actions.remove(action)
        return action

    def train(self):
        # self.time = time.perf_counter()
        for i in range(self.train_time):
            self.initial()
            self.isTest = False
            self.isGreed = False
            while not self.done():
                next_action = self.take_action()

                self.q_table[self.action, next_action] = (1 - self.alpha) * self.q_table[
                    self.action, next_action
                ] + self.alpha * (
                    self.r_table[self.action, next_action]
                    + self.gamma * q.q_best(next_action)
                )
                self.update(next_action)

                self.action = next_action

            if self.isPLT and i % self.batch_size == self.batch_size - 1:
                test = self.test()
                self.plot_list.append(test)
                # print(self.q_table)
        # self.time = time.perf_counter() - self.time
        # print("time:",time.perf_counter()-self.time)
        # print(self.test())
    def initial(self):
        self.attribute = random.randint(0, 15)
        self.state = states_list[self.attribute]
        self.action = 0
        self.scores = 0
        self.fake = fake_list[self.attribute]
        self.actions = [1, 2, 3, 4, 5, 6, 7, 8]
        self.sets = {1, 2, 3, 4, 5, 6, 7, 8}
        self.steps = []
        self.epsilon = 0.9
        self.info = 0
        self.findFake=False

    def test(self):
        rewards_array = []

        self.isTest = True

        for _ in range(self.test_time):
            reward = 0
            self.initial()
            # print(self.state)

            while not self.done():
                next_action = self.take_action()
                reward += self.r_table[self.action, next_action]
                self.update(next_action)
                self.action = next_action
            if self.scores == 3 or len(self.actions) == 0:
                reward += self.r_table[self.action,self.destination]
            if self.scores < 3 and len(self.actions) > 0:

                permutations = list(itertools.permutations(self.actions))
                reward_list = []
                for permutation in permutations:
                    
                    permutation = list(permutation)
                    permutation_action = copy.deepcopy(self.action)
                    permutation_reward = 0
                    while len(permutation) > 0:
                        permutation_reward += self.r_table[permutation_action, permutation[0]]
                        permutation_action=permutation.pop(0)
                    permutation_reward += self.r_table[permutation_action,self.destination]
                    reward_list.append(permutation_reward)
                reward += max(reward_list)
                
            rewards_array.append(reward)
        # print(np.std(rewards_array))
        return np.mean(rewards_array)

    def done(self):
        return self.scores == 3 or len(self.actions) == 0 or (self.info == 3 and self.findFake)
    def greed(self):
        greed_time = 500
        rewards_array = []
        self.isGreed = True
        self.isTest = False
        for _ in range(greed_time):
            reward = 0
            self.initial()
            while not self.done():
                next_action = self.take_action()

                reward += self.r_table[self.action, next_action]
                self.update(next_action)
                

                self.action = next_action
            if self.scores == 3 or len(self.actions) == 0:
                reward += self.r_table[self.action,self.destination]
            else:
                # print(self.actions)
                # print(reward)
                permutations = list(itertools.permutations(self.actions))
                reward_list = []
                for permutation in permutations:
                    
                    permutation = list(permutation)
                    permutation_action = copy.deepcopy(self.action)
                    permutation_reward = 0
                    while len(permutation) > 0:
                        permutation_reward += self.r_table[permutation_action, permutation[0]]
                        permutation_action=permutation.pop(0)
                    permutation_reward += self.r_table[permutation_action,self.destination]
                    reward_list.append(permutation_reward)
                print(reward_list)
                reward += max(reward_list)
                # print(reward)
            rewards_array.append(reward)
        mean = np.mean(rewards_array)
        std = np.std(rewards_array)
        return mean, std
if __name__ == "__main__":

    learning_rate = 0.01
    train_time = 200
    batch_size = 10
    for pic in range(1, 2):
        train_time *= 10
        batch_size *= 10

        plot_size = int(train_time / batch_size)
        plot_x = np.arange(0, plot_size) * batch_size
        plt.subplot(2, 1, pic)
        plt.title(f"lr: {learning_rate}, train_time: {train_time}")
        plt.subplot(2, 1, pic + 1)
        plt.title(f"lr: {learning_rate}, train_time: {train_time}")
        for i in range(3):
            q = RL(
                r_list[i],
                learning_rate=learning_rate,
                train_time=train_time,
                batch_size=batch_size,
            )
            epoch = 10
            test_reward_matrix = []
            test_reward = np.zeros(plot_size)
            for _ in range(epoch):
                q.reset()
                q.train()
                test_reward_matrix.append(q.plot_list)
                test_reward += np.array(q.plot_list)
            print(q.q_table)
            test_reward /= epoch
            test_reward_matrix = np.array(test_reward_matrix)
            test_reward_std = np.std(test_reward_matrix, axis=0)
            plt.subplot(2, 1, pic)
            plt.xlabel("Train Time")
            plt.ylabel("Maximum Reward")
            plt.plot(plot_x, test_reward, color=color_list[i])
            plt.axhline(y=greed_rewards[i], color=color_list[i], linestyle="--")
            plt.subplot(2, 1, pic + 1)
            plt.xlabel("Train Time")
            plt.ylabel("Standard Deviation")
            plt.plot(plot_x, test_reward_std, color=color_list[i])
    plt.tight_layout()
    plt.show()

