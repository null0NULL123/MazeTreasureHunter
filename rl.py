import random
import time
import tqdm
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
    def __init__(self, r, capture_error=0.1, action_error=0.05):
        self.r_table = np.matrix(r)
        self.INIT = 0
        self.FINAL = 9
        self.HALF = int(self.FINAL / 2)
        self.TY = 1
        self.NT = 2
        self.FY = 3
        self.FN = 4
        self.NONE = 0
        self.n_state = 5
        self.info = 0
        self.info_fake = False
        self.scores = 0
        self.capture_error = capture_error
        self.action_error = action_error
        self.action = 0
        self.attribute = random.randint(0, 15)
        self.state = states_list[self.attribute]
        self.fake = fake_list[self.attribute]
        self.valid_action = {1, 2, 3, 4, 5, 6, 7, 8}

        def ty():
            self.scores += 1
            self.valid_action.discard(self.FINAL - self.action)
            self.valid_action.discard(
                self.action + 1
            ) if self.action % 2 == 1 else self.valid_action.discard(self.action - 1)

        def nt():
            self.valid_action.discard(
                self.FINAL - 1 - self.action
            ) if self.action % 2 == 1 else self.valid_action.discard(
                self.FINAL + 1 - self.action
            )

        def fy():
            self.valid_action.discard(
                self.action + 1
            ) if self.action % 2 == 1 else self.valid_action.discard(self.action - 1)
            self.valid_action.discard(self.FINAL - self.action)
            self.info_fake = True

        def fn():
            self.valid_action.discard(self.FINAL - self.action)

            self.info_fake = True

        self.capture_func = {
            self.TY: ty,
            self.NT: nt,
            self.FY: fy,
            self.FN: fn,
        }

    def done(self):
        return (
            self.scores == 3
            or len(self.valid_action) == 0
            # or (self.info == 3 and self.info_fake)
        )

    def get_info(self, action):
        if self.info == 0:
            self.info = 1 if abs(2 * action - self.FINAL) > self.HALF else 2
        elif (self.info == 1 and abs(2 * action - self.FINAL) < self.HALF) or (
            self.info == 2 and abs(2 * action - self.FINAL) > self.HALF
        ):
            self.info = 3

    def step(self, action):

        self.capture_func[self.state[action]]()

        self.info = self.get_info(action)
        self.valid_action.discard(action)
        reward = self.r_table[self.action, action]
        done = self.done()
        self.action = action
        state = action * self.n_state - self.state[action]

        return (
            state,
            reward,
            done,
            list(self.valid_action),
        )
        # state, reward, done, valid_actions

    def reset(self):
        self.scores = 0
        self.info = 0
        self.info_fake = False
        self.attribute = random.randint(0, 15)
        self.state = states_list[self.attribute]
        self.fake = fake_list[self.attribute]
        self.valid_action = {1, 2, 3, 4, 5, 6, 7, 8}
        return self.INIT,self.INIT, list(self.valid_action)


class Sarsa:
    pass


class QLearning:
    def __init__(
        self,
        col=5,
        row=8,
        learning_rate=0.1,
        reward_decay=0.9,
        e_greedy=0,
        n_action=10,
    ):
        self.alpha = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = np.zeros((col * row + 1, n_action))
        self.actions = [1, 2, 3, 4, 5, 6, 7, 8]
        self.action = 0
        self.reward = 0
        self.state = 0
        self.final = 9

    def update(self):
        td_error = (
            self.reward
            + self.gamma
            * (
                np.max(self.q_table[self.state, self.actions])
                if len(self.actions) > 0
                else self.q_table[self.state, self.final]
            )
            - self.q_table[self.state, self.action]
        )
        self.q_table[self.state, self.action] += self.alpha * td_error

    def take_action(self):
        if self.actions == []:
            raise
        if np.random.random() > self.epsilon:
            return self.actions[np.argmax(self.q_table[self.action, self.actions])]
        else:
            return np.random.choice(self.actions)


class RL:
    def __init__(
        self,
        train_time=20000,
        batch_size=1000,
        pic_error=0.1,
        action_error=0.05,
        plot=True,
    ):
        self.train_times = train_time
        self.batch_size = batch_size

        self.pic_error = pic_error
        self.action_error = action_error
        self.isPLT = plot
        self.test_times = 1000
        self.isTest = False
        self.main()

    def main(self):
        plot_size = int(self.train_times / self.batch_size)
        plot_x = np.arange(0, plot_size) * self.batch_size
        plt.figure()
        agent = QLearning()
        plt.title(f"lr: {agent.alpha}, train_time: {self.train_times}")

        for i in range(9):
            env = Environment(r_list[i])
            epoch = 10

            reward_matrix = np.zeros(plot_size)
            for _ in range(epoch):
                agent.__init__()
                for j in range(len(agent.q_table)):
                    agent.q_table[j, agent.final] = env.r_table[
                        int((j + 4) / 5), agent.final
                    ]
                # print(agent.q_table)
                reward_list = []
                for j in range(self.train_times):
                    agent.state,agent.action, agent.actions = env.reset()
                    done = False
                    while not done:
                        try:
                            agent.action = agent.take_action()
                            state, agent.reward, done, agent.actions = env.step(
                                agent.action
                            )
                            agent.update()
                            agent.state = state
                        except:
                            print(agent.action)
                            print(agent.actions)
                            print(agent.state)
                            raise
                    if self.isPLT and j % self.batch_size == self.batch_size - 1:
                        # print(agent.q_table)
                        test_reward = []
                        for _ in range(self.test_times):
                            agent.state,agent.action, agent.actions = env.reset()
                            done = False
                            reward = 0
                            while not done:
                                next_action = agent.take_action()
                                (
                                    agent.state,
                                    agent.reward,
                                    done,
                                    agent.actions,
                                ) = env.step(next_action)
                                reward += agent.reward
                            reward += env.r_table[agent.action, agent.final]
                            test_reward.append(reward)
                        reward_list.append(np.mean(test_reward))
                reward_matrix += np.array(reward_list) / epoch
            print(agent.q_table)
            plt.xlabel("Train Time")
            plt.ylabel("Average Reward")
            plt.plot(plot_x, reward_matrix, color=color_list[i])
            # plt.axhline(y=greed_rewards[i], color=color_list[i], linestyle="--")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    rl = RL(train_time=200000, batch_size=1000, plot=True)
