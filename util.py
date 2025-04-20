import random
import time
import tkinter as tk
import pandas as pd
import numpy as np
import ast
import heapq
import matplotlib.pyplot as plt


class MatplotBoard:
    def __init__(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.ax.set_xlabel("episode")
        self.ax.set_ylabel("reward")
        self.ax.set_title("Q-Learning")
        self.ax.grid(True)
        self.x = []
        self.y = []
        (self.line,) = self.ax.plot(self.x, self.y, color="blue")

    def update(self, x, y):
        self.x.append(x)
        self.y.append(y)
        self.line.set_data(self.x, self.y)
        self.ax.set_xlim(0, x + 10)
        self.ax.set_ylim(0, y + 10)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class GUI(tk.Tk):
    def __init__(self, treasure):
        super().__init__()

        self.ACTIONS = ["U", "D", "L", "R"]
        self.ACTION = len(self.ACTIONS)

        self.title("maze")
        self.maze = np.loadtxt("maze.txt", dtype=int)

        self.MAZE_R = len(self.maze)
        self.MAZE_C = len(self.maze[0])
        self.UNIT = 20
        h = self.MAZE_R * self.UNIT
        w = self.MAZE_C * self.UNIT
        self.geometry("{0}x{1}".format(h, w))
        self.canvas = tk.Canvas(self, bg="white", height=h, width=w)
        self.step = 0
        for i in range(self.MAZE_R):
            for j in range(self.MAZE_C):
                if self.maze[i][j] == 1:
                    self._draw_rect(i, j, "black")
        self.draw_treasure(treasure)

        self.canvas.pack()
        self.update()

    def draw_treasure(self, treasure):
        for i in range(len(treasure)):
            x, y = treasure[i]
            self._draw_rect(x, y, "red")
            self._draw_text(x, y, str(i + 1))

    def _draw_text(self, y, x, text):
        center_x = self.UNIT * (x + 0.5)
        center_y = self.UNIT * (y + 0.5)
        return self.canvas.create_text(center_x, center_y, text=text)

    def _draw_rect(self, y, x, color):
        coor = [
            self.UNIT * x,
            self.UNIT * y,
            self.UNIT * (x + 1),
            self.UNIT * (y + 1),
        ]
        return self.canvas.create_rectangle(*coor, fill=color)

    def _draw_oval(self, y, x, color):
        padding = 6
        coor = [
            self.UNIT * x + padding,
            self.UNIT * y + padding,
            self.UNIT * (x + 1) - padding,
            self.UNIT * (y + 1) - padding,
        ]
        return self.canvas.create_oval(*coor, fill=color)

    def move_agent_to(self, action):
        s = self.canvas.coords(self.oval)

        next_action = np.array([0, 0])
        if action == 0:  # up
            if s[1] > self.UNIT:
                next_action[1] -= self.UNIT
        elif action == 1:  # down
            if s[1] < (self.MAZE_R - 1) * self.UNIT:
                next_action[1] += self.UNIT
        elif action == 2:  # right
            if s[0] < (self.MAZE_C - 1) * self.UNIT:
                next_action[0] += self.UNIT
        elif action == 3:  # left
            if s[0] > self.UNIT:
                next_action[0] -= self.UNIT

        self.canvas.move(self.oval, next_action[0], next_action[1])
        s_ = self.canvas.coords(self.oval)

        done = False
        self.step += 1
        if s_ == self.canvas.coords(self.oval):
            done = True
        return s_, done

    def render(self):
        time.sleep(0.1)
        self.canvas.update()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.oval)
        self.oval = self._draw_oval(0, self.MAZE_C - 1, "yellow")

        return self.canvas.coords(self.oval)


def bool_permutations(lst):
    result = []
    permute(lst, [], result)
    return result


def permute(lst, current, result):
    if len(current) == len(lst):
        result.append(current)
        return
    for i in [True, False]:
        permute(lst, current + [i], result)


def load_float_list_from_file(filename):
    with open(filename, "r") as file:
        content = file.read()
        my_list = ast.literal_eval(content)


def MatrixTransform(item):
    for i in range(len(item)):
        item[i] = list(item[i])
        item[i][0] = 18 - (2 * item[i][0])
        item[i][1] = 2 * item[i][1]
    return item


def ListTransform(item):
    item[0] = 18 - (2 * item[0])
    item[1] = 2 * item[1]
    return item


def NegativeR(item):
    item = np.array(item)
    item = -item
    return item.tolist()


def ListFake(states):
    fake = []
    for state in states:
        for i in range(len(state)):
            if i % 2 == 0 and state[i] == False and state[i + 1] == False:
                if state[7 - i] == False:
                    fake.append(i + 1)
                else:
                    fake.append(2 + i)
    # print(fake)
    return fake


def MatrixPosition(item):
    for i in range(len(item)):
        item[i][0] = int(9 - item[i][0] / 2)
        item[i][1] = int(item[i][1] / 2)
    return item


def sets(targetList):
    t_sets = set()
    for i in range(1000):
        t_gets = gets(targetList)
        t_gets.sort()
        t_sets.add(tuple(t_gets))
    return t_sets


def gets(targetList):
    countUp = 0
    countDown = 0
    t_gets = []
    countCorner = int(len(targetList) / 2)
    t_gets_visited = [False for _ in range(countCorner)]
    while countUp < 2 or countDown < 2:
        gets = random.randint(0, countCorner - 1)
        if t_gets_visited[gets] == True:
            continue
        x, y = targetList[gets]
        flag = False
        if countUp < 2 and y < 5:
            countUp += 1
            flag = True
        elif countDown < 2 and y >= 5:
            flag = True
            countDown += 1
        if flag == False:
            continue
        t_gets_visited[gets] = True
        t_gets.append((x, y))
        x, y = targetList[gets + countCorner]
        t_gets.append((x, y))
    return t_gets


def MatrixR(distances):
    length = 8
    r = np.zeros((length + 2, length + 2))

    for i in range(length + 2):
        if i == 0:
            for j in range(length + 2):
                if j == 0:
                    r[i][j] = 0
                else:
                    r[i][j] = distances[length][j - 1]
        elif i == length + 1:
            for j in range(length + 2):
                if j == length + 1:
                    r[i][j] = 0
                elif j == 0:
                    r[i][j] = distances[length][length]
                else:
                    r[i][j] = distances[j - 1][length]
        else:
            for j in range(length + 2):
                if j == 0:
                    r[i][j] = distances[i - 1][i - 1]
                elif j == i:
                    r[i][j] = 0
                else:
                    r[i][j] = distances[i - 1][j - 1]
    r = NegativeR(r)
    return r
