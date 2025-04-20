import matplotlib.pyplot as plt
import random
from package import *
from util import *

class Time:
    def dfs_ccl(item):
        global dfs_distances
        global dfs_index
        global dfs_visited
        global dfs_p
        dfs_visited = copy.deepcopy(item)
        dfs_distances = float("inf")
        dfs_p = []
        dfs_index = -1
        Time.dfs(0)

        return dfs_index

    def dijkstra(start, end):
        global startx, starty, maze_endx, maze_endy
        startx, starty = start
        maze_endx, maze_endy = end
        bool_end = [maze_endx, maze_endy] in t_wall
        bool_start = [startx, starty] in t_wall
        global maze
        # start=ListTransform(start)
        # start=tuple(start)
        visited = [[False] * n for _ in range(m)]
        global distances
        distances = [[float("inf")] * n for _ in range(m)]
        global predecessors
        predecessors = [[None] * n for _ in range(m)]  # 记录每个位置的前驱节点
        distances[start[0]][start[1]] = 0

        maze[startx][starty] = 0
        maze[maze_endx][maze_endy] = 0

        if bool_start == True:
            if pathx - startx == 0:
                if starty != 18:
                    maze[startx + 1][starty] = 1
                if starty != 0:
                    maze[startx - 1][starty] = 1
            elif pathy - starty == 0:
                if startx != 18:
                    maze[startx][starty + 1] = 1
                if startx != 18:
                    maze[startx][starty - 1] = 1
        heap = [(0, start)]
        while heap:
            dist, (x, y) = heapq.heappop(heap)
            if visited[x][y]:
                continue
            visited[x][y] = True

            if (x, y) == end:
                if bool_end == True:
                    maze[x][y] = 1
                if bool_start == True:
                    maze[startx][starty] = 1
                break
            # traverse all the nodes
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                next_x, next_y = x + dx, y + dy
                if (
                    next_x < 0
                    or next_x >= m
                    or next_y < 0
                    or next_y >= n
                    or visited[next_x][next_y]
                    or maze[next_x][next_y] == 1
                ):
                    continue
                if predecessors[x][y] != None:
                    px, py = predecessors[x][y]
                    if x - px == dx and y - py == dy:
                        element = (x, y) in cross
                        if element == True:
                            new_dist = dist + capture_parameter
                        new_dist = dist + direct_parameter
                    elif x - px == -dx and y - py == -dy:
                        new_dist = dist + turn_parameter
                    else:
                        new_dist = dist + cross_parameter
                else:
                    new_dist = dist + 1
                if new_dist < distances[next_x][next_y]:
                    distances[next_x][next_y] = new_dist
                    predecessors[next_x][next_y] = (x, y)
                    heapq.heappush(heap, (new_dist, (next_x, next_y)))

    def greed(pos):
        short_dis = float("inf")
        count = copy.deepcopy(pos)
        x0, y0 = Traverse.list_val(count)

        for i in range(t_len):
            if t_visited[i] == False:
                continue
            x, y = Traverse.list_val(i)
            Time.dijkstra((x0, y0), (x, y))
            dis = distances[x][y]
            if dis < short_dis:
                short_dis = dis
                pos = i

        return pos

    def dfs(d):
        global dfs_visited
        global dfs_p
        global dfs_index
        global dfs_distances
        global dfs_last

        element = True in dfs_visited
        if element == False:
            d += distances[0][18]
            if d < dfs_distances:
                dfs_distances = copy.deepcopy(d)
                dfs_list = copy.deepcopy(dfs_p)
                dfs_index = dfs_list[0]

            return

        dis = 0
        x0, y0 = Traverse.list_val[dfs_last]
        for i in range(len(dfs_visited)):
            if dfs_visited[i] == False:
                continue

            dfs_visited[i] = False
            dfs_p.append(i)
            x, y = Traverse.list_val[i]

            Time.dijkstra((x0, y0), (x, y))
            dis = distances[x][y]
            d += dis
            dfs_last = copy.deepcopy(i)
            Time.dfs(d)
            dfs_visited[i] = True
            dfs_p.pop()
            if len(dfs_p) == 0:
                dfs_last = t_last
            else:
                dfs_last = copy.deepcopy(dfs_p[-1])
            d -= dis


class Global:
    def dfs_ccl(item):
        global dfs_distances
        global dfs_index
        global dfs_visited
        global dfs_p
        dfs_visited = copy.deepcopy(item)
        dfs_distances = float("inf")
        dfs_p = []
        dfs_index = -1
        Global.dfs(0)

        return dfs_index

    def greedy(pos):
        short_dis = float("inf")

        for i in range(t_len):
            if t_visited[i] == False:
                continue

            if t_last == -1:
                dis = t_distances[i][i]
            else:
                dis = t_distances[pos][i]
            if dis < short_dis:
                short_dis = dis
                pos = i

        return pos

    def dijkstra(start):
        visited = [[False] * n for _ in range(m)]
        global distances
        distances = [[float("inf")] * n for _ in range(m)]
        global predecessors
        predecessors = [[None] * n for _ in range(m)]  # 记录每个位置的前驱节点
        distances[start[0]][start[1]] = 0

        # if element == True:

        #     maze[x][y] =1
        node = 0
        heap = [(0, start)]
        while heap:
            dist, (x, y) = heapq.heappop(heap)
            if visited[x][y]:
                continue
            visited[x][y] = True
            node += 1

            if node == maze_parameter:
                break
            # traverse all the nodes
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                next_x, next_y = x + dx, y + dy
                if (
                    next_x < 0
                    or next_x >= m
                    or next_y < 0
                    or next_y >= n
                    or visited[next_x][next_y]
                    or maze[next_x][next_y] == 1
                ):
                    continue
                if predecessors[x][y] != None:
                    px, py = predecessors[x][y]
                    if x - px == dx and y - py == dy:
                        element = (x, y) in cross
                        if element == True:
                            pass
                        new_dist = dist + direct_parameter
                    else:
                        new_dist = dist + 1
                else:
                    new_dist = dist + 1
                if new_dist < distances[next_x][next_y]:
                    distances[next_x][next_y] = new_dist
                    predecessors[next_x][next_y] = (x, y)
                    heapq.heappush(heap, (new_dist, (next_x, next_y)))

    def dfs(d):
        global dfs_visited
        global dfs_p
        global dfs_index
        global dfs_distances
        global dfs_last
        element = True in dfs_visited
        if element == False:
            d += t_distances[dfs_p[-1]][t_len]
            if d < dfs_distances:
                dfs_distances = copy.deepcopy(d)
                dfs_index = copy.deepcopy(dfs_p[0])

            return

        dis = 0
        for i in range(len(dfs_visited)):
            if dfs_visited[i] == False:
                continue

            dfs_visited[i] = False
            dfs_p.append(i)
            if dfs_last == -1:
                dis = t_distances[i][i]

            else:
                dis = t_distances[dfs_last][i]
            d += dis
            dfs_last = copy.deepcopy(i)
            Global.dfs(d)
            dfs_visited[i] = True
            dfs_p.pop()
            if len(dfs_p) == 0:
                dfs_last = t_last
            else:
                dfs_last = copy.deepcopy(dfs_p[-1])
            d -= dis


class Traverse:
    def wall():
        global t_wall
        global maze
        for i in range(t_len):
            x, y = Traverse.list_val(i)
            element = [x, y] in t_wall_maze

            if element == True:
                t_wall.append([x, y])
                maze[x][y] = 1

    def judge(visited):
        global t_self1, t_self2
        global t_fake, t_score, t_error
        global t_first1, t_first2
        global t_last, t_next
        global dfs_last
        global maze
        global t_wall
        global pic_list
        t_last = copy.deepcopy(t_next)
        dfs_last = t_last
        visited[t_last] = False
        pic_list[t_last] = picture

        if abs(picture) == 1:
            if picture == -1:
                pic_list[7 - t_last] = 1
                # x,y=Treasure.list_val(7-t_last)
            else:
                pic_list[7 - t_last] = -1

                t_score += 1
                # x,y=Treasure.list_val(t_last)

        elif abs(picture) == 2:
            t_fake = True
            visited[t_len - 1 - t_last] = False
            if picture == -2:
                pic_list[7 - t_last] = 2
            else:
                pic_list[7 - t_last] = -2

        if t_score == 3:
            for i in range(len(visited)):
                visited[i] = False
        elif picture > 0:
            # 我方宝藏
            visited[t_len - 1 - t_last] = False
            for i in range(t_len):
                if mat_list[i] == mat_list[t_last]:
                    visited[i] = False
            if mat_list[t_last] == 1 or mat_list[t_last] == 4:
                t_self1 = 1
                t_first1 = False
            else:
                t_self2 = 1
                t_first2 = False
        elif picture < 0:
            for i in range(t_len):
                if mat_list[i] + mat_list[t_last] == 5 and i != t_len - 1 - t_last:
                    visited[i] = False
            if mat_list[t_last] == 1 or mat_list[t_last] == 4:
                t_self1 = -1
                t_first1 = False
            else:
                t_self2 = -1
                t_first2 = False

        return visited

    def transform():
        global treasure
        l = copy.deepcopy(len(treasure))
        for i in range(l):
            x = treasure[i][0]
            y = treasure[i][1]
            treasure.append([9 - x, 9 - y])

    def list_val(count):
        global t_tranform
        if count == -1:
            return 18, 0
        else:
            return t_tranform[count][0], t_tranform[count][1]

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
            t_gets.append([x, y])
            x, y = targetList[gets + countCorner]
            t_gets.append([x, y])
        return t_gets

    def initial():
        global treasure
        global t_len
        global mat_list
        global t_tranform
        global t_direction
        global t_distances
        global t_visited
        global pic_list
        global mat_id
        global permutations
        global t_path
        global t_path_corner
        treasure.sort()
        t_len = len(treasure)

        for i in range(t_len):
            x = treasure[i][0]
            y = treasure[i][1]
            if y < 5:
                if x < 5:
                    mat_list.append(1)
                else:
                    mat_list.append(2)
            else:
                if x < 5:
                    mat_list.append(3)
                else:
                    mat_list.append(4)
        # print(mat_list)
        for index in range(2):
            if mat_list[index] == 3:
                while mat_list[mat_id] != 1:
                    mat_id += 1
                indop = t_len - 1 - index
                indro = t_len - 1 - mat_id
                # mat_list[mat_id],mat_list[index]=mat_list[index],mat_list[mat_id]
                # mat_list[indop],mat_list[indro]=mat_list[indro],mat_list[indop]

                treasure[mat_id], treasure[index] = treasure[index], treasure[mat_id]
                treasure[indop], treasure[indro] = treasure[indro], treasure[indop]
        # print(treasure)
        for index in range(2, 4):
            mat_id = t_len - 1 - index
            treasure[mat_id], treasure[index] = treasure[index], treasure[mat_id]
            # mat_list[mat_id],mat_list[index]=mat_list[index],mat_list[mat_id]
        # print(treasure)
        mat_list = [1, 1, 2, 2, 3, 3, 4, 4]

        t_tranform = copy.deepcopy(treasure)

        t_tranform = MatrixTransform(t_tranform)
        # print("treasure:", "\n", t_tranform)
        Traverse.wall()

        t_distances = [[float("inf")] * (t_len + 1) for _ in range(t_len + 1)]
        t_direction = [[] for _ in range(t_len + 1)]
        t_path_corner = [[] for _ in range(t_len + 1)]
        t_path = [[] for _ in range(t_len + 1)]

        t_visited = [True for _ in range(t_len)]
        pic_list = [0 for _ in range(t_len)]
        # permutations = bool_permutations(t_visited)

        # # 此处必改，考虑
        for i in range(t_len + 1):
            if i != t_len:
                x0 = t_tranform[i][0]
                y0 = t_tranform[i][1]

                for j in range(t_len + 1):
                    direction = []
                    if j == i:
                        x = 18
                        y = 0
                        # Time.dij_time((x0,y0),(18,0))
                        # t_distances[i][j]=distances[18][0]

                    elif j == t_len:
                        x = 0
                        y = 18
                        # Time.dij_time((x0,y0),(0,18))

                        # direction=DirectTransform((x0,y0),(x,y))
                        # t_distances[i][j]=distances[0][18]

                    else:
                        x = t_tranform[j][0]
                        y = t_tranform[j][1]
                    Time.dijkstra((x0, y0), (x, y))

                    direction = DirectTransform((x0, y0), (x, y))
                    t_distances[i][j] = distances[x][y]
                    t_direction[i].append(direction)
                    t_path[i].append(PathOriginal)
                    t_path_corner[i].append(path_temp)
            else:
                for j in range(t_len + 1):
                    if j == t_len:
                        x = 0
                        y = 18
                    else:
                        x = t_tranform[j][0]
                        y = t_tranform[j][1]

                    Time.dijkstra((18, 0), (x, y))
                    t_distances[i][j] = distances[x][y]
                    direction = DirectTransform((18, 0), (x, y))
                    t_direction[i].append(direction)
                    t_path[i].append(PathOriginal)
                    t_path_corner[i].append(path_temp)

        # print(t_direction)
        t_distances = np.round(t_distances, 2)

        # print(t_distances)
        t_distances = t_distances.tolist()


class Infer:
    def permutation():
        # calculate the negative posibility
        def ccl():
            global inf_visited
            global inf_distances
            global inf_index
            global dfs_distances
            global dfs_index
            global dfs_p
            global dfs_visited

            inf_index = [0 for _ in range(t_len)]
            inf_distances = [0 for _ in range(t_len)]
            permutation_answer = []
            for permutation in permutations:
                count = permutation.count(True) + t_score

                if count >= 3:
                    if t_first1 == False:
                        if t_fake == True:
                            if permutation[0] != t_visited[0]:
                                continue
                            elif permutation[1] != t_visited[1]:
                                continue
                            elif permutation[6] != t_visited[6]:
                                continue
                            elif permutation[7] != t_visited[7]:
                                continue
                        else:
                            if t_visited[0] == False and permutation[0] != t_visited[0]:
                                continue
                            elif (
                                t_visited[1] == False and permutation[1] != t_visited[1]
                            ):
                                continue
                            elif (
                                t_visited[6] == False and permutation[6] != t_visited[6]
                            ):
                                continue
                            elif (
                                t_visited[7] == False and permutation[7] != t_visited[7]
                            ):
                                continue
                    if t_first2 == False:
                        if t_fake == True:
                            if permutation[2] != t_visited[2]:
                                continue
                            elif permutation[3] != t_visited[3]:
                                continue
                            elif permutation[4] != t_visited[4]:
                                continue
                            elif permutation[5] != t_visited[5]:
                                continue
                        else:
                            if t_visited[2] == False and permutation[2] != t_visited[2]:
                                continue
                            elif (
                                t_visited[3] == False and permutation[3] != t_visited[3]
                            ):
                                continue
                            elif (
                                t_visited[4] == False and permutation[4] != t_visited[4]
                            ):
                                continue
                            elif (
                                t_visited[5] == False and permutation[5] != t_visited[5]
                            ):
                                continue
                    permutation_answer.append(permutation)
                    dfs_index = Global.dfs_ccl(permutation)
                    if dfs_index > 0:
                        inf_index[dfs_index] += 1
                        inf_distances[dfs_index] += dfs_distances
            # print(permutation_answer)
            p_count = -1

            pos = -1

            for i in range(t_len):
                if inf_index[i] > p_count:
                    p_count = inf_index[i]
                    pos = i

                elif inf_index[i] == p_count:
                    if inf_distances[i] < inf_distances[pos]:
                        pos = i
                    elif inf_distances[i] == inf_distances[pos]:
                        dis_pos = t_distances[t_len][pos]
                        dis_i = t_distances[t_len][i]
                        if t_last != -1:
                            dis_pos = t_distances[t_last][pos]
                            dis_i = t_distances[t_last][i]
                        if dis_i < dis_pos:
                            pos = i
            return pos

        global t_next
        t_next = ccl()
        return t_next

    def ccl():
        # calculate the positive posibility
        global inf_visited
        global inf_distances
        global inf_index
        global dfs_distances
        global dfs_index
        global dfs_p
        global dfs_visited
        inf_index = [0 for _ in range(t_len)]
        inf_distances = [0 for _ in range(t_len)]

        inf_visited = Infer.inf(t_visited)
        for item in inf_visited:
            dfs_index = Global.dfs_ccl(item)
            if dfs_index >= 0:
                inf_index[dfs_index] += 1
                inf_distances[dfs_index] += dfs_distances
        p_count = -1
        pos = -1

        for i in range(t_len):
            if inf_index[i] > p_count:
                p_count = inf_index[i]
                pos = i
            elif inf_index[i] == p_count:
                if inf_distances[i] < inf_distances[pos]:
                    pos = i
                elif inf_distances[i] == inf_distances[pos]:
                    dis_pos = t_distances[t_len][pos]
                    dis_i = t_distances[t_len][i]
                    if t_last != -1:
                        dis_pos = t_distances[t_last][pos]
                        dis_i = t_distances[t_last][i]
                    if dis_i < dis_pos:
                        pos = i
        return pos

    def inf(visited):
        def inf_fake(visited, fake):
            visited_temp = copy.deepcopy(visited)
            visited_temp[fake] = False
            visited_temp[7 - fake] = False
            return visited_temp

        def inf_false(visited, false):
            visited_temp = copy.deepcopy(visited)
            visited_temp[false] = False
            if false % 2 == 1:
                visited_temp[8 - false] = False
            else:
                visited_temp[6 - false] = False
            return visited_temp

        visited_list = []

        if t_fake == False:
            for i in range(4):
                visited_temp = copy.deepcopy(visited)
                visited_temp = inf_fake(visited_temp, i)
                if t_first1 == True and t_first2 == True:
                    for j in range(2):
                        visited_temp2 = copy.deepcopy(visited_temp)
                        visited_temp2 = inf_false(visited_temp2, j)
                        for k in range(2, 4):
                            visited_temp3 = copy.deepcopy(visited_temp2)
                            visited_temp3 = inf_false(visited_temp3, k)
                            visited_list.append(visited_temp3)
                elif t_first1 == True:
                    for j in range(2):
                        visited_temp2 = copy.deepcopy(visited_temp)
                        visited_temp2 = inf_false(visited_temp2, j)
                        visited_list.append(visited_temp2)

                elif t_first2 == True:
                    for j in range(2, 4):
                        visited_temp2 = copy.deepcopy(visited_temp)
                        visited_temp2 = inf_false(visited_temp2, j)
                        visited_list.append(visited_temp2)

        return visited_list


def DirectTransform(start, last):
    global PathOriginal
    global path_temp
    key = []
    PathOriginal = []
    path = []
    # x,y =start
    # maze_direct[x][y]=2
    x, y = last

    # maze_direct[x][y]=2
    if predecessors[x][y] == None or predecessors[x][y] == start:
        return key

    a, b = 0, 0
    # 用来做差比较

    while predecessors[x][y] != start:
        px, py = predecessors[x][y]

        # maze_visited[px][py]=2
        # maze_direct[px][py]=2
        element = (x, y) in cross
        # 指的是这个点是不是拐点或者路口
        PathOriginal.append((px, py))
        x -= px
        y -= py
        if x == a and y == b:
            if element == False:
                x, y = px, py
                # 考虑是在分岔路口的情况
                continue
        else:
            a = x
            b = y
        if len(path) != 0:
            tx, ty = path[-1]
            kx, ky = tx - x, ty - y
            if kx and ky:
                if x:
                    if ty + x:
                        key.append(1)
                    else:
                        key.append(2)
                else:
                    if tx + y:
                        key.append(2)
                    else:
                        key.append(1)
            else:
                key.append(0)
        path.append((a, b))
        x, y = px, py

    key.reverse()
    index = 2
    if last == (0, 18):
        index = 9
    elif len(PathOriginal) < 2:
        pass
    else:
        PathOriginal.reverse()
        path_temp = copy.deepcopy(PathOriginal)
        endingx, endingy = PathOriginal[-1]
        lastx, lasty = PathOriginal[-2]
        element = (lastx, lasty) in cross
        while True:
            path_temp.pop()
            if len(path_temp) < 2:
                index = 4
                break
            lastx, lasty = path_temp[-2]
            element = (lastx, lasty) in cross
            bookx = endingx - lastx
            booky = endingy - lasty
            if bookx != 0 and booky != 0:
                break
            index += 1
            if element == True:
                if index == 4:
                    path_temp.pop()
                break
    key.append(index)

    # print(path_temp)
    # print(PathOriginal)

    return key


def search():
    def update():
        global t_direction

        x0, y0 = Traverse.list_val(t_last)
        x, y = Traverse.list_val(t_next)

        Time.dijkstra((x0, y0), (x, y))
        dir = DirectTransform((x0, y0), (x, y))
        t_direction[t_last][t_next] = dir
        if t_last == -1 and t_next != -1 and t_next != t_len:
            t_distances[t_len][t_next] = distances[t_len][t_next]
        elif t_last == -1 and t_next != -1 and t_next != t_len:
            t_distances[t_last][t_next] = distances[t_last][t_next]

    def dfs():
        global t_next
        t_next = Global.dfs_ccl(t_visited)

        return t_next

    def dfs_inf():
        global t_next
        element = False
        if t_first1 == False and t_first2 == True:
            element = True
        elif t_first1 == True and t_first2 == False:
            element = True
        if element == False:
            t_next = Global.dfs_ccl(t_visited)
        else:
            t_next = Infer.ccl()

        return t_next

    global t_next
    element = True in t_visited
    if element == False:
        t_next = t_len
    else:
        t_next = dfs_inf()
        update()
    dir = t_direction[t_len][t_next]

    # print(t_next)
    return dir


if __name__ == "__main__":
    expection = 0
    epoch = 9
    for i in range(epoch):
        treasure = t_sets[i]
        Traverse.initial()

        gui = GUI(t_tranform)
        gui.mainloop()
