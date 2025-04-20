import time
import math
import cv2
import numpy as np
import copy

# import ast
import heapq
import serial
from .trial_algorithm import *


def turn_90(photo_place):
    # 逆时针90
    # 左下             左上           右上            右下
    # 读取原始图像
    src_image = cv2.imread(photo_place)
    # 构造旋转矩阵
    center = (src_image.shape[1] / 2, src_image.shape[0] / 2)
    M = cv2.getRotationMatrix2D(center, 90, 1)

    # 应用旋转变换
    rotated = cv2.warpAffine(src_image, M, (src_image.shape[1], src_image.shape[0]))
    # 显示结果
    # cv2.imshow("Rotated Image", rotated)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("maze_reshaped_90.jpg", rotated)


def green_yellow_detect(hsv):
    code_1 = 0
    code_2 = 0
    kernel = np.ones((5, 5), np.uint8)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_green = cv2.erode(mask_green, kernel, iterations=1)
    mask_green = cv2.dilate(mask_green, kernel, iterations=1)
    # cv2.imshow('green', mask_green)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 找到绿色轮廓并计算中心点坐标
    contours_green, _ = cv2.findContours(
        mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in contours_green:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        print(area)
        cnt_green = max(contours_green, key=cv2.contourArea)
        M_green = cv2.moments(cnt_green)
        cx_green = int(M_green["m10"] / M_green["m00"])
        cy_green = int(M_green["m01"] / M_green["m00"])
        green_center = (cx_green, cy_green)
        print(green_center)
        cv2.circle(img, green_center, 5, (0, 255, 0), -1)
        cv2.imwrite("green.jpg", img)
        code_1 = 1

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    # cv2.imshow('yellow', mask_yellow)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 找到绿色轮廓并计算中心点坐标
    contours_yellow, _ = cv2.findContours(
        mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in contours_yellow:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        print(area)
        cnt_yellow = max(contours_yellow, key=cv2.contourArea)
        M_yellow = cv2.moments(cnt_yellow)
        cx_yellow = int(M_yellow["m10"] / M_yellow["m00"])
        cy_yellow = int(M_yellow["m01"] / M_yellow["m00"])
        yellow_center = (cx_yellow, cy_yellow)
        print(yellow_center)
        cv2.circle(img, yellow_center, 5, (0, 255, 0), -1)
        cv2.imwrite("yellow.jpg", img)
        code_2 = 1

    if code_1 == 1 and code_2 == 0:
        # green
        return 1

    if code_1 == 0 and code_2 == 1:
        # yellow
        return 2
    if code_1 == 0 and code_2 == 0:
        # error
        return 0

    # 如果图片拍摄模糊，则很有可能模糊的区域会被识别为黄色，且通常面积比绿色多一倍
    if code_1 == code_2:
        # 对黄色轮廓和绿色轮廓分别计算面积、外接矩形和凸包
        area_yellow = cv2.contourArea(contours_yellow[0])
        area_green = cv2.contourArea(contours_green[0])

        rect_yellow = cv2.boundingRect(contours_yellow[0])
        rect_green = cv2.boundingRect(contours_green[0])

        # 检查绿色轮廓是否被黄色轮廓包围
        if (
            area_yellow > area_green
            and rect_yellow[0] < rect_green[0]
            and rect_yellow[1] < rect_green[1]
            and rect_yellow[0] + rect_yellow[2] > rect_green[0] + rect_green[2]
            and rect_yellow[1] + rect_yellow[3] > rect_green[1] + rect_green[3]
        ):
            print("The yellow contour encloses the green contour.")
            return 1
        else:
            print("error.There have yellow and green at the same time!!!")
            return 0


def red_true(hsv):  # red rectangle and green triangle
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_red = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_red:
        area = cv2.contourArea(cnt)
        if area < 1000:
            continue
        print(3333)
        x, y, w, h = cv2.boundingRect(cnt)
        print(x, y, w, h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imwrite("red_true.jpg", img)
        return 3
    return 4


# 3 have red 4 no red

# Color Range
lower_green = np.array([45, 50, 50])
upper_green = np.array([75, 255, 255])
lower_yellow = np.array([13, 100, 100])
upper_yellow = np.array([30, 255, 255])
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([125, 255, 255])
lower_red1 = np.array([0, 100, 100])  # red 1
upper_red1 = np.array([8, 255, 255])
lower_red2 = np.array([170, 100, 100])  # red 2
upper_red2 = np.array([180, 255, 255])


def capture_red():
    print("START")

    global cap
    # ret, frame = cap.read()
    # cv2.imwrite('frame_%d.jpg' % frame_count, frame)
    # cv2.imwrite("frame_cache.jpg", frame)
    # 	frame_count = frame_count + 1
    # cv2.imshow("frame", frame)
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break
    global img
    gamma = 0.55
    img = cv2.imread("frame_cache.jpg")
    print(img.shape)
    y0 = 300  # 这个是指矩形从上面在高度上切掉300
    y1 = img.shape[0]
    x0 = 0
    x1 = img.shape[1]
    cropped = img[y0:y1, x0:x1]  # 裁剪坐标为[y0:y1, x0:x1]
    cv2.imwrite("cropped.jpg", cropped)

    img = cv2.imread("cropped.jpg")
    print(img.shape)

    img_gamma = np.power(img / 255, gamma)
    img_gamma = np.uint8(img_gamma * 255)
    cv2.imwrite("frame_cache_gamma.jpg", img_gamma)

    hsv = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2HSV)

    count = green_yellow_detect(hsv)  # green 1 yellow 2 error 0

    print("yes!")
    print(count)

    if count == 1:  # green_tri
        count = red_true(hsv)
        print(count)
        if count == 3:
            print("red_true")
            print("red and green")
            return 1
        else:
            print("blue_fake")
            print("blue and green")
            return -2

    if count == 2:  # yellow_circles
        count = red_true(hsv)
        print(count)
        if count == 3:
            print("red_fake")
            print("red and yellow")
            return 2
        else:
            print("blue_true")
            print("blue and yellow")
            return -1

    if count == 0:
        print("error")
        return 0


def capture_blue():
    print("START")

    global cap
    ret, frame = cap.read()
    # cv2.imwrite('frame_%d.jpg' % frame_count, frame)
    cv2.imwrite("frame_cache.jpg", frame)
    # 	frame_count = frame_count + 1
    # cv2.imshow("frame", frame)
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break
    global img
    gamma = 0.55
    img = cv2.imread("frame_cache.jpg")
    print(img.shape)
    y0 = 300  # 这个是指矩形从上面在高度上切掉300
    y1 = img.shape[0]
    x0 = 0
    x1 = img.shape[1]
    cropped = img[y0:y1, x0:x1]  # 裁剪坐标为[y0:y1, x0:x1]
    cv2.imwrite("cropped.jpg", cropped)

    img = cv2.imread("cropped.jpg")
    print(img.shape)

    img_gamma = np.power(img / 255, gamma)
    img_gamma = np.uint8(img_gamma * 255)
    cv2.imwrite("frame_cache_gamma.jpg", img_gamma)

    hsv = cv2.cvtColor(img_gamma, cv2.COLOR_BGR2HSV)

    count = green_yellow_detect(hsv)  # green 1 yellow 2 error 0

    print("yes!")
    print(count)

    if count == 1:  # green_tri
        count = red_true(hsv)
        print(count)
        if count == 3:
            print("red_true")
            print("red and green")
            return -1
        else:
            print("blue_fake")
            print("blue and green")
            return 2

    if count == 2:  # yellow_circles
        count = red_true(hsv)
        print(count)
        if count == 3:
            print("red_fake")
            print("red and yellow")
            return -2
        else:
            print("blue_true")
            print("blue and yellow")
            return 1

    if count == 0:
        print("error")
        return 0


def dij_time(start, end):
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
    print(treasure)
    treasure.sort()
    print(treasure)
    t_len = len(treasure)

    # mat_list=[3, 1, 3, 1, 4, 2, 4, 2]

    print(treasure)
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
    print(mat_list)
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
    print(treasure)
    # print(mat_list)
    for index in range(2, 4):
        mat_id = t_len - 1 - index
        treasure[mat_id], treasure[index] = treasure[index], treasure[mat_id]
        # mat_list[mat_id],mat_list[index]=mat_list[index],mat_list[mat_id]
    print(treasure)
    mat_list = [1, 1, 2, 2, 3, 3, 4, 4]

    t_tranform = copy.deepcopy(treasure)
    t_tranform = MatrixTransform(t_tranform)
    print(t_tranform)
    wall()

    t_distances = [[float("inf")] * (t_len + 1) for _ in range(t_len + 1)]
    t_direction = [[] for _ in range(t_len + 1)]
    t_path_corner = [[] for _ in range(t_len + 1)]
    t_path = [[] for _ in range(t_len + 1)]

    t_visited = [True for _ in range(t_len)]
    pic_list = [0 for _ in range(t_len)]
    permutations = bool_permutations(t_visited)

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
                    # dij_time((x0,y0),(18,0))
                    # t_distances[i][j]=distances[18][0]

                elif j == t_len:
                    x = 0
                    y = 18
                    # dij_time((x0,y0),(0,18))

                    # direction=DirectTransform((x0,y0),(x,y))
                    # t_distances[i][j]=distances[0][18]

                else:
                    x = t_tranform[j][0]
                    y = t_tranform[j][1]
                dij_time((x0, y0), (x, y))

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

                dij_time((18, 0), (x, y))
                t_distances[i][j] = distances[x][y]
                direction = DirectTransform((18, 0), (x, y))
                t_direction[i].append(direction)
                t_path[i].append(PathOriginal)
                t_path_corner[i].append(path_temp)

    # print(t_direction)


def grd_time(pos):

    short_dis = float("inf")
    count = copy.deepcopy(pos)
    x0, y0 = list_val(count)
    # if t_last==-1:
    #     x0=18
    #     y0=0
    # else:
    #     x0=t_tranform[pos][0]
    #     y0=t_tranform[pos][1]

    #         dij_time((18,0),(x,y))
    #         dis=distances[x][y]
    #     else:
    #         dij_time((18,0),(x,y))

    #         dis=t_distances[pos][i]
    for i in range(t_len):

        if t_visited[i] == False:
            continue
        x, y = list_val(i)
        dijkstra((x0, y0), (x, y))
        dis = distances[x][y]
        if dis < short_dis:
            short_dis = dis
            pos = i

    return pos


def DirectTransform(start, last):
    # global maze_direct
    # 用来可视化路径
    # maze_direct = copy.deepcopy(maze)
    global PathOriginal
    global path_temp
    # global maze_visited
    # 用来确定走过的所有点是否合理
    # start=ListTransform(start)
    # start=tuple(start)
    # last=ListTransform(last)
    # last=tuple(last)

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


def dijkstra(start):

    # x，y=start
    # element =[x,y]  in t_wall_maze
    # start=ListTransform(start)
    # start=tuple(start)
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
        dfs(d)
        dfs_visited[i] = True
        dfs_p.pop()
        if len(dfs_p) == 0:
            dfs_last = t_last
        else:
            dfs_last = copy.deepcopy(dfs_p[-1])
        d -= dis


def dfs_time(d):
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
    x0, y0 = list_val[dfs_last]
    for i in range(len(dfs_visited)):
        if dfs_visited[i] == False:
            continue

        dfs_visited[i] = False
        dfs_p.append(i)
        x, y = list_val[i]

        dij_time((x0, y0), (x, y))
        dis = distances[x][y]
        d += dis
        dfs_last = copy.deepcopy(i)
        dfs_time(d)
        dfs_visited[i] = True
        dfs_p.pop()
        if len(dfs_p) == 0:
            dfs_last = t_last
        else:
            dfs_last = copy.deepcopy(dfs_p[-1])
        d -= dis


def trd_per_inf():
    global t_next
    t_next = inf_per_ccl()

    return t_next


def trd_dfs():
    global t_next
    t_next = dfs_ccl(t_visited)

    return t_next


def update():
    global t_direction

    x0, y0 = list_val(t_last)
    x, y = list_val(t_next)

    dij_time((x0, y0), (x, y))
    dir = DirectTransform((x0, y0), (x, y))
    t_direction[t_last][t_next] = dir
    if t_last == -1 and t_next != -1 and t_next != t_len:
        t_distances[t_len][t_next] = distances[t_len][t_next]
    elif t_last == -1 and t_next != -1 and t_next != t_len:
        t_distances[t_last][t_next] = distances[t_last][t_next]


def inf(visited):
    visited_list = []
    # if t_last==3:
    #     print("!")
    # visited_shortest=[]

    for inf in range(4):
        for fake in range(1, 5):
            #  inf 从0到3，表示全错，1对2错，1对2错，全对，
            # 二进制转化后可以发现规律

            visited_temp = copy.deepcopy(visited)

            # 表示的是否为该象限中第一个访问点
            true1 = True
            true2 = True
            if inf == 0:
                if t_self1 == 1 or t_self2 == 1:
                    continue
                true1 = False
                true2 = False
            elif inf == 1:
                if t_self1 == 1 or t_self2 == -1:
                    continue
                true1 = False
            elif inf == 2:
                if t_self1 == -1 or t_self2 == 1:
                    continue
                true2 = False
            elif inf == 2:
                if t_self1 == -1 or t_self2 == -1:
                    continue
            if t_fake == True:
                pass
            else:
                visited_temp[2 * fake - 2] = False
                visited_temp[2 * fake - 1] = False
            if t_first1 == True:
                if true1 == True:
                    visited_temp[1] = False
                    visited_temp[7] = False
                else:
                    visited_temp[0] = False
                    visited_temp[6] = False
            if t_first2 == True:
                if true2 == True:
                    visited_temp[3] = False
                    visited_temp[4] = False
                else:
                    visited_temp[2] = False
                    visited_temp[5] = False

            count = visited_temp.count(True) + t_score
            element = visited_temp in visited_list
            if element == False and count == 3:

                visited_list.append(visited_temp)
    return visited_list


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


def dfs_ccl_time(item):
    global dfs_distances
    global dfs_index
    global dfs_visited
    global dfs_p
    dfs_visited = copy.deepcopy(item)
    dfs_distances = float("inf")
    dfs_p = []
    dfs_index = -1
    dfs_time(0)

    return dfs_index


def wall():
    global t_wall
    global maze
    for i in range(t_len):
        x, y = list_val(i)
        element = [x, y] in t_wall_maze

        if element == True:
            t_wall.append([x, y])
            maze[x][y] = 1


def dfs_ccl(item):
    global dfs_distances
    global dfs_index
    global dfs_visited
    global dfs_p
    dfs_visited = copy.deepcopy(item)
    dfs_distances = float("inf")
    dfs_p = []
    dfs_index = -1
    dfs(0)

    return dfs_index


def trd_dfs_inf():

    global t_next
    element = False
    if t_first1 == False and t_first2 == True:
        element = True
    elif t_first1 == True and t_first2 == False:
        element = True
    if element == False:
        t_next = dfs_ccl(t_visited)
    else:
        t_next = inf_ccl()

    return t_next


def list_val(count):
    if count == -1:
        return 18, 0
    else:
        return t_tranform[count][0], t_tranform[count][1]


def search():
    global t_next
    element = True in t_visited
    if begin == True:

        if element == False:
            t_next = t_len
        else:
            # t_next=trd_dfs()
            t_next = trd_dfs_inf()
            # t_next=trd_per_inf()
            # print(dfs_last)
            update()
        dir = t_direction[t_len][t_next]
    else:
        if element == False:
            t_next = t_len

        else:
            t_next = trd_dfs_inf()
            # t_next=trd_per_inf()
            t_next = trd_dfs()

            update()

        dir = t_direction[t_last][t_next]
    print(t_next)
    return dir


def inf_ccl():
    global inf_visited
    global inf_distances
    global inf_index
    global dfs_distances
    global dfs_index
    global dfs_p
    global dfs_visited
    inf_index = [0 for _ in range(t_len)]
    inf_distances = [0 for _ in range(t_len)]

    inf_visited = inf(t_visited)
    for item in inf_visited:
        dfs_index = dfs_ccl(item)
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


def inf_per_ccl():
    global inf_visited
    global inf_distances
    global inf_index
    global dfs_distances
    global dfs_index
    global dfs_p
    global dfs_visited

    inf_index = [0 for _ in range(t_len)]
    inf_distances = [0 for _ in range(t_len)]

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
                    elif t_visited[1] == False and permutation[1] != t_visited[1]:
                        continue
                    elif t_visited[6] == False and permutation[6] != t_visited[6]:
                        continue
                    elif t_visited[7] == False and permutation[7] != t_visited[7]:
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
                    elif t_visited[3] == False and permutation[3] != t_visited[3]:
                        continue
                    elif t_visited[4] == False and permutation[4] != t_visited[4]:
                        continue
                    elif t_visited[5] == False and permutation[5] != t_visited[5]:
                        continue
            dfs_index = dfs_ccl(permutation)
            if dfs_index > 0:

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


def dijkstra(start):

    # x，y=start
    # element =[x,y]  in t_wall
    # start=ListTransform(start)
    # start=tuple(start)
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


def t_judge(visited):
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
            x, y = list_val(7 - t_last)
        else:
            pic_list[7 - t_last] = -1

            t_score += 1
            x, y = list_val(t_last)
        # maze[x][y]=0
        # element=[x,y]in t_wall
        # if element==True:
        #     t_wall.remove([x,y])

    elif abs(picture) == 2:
        t_fake = True
        visited[t_len - 1 - t_last] = False
        if picture == -2:

            pic_list[7 - t_last] = 2
        else:
            pic_list[7 - t_last] = -2
    # elif picture==0:
    #     x,y=list_val(t_last)
    # maze[x][y]=0
    # element=[x,y]in t_wall
    # if element==True:
    #     t_wall.remove([x,y])

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


def t_trans():
    global treasure
    l = copy.deepcopy(len(treasure))
    for i in range(l):
        x = treasure[i][0]
        y = treasure[i][1]
        treasure.append([9 - x, 9 - y])


from origin.trial_algorithm import *


def color_find_red_1(img):
    # 转为HSL色彩空间
    hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # 定义红色区间
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    # 使用inRange函数进行阈值分割，提取对应颜色区域
    mask_red = cv2.inRange(hsl, lower_red, upper_red)
    mask_red2 = cv2.inRange(hsl, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red, mask_red2)
    # 使用 findContours 函数查找红色矩形的轮廓
    contours, hierarchy = cv2.findContours(
        mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # 遍历红色矩形的轮廓，获取其坐标信息
    coordinates_red = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 1 and h > 1:  # 过滤掉面积太小的轮廓
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            coordinates_red.append([x, y, w, h])  # 获取的是矩形左上角坐标已经长和宽
            coordinates_red = np.array(coordinates_red)
    # print(coordinates_red)
    # 显示结果图像
    # cv2.imshow("red", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return coordinates_red


def color_find_blue(img):
    # 将图像转换为HSV颜色空间
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 定义要提取的颜色范围
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    # 提取蓝色区域的掩膜
    mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
    # 对掩膜进行形态学处理，去除噪声
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    # 使用 findContours 函数查找红色矩形的轮廓
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # 遍历红色矩形的轮廓，获取其坐标信息
    coordinates_blue = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 20 and h > 20:  # 过滤掉面积太小的轮廓
            #            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            coordinates_blue.append([x, y, w, h])  # 获取的是矩形左上角坐标已经长和宽
            coordinates_blue = np.array(coordinates_blue)
    # print(coordinates_blue)
    # 显示结果图像
    # cv2.imshow("blue", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return coordinates_blue


def detect_circle(pic):
    count_all_all = 1
    size = 70
    while count_all_all == 1:
        # 二值化处理
        ret, pic = cv2.threshold(
            pic, size, 255, cv2.THRESH_BINARY
        )  # _INV | cv2.THRESH_OTSU)[1]
        count_all = 1
        ksize = (5, 5)
        count_more_1 = 3
        while count_all == 1:
            pic = cv2.GaussianBlur(pic, ksize, 0)  ### 高斯滤波
            cv2.imshow("result", pic)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.imwrite("test.png", pic)

            key_param2 = 15

            count = 1
            count_more = 0
            median_1 = []
            while count == 1:
                arr = []
                idx = []
                circles = cv2.HoughCircles(
                    pic,
                    cv2.HOUGH_GRADIENT,
                    1,
                    10,
                    param1=200,
                    param2=key_param2,
                    minRadius=0,
                    maxRadius=30,
                )  ### 返回三个值，圆心x ，圆心y ，半径
                circles = circles[0, :, :]
                circles = np.uint16(np.around(circles))

                # 对识别到的圆形进行检测
                for i in range(len(circles)):
                    arr.append(circles[i][2])
                # print(arr)
                # 剔除异常值
                median = np.median(arr)
                median_1.append(median)

                # 防止中间值随整体数值变大而变大
                mean_median = np.mean(median_1)
                if median - mean_median > 2:
                    median = mean_median
                    median_1.pop()

                # print(median)
                z_scores = arr - median
                idx = np.where(np.abs(z_scores) > 3)[0]
                # print(idx)
                if len(idx) > 0:
                    for i in range(len(idx)):
                        row_idx = idx[i] - i
                        circles[row_idx:-1] = circles[row_idx + 1 :]
                        circles = circles[:-1]
                # print("剩余" + str(len(circles)))

                # 对异常情况处理
                if count_more > 8 and count_more_1 == 3:
                    ksize = (3, 3)
                    count = 0
                    count_more_1 = 1
                if count_more > 8 and count_more_1 == 1:
                    ksize = (1, 1)
                    count = 0

                if count_more_1 == 1 and count_more > 8:
                    count = 0
                    count_all = 0
                    size = size - 10
                    print("-10")

                # 判断参数下一步如何调整
                if len(circles) < 8:
                    key_param2 = key_param2 - 1
                elif len(circles) > 8:
                    key_param2 = key_param2 + 1
                    count_more = count_more + 1
                else:  # 退出所有循环
                    count = 0
                    count_all = 0
                    count_all_all = 0
                    # print(circles)
                    return circles


def color_find_blue_1(img):
    # 将图像转换为HSV颜色空间
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 定义要提取的颜色范围
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    # 提取蓝色区域的掩膜
    mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
    # 对掩膜进行形态学处理，去除噪声
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    # 使用 findContours 函数查找红色矩形的轮廓
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # 遍历红色矩形的轮廓，获取其坐标信息
    coordinates_blue = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 1 and h > 1:  # 过滤掉面积太小的轮廓
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            coordinates_blue.append([x, y, w, h])  # 获取的是矩形左上角坐标已经长和宽
            coordinates_blue = np.array(coordinates_blue)
    # print(coordinates_blue)

    # 显示结果图像
    # cv2.imshow("blue", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return coordinates_blue


def color_find_red(img):
    # 转为HSL色彩空间
    hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # 定义红色区间
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    # 使用inRange函数进行阈值分割，提取对应颜色区域
    mask_red = cv2.inRange(hsl, lower_red, upper_red)
    mask_red2 = cv2.inRange(hsl, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red, mask_red2)
    # 使用 findContours 函数查找红色矩形的轮廓
    contours, hierarchy = cv2.findContours(
        mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # 遍历红色矩形的轮廓，获取其坐标信息
    coordinates_red = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 20 and h > 20:  # 过滤掉面积太小的轮廓
            #           cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            coordinates_red.append([x, y, w, h])  # 获取的是矩形左上角坐标已经长和宽
            coordinates_red = np.array(coordinates_red)
    # print(coordinates_red)
    # 显示结果图像
    # cv2.imshow("red", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return coordinates_red


def version():

    while True:
        ret, frame = cap.read()
        size = ser.inWaiting()
        res = ser.read(size)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
        cv2.imshow("thresh", thresh)
        if cv2.waitKey(1) & res == b"\x04\x04\x04":
            cv2.imwrite("333_test.jpg", thresh)
            break
    cv2.destroyAllWindows()
    # for i in range(3):
    #     cap = cv2.VideoCapture(0)
    #     time.sleep(3)
    #     ret, frame = cap.read()
    #     cv2.imwrite("maze_photo_{0}.jpg".format(i), frame)
    #     print(i)

    global treasure
    # 读取校正后的迷宫图像
    while True:
        ret, frame = cap.read()
        treasure = []

        cv2.imwrite("maze_photo.jpg", frame)
        # src = cv2.imread("maze_photo_0.jpg")

        # 转换为灰度图像
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)  ### 高斯滤波
        red = color_find_red(src)
        blue = color_find_blue(src)
        if len(blue) == 0 or len(red) == 0:
            continue

        if blue[0][0] > red[0][0]:  # blue above
            blue[0][0] = blue[0][0] + blue[0][2]
            red[0][1] = red[0][1] + red[0][2]
            blue[0][0] += 5
            blue[0][1] -= 5
            red[0][0] -= 5
            red[0][1] += 5
        else:
            red[0][0] = red[0][0] + red[0][2]
            blue[0][1] = blue[0][1] + blue[0][2]
            red[0][0] += 5
            red[0][1] -= 5
            blue[0][0] -= 5
            blue[0][1] += 5

        west_down = np.zeros((1, 2))
        west_up = np.zeros((1, 2))
        east_down = np.zeros((1, 2))
        east_up = np.zeros((1, 2))

        west_down[0][0] = red[0][0]
        west_down[0][1] = red[0][1]
        west_up[0][0] = red[0][0]
        west_up[0][1] = blue[0][1]
        east_up[0][0] = blue[0][0]
        east_up[0][1] = blue[0][1]
        east_down[0][0] = blue[0][0]
        east_down[0][1] = red[0][1]

        # 设置sortPoints
        sortPoints = np.zeros((4, 2), dtype=np.int32)
        sortPoints[0][0] = west_down[0][0]  # 左下
        sortPoints[0][1] = west_down[0][1]
        sortPoints[1][0] = west_up[0][0]  # 左上
        sortPoints[1][1] = west_up[0][1]
        sortPoints[2][0] = east_up[0][0]  # 右上
        sortPoints[2][1] = east_up[0][1]
        sortPoints[3][0] = east_down[0][0]  # 右下
        sortPoints[3][1] = east_down[0][1]

        # 利用sortPoints的四个点进行透视变换
        minSide = min(src.shape[:2])
        center_x = src.shape[0] // 2
        center_y = src.shape[1] // 2

        srcTri = np.array(
            [sortPoints[0], sortPoints[1], sortPoints[2], sortPoints[3]],
            dtype=np.float32,
        )
        dstTri = np.array(
            [
                (center_x - 0.45 * minSide, center_y - 0.45 * minSide),
                (center_x + 0.45 * minSide, center_y - 0.45 * minSide),
                (center_x + 0.45 * minSide, center_y + 0.45 * minSide),
                (center_x - 0.45 * minSide, center_y + 0.45 * minSide),
            ],
            dtype=np.float32,
        )

        perspImage = np.zeros(
            (src.shape[0], src.shape[1], src.shape[2]), dtype=np.uint8
        )
        # 提取图像映射
        transmtx = cv2.getPerspectiveTransform(srcTri, dstTri)

        perspImage = cv2.warpPerspective(src, transmtx, perspImage.shape[:2])

        # cv2.imshow("result5", perspImage)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite("maze_reshaped.jpg", perspImage)

        turn_90("maze_reshaped.jpg")

        # 读取图像
        src = cv2.imread("maze_reshaped_90.jpg")
        img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (3, 3), 0)  ### 高斯滤波

        # 创建FAST角点检测器
        fast = cv2.FastFeatureDetector_create()

        # 检测角点
        keypoints = fast.detect(img, None)

        # 二值化处理
        ret, thresh = cv2.threshold(
            img, 120, 255, cv2.THRESH_BINARY
        )  # _INV | cv2.THRESH_OTSU)[1]

        # 获取迷宫矩阵
        grid_size = 1  # 设置网格大小
        maze_matrix = np.zeros_like(thresh, dtype=np.int64)
        # for contour in contours:
        for y in range(0, maze_matrix.shape[0], grid_size):
            for x in range(0, maze_matrix.shape[1], grid_size):
                if thresh[y, x] == 0:  # 像素点为黑色
                    maze_matrix[y, x] = 1
        print(maze_matrix[0][0])
        # 保存为文本
        np.savetxt("matrix.txt", maze_matrix.astype(np.int64), fmt="%d")

        # 绘制角点
        img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))

        # 获取角点坐标
        points = []
        for kp in keypoints:
            points.append(kp.pt)

        # 检测每个角点的像素值
        west_up = 10000
        west_up_array = np.zeros((1, 2))  # (y,x)
        west_down = 10000
        west_down_array = np.zeros((1, 2))
        east_up = 10000
        east_up_array = np.zeros((1, 2))
        east_down = 10000
        east_down_array = np.zeros((1, 2))

        h, w = img.shape
        # 先判断点是否为黑色，然后再判断到边界点的距离，并记录下距离最小点的坐标
        for p in points:
            x, y = p  # 点在矩阵中的坐标为(y,x)
            x = int(x)
            y = int(y)
            if (
                maze_matrix[y, x] == 1
            ):  # 检测该点为0即黑色, maze_matrix[w, h] = 1  # maze_matrix 为像素矩阵
                distance_wu = math.sqrt((y - 0) ** 2 + (x - 0) ** 2)
                distance_wd = math.sqrt((h - y) ** 2 + (x - 0) ** 2)
                distance_eu = math.sqrt((y - 0) ** 2 + (w - x) ** 2)
                distance_ed = math.sqrt((h - y) ** 2 + (w - x) ** 2)
                if distance_wu < west_up:
                    west_up = distance_wu
                    west_up_array[0][0] = y
                    west_up_array[0][1] = x
                if distance_wd < west_down:
                    west_down = distance_wd
                    west_down_array[0][0] = y
                    west_down_array[0][1] = x
                if distance_eu < east_up:
                    east_up = distance_eu
                    east_up_array[0][0] = y
                    east_up_array[0][1] = x
                if distance_ed < east_down:
                    east_down = distance_ed
                    east_down_array[0][0] = y
                    east_down_array[0][1] = x

        red = color_find_red(src)
        blue = color_find_blue(src)

        if blue[0][0] > red[0][0]:  # blue above
            red[0][0] = red[0][0] + red[0][2]
            red[0][1] = red[0][1] + red[0][2]
            east_up_array[0][0] = blue[0][1]
            east_up_array[0][1] = blue[0][0]
            west_down_array[0][0] = red[0][1]
            west_down_array[0][1] = red[0][0]
        else:
            blue[0][0] = blue[0][0] + blue[0][2]
            blue[0][1] = blue[0][1] + blue[0][2]
            east_up_array[0][0] = red[0][1]
            east_up_array[0][1] = red[0][0]
            west_down_array[0][0] = blue[0][1]
            west_down_array[0][1] = blue[0][0]

        # 显示图像
        # cv2.imshow("Image with keypoints", img_with_keypoints)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite("keypoints.jpg", img_with_keypoints)

        # 设置sortPoints
        sortPoints = np.zeros((4, 2), dtype=np.int32)
        sortPoints[0][0] = west_down_array[0][1]
        sortPoints[0][1] = west_down_array[0][0]
        sortPoints[1][0] = west_up_array[0][1]
        sortPoints[1][1] = west_up_array[0][0]
        sortPoints[2][0] = east_up_array[0][1]
        sortPoints[2][1] = east_up_array[0][0]
        sortPoints[3][0] = east_down_array[0][1]
        sortPoints[3][1] = east_down_array[0][0]

        # 利用sortPoints的四个点进行透视变换
        minSide = min(thresh.shape[:2])
        center_x = thresh.shape[0] // 2
        center_y = thresh.shape[1] // 2

        srcTri = np.array(
            [sortPoints[0], sortPoints[1], sortPoints[2], sortPoints[3]],
            dtype=np.float32,
        )
        dstTri = np.array(
            [
                (center_x - 0.45 * minSide, center_y - 0.45 * minSide),
                (center_x + 0.45 * minSide, center_y - 0.45 * minSide),
                (center_x + 0.45 * minSide, center_y + 0.45 * minSide),
                (center_x - 0.45 * minSide, center_y + 0.45 * minSide),
            ],
            dtype=np.float32,
        )

        perspImage = np.zeros(
            (src.shape[0], src.shape[1], src.shape[2]), dtype=np.uint8
        )
        # 提取图像映射
        transmtx = cv2.getPerspectiveTransform(srcTri, dstTri)

        # binImage = cv2.bitwise_not(thresh)  # 二值图像反色
        perspImage = cv2.warpPerspective(src, transmtx, perspImage.shape[:2])

        print(sortPoints)
        # 透视变换后的四个角点坐标
        perspPoints = cv2.perspectiveTransform(
            np.array([sortPoints], dtype=np.float32), transmtx
        )
        print(perspPoints)
        # perspPoints[0][i][0] and [0][i][1]
        # cv2.imshow("result5", perspImage)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite("maze_reshaped.jpg", perspImage)

        # 0 west_up
        # 1 east_up
        # 2 east_down
        # 3 west_down

        w = perspPoints[0][1][0] - perspPoints[0][0][0]
        h = perspPoints[0][3][1] - perspPoints[0][0][1]
        #    print(w)
        #    print(h)
        w_per = w / 10
        h_per = h / 10
        #    print(w_per)
        #    print(h_per)

        src = cv2.imread("maze_reshaped.jpg")
        img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # img = cv2.GaussianBlur(img, (5, 5), 0)        ### 高斯滤波
        # img = cv2.medianBlur(img, 5)
        # img = cv2.bilateralFilter(img, 9, 75, 75)
        circles = detect_circle(img)

        src = cv2.imread("maze_reshaped.jpg")
        img = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        for i in range(len(circles)):
            cv2.circle(
                src, (int(circles[i][0]), int(circles[i][1])), 25, (0, 0, 255), -1
            )
        cv2.imshow("result", src)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imwrite("circles_points.jpg", src)

        count = 1  # 1 go on  0 go back detect_circles

        for i in range(len(circles)):
            cv2.circle(
                src, (int(circles[i][0]), int(circles[i][1])), 25, (0, 0, 255), -1
            )
            circles[i][0] = (circles[i][0] - perspPoints[0][0][0]) / w_per
            circles[i][1] = (circles[i][1] - perspPoints[0][0][1]) / h_per
            print(circles[i])

        # 判断是否有重复 or 位于临近一格
        for i in range(len(circles)):
            for j in range(i, len(circles)):
                if i == j:
                    continue
                w_count = abs(int(circles[i][0]) - int(circles[j][0]))
                h_count = abs(int(circles[i][1]) - int(circles[j][1]))
                length = w_count + h_count
                if length < 2:
                    count = 0
                    break
            if count == 0:
                break
            x = circles[i][0]
            y = circles[i][1]
            treasure.append([x, y])
        # 在每一个不同的模块识别判断后加入go back 指令，节省时间
        if count == 0:

            print("go back1")

            continue

        for i in range(len(treasure)):
            # for j in range(len(possible_treasure)):
            x = treasure[i][0]
            y = treasure[i][1]
            element = [9 - x, 9 - y] in treasure
            if element == False:

                count = 0
                break

        if count == 0:

            print("go back2")

            continue
        break
    # 读取图像


cap = cv2.VideoCapture(0)
ser = serial.Serial("/dev/ttyAMA0", 115200)

if __name__ == "__main__":

    version()
    # 这个是拍照识别宝藏图的执行函数

    initial()
    # 这个是路径规划执行初始化的函数
    # ser = serial.Serial("/dev/ttyAMA0", 115200)

    try:
        while True:
            # 指令这样的：正常行驶还是原来那样子，加了03 05指令，表示掉头前行（撞击拐点宝藏后）和将到达终点冲出去
            # 但是，到了宝藏前，我会发FF F0 00 01 02,表示直行，左转，右转后有宝藏，
            # 识别到宝藏后03 04 05 06 07，
            # 分别表示胡同点真宝藏距离一格撞击掉头前行到上一个节点，两格撞击掉头前行到上一个节点，假宝藏原地掉头，
            # 拐点宝藏距离一个格子撞击停止，两格撞击停止，

            begin = True
            end = True
            size = -1
            res = bytes(2)
            t_last = -1
            dfs_last = t_last
            count = -1
            key = []
            # t_temp=copy.deepcopy(t_last)
            key = search()
            # 这个是路径规划指令的更新的代码，
            # 但是路径规划的距离没有考虑动态墙去掉的情况，还在写，有一个t_distances的更新，
            # 但是这个算好下一个宝藏点以后对于前面一个宝藏点与下一个点距离的更新
            # 距离的更新可以继续改，不过运算量会增加，可能影响小车速度
            # 距离路线还是对的，但是搜索算出的距离的情况有可能是让小车多绕了一小段路，对于整体结果可能有细微的影响

            LenList = len(key)
            print(key)

            print(t_next)
            ser.flushInput()

            while True:

                print("loading")

                size = ser.inWaiting()
                time.sleep(0.01)
                ser.write(b"\xFF\xFF\xFF\xFE")

                res = ser.read(size)

                print(res)

                if res == b"\x02\x02\x02":
                    print(1)
                    break

            begin = False
            while True:
                while True:
                    size = ser.inWaiting()
                    if size != 0:
                        res = ser.read(size)
                        print(res)

                    if res == b"\x00\x00\x00" or res == b"\x03\x03\x03":
                        count += 1
                        break
                if res == b"\x03\x03\x03":
                    break
                if count == LenList - 2:
                    print("before_target")
                    if t_next == t_len:
                        # 表示到达终点的情况
                        ser.write(b"\xFF\xFF\x05\xFE")
                        print("out")
                        end = False
                        count = 0
                        print(b"\xFF\xFF\x05\xFE")
                        while True:
                            pass
                    else:
                        # 表示到达宝藏点的情况

                        if key[count] == 2:
                            ser.write(b"\xFF\xF0\x02\xFE")

                            print(b"\xFF\xF0\x02\xFE")
                        elif key[count] == 1:
                            ser.write(b"\xFF\xF0\x01\xFE")

                            print(b"\xFF\xF0\x01\xFE")

                        elif key[count] == 0:

                            ser.write(b"\xFF\xF0\x00\xFE")

                            print(b"\xFF\xF0\x00\xFE")
                    ser.flushInput()
                    while True:
                        size = ser.inWaiting()

                        if size != 0:
                            res = ser.read(3)
                            print(res)
                            # if res != b"\x01\x01\x01":
                            #     ser.flushInput()
                            #     continue
                            if res == b"\x01\x01\x01" or res == b"\x03\x03\x03":
                                ser.flushInput()
                                break

                elif count == LenList - 1:
                    # 为什么没有if t_next==t_len的判断？因为if t_next==t_len:
                    # 那么在前面一步我给发送05指令就直接跳出了，不会进入这个循环
                    # 两段代码就是除了那个预判那里不一样而已，后面还有注释
                    print(pic_list)
                    # 存的是宝藏点的拍照和象限预判的结果
                    print("treasure")

                    if pic_list[t_next] == 0:

                        # 0表示没有预判到宝藏信息的情况
                        time.sleep(0.5)

                        picture = capture_red()
                        # time.sleep(1)
                        # print(PathOriginal)

                        # dir_last=copy.deepcopy(PathOriginal)
                        t_visited = t_judge(t_visited)
                        print("end")
                        print(PathOriginal)
                        pathx, pathy = PathOriginal[-1]
                        key_last = copy.deepcopy(key)
                        print(t_visited)
                        key = search()
                        print(PathOriginal)

                        count = -1
                        print(key)
                        if picture == 1:
                            x, y = list_val(t_last)
                            print((x, y))
                            element = [x, y] in corner
                            if element == False:
                                # 拐点
                                if key_last[LenList - 1] == 2:
                                    print(b"\xFF\xF0\x06\xFE")
                                    ser.write(b"\xFF\xF0\x06\xFE")

                                elif key_last[LenList - 1] == 4:
                                    print(b"\xFF\xF0\x07\xFE")
                                    ser.write(b"\xFF\xF0\x07\xFE")

                                # print((pathx,pathy))
                                # turn=(pathx,pathy) in PathOriginal
                                # if turn==True:
                                #     key.insert(0,3)
                                # else:

                                #     deltax,deltay= PathOriginal[0]
                                #     if deltax-pathx+deltay-pathy==0:
                                #         key.insert(0,2)
                                #     else:
                                #         key.insert(0,1)
                            else:
                                if key_last[LenList - 1] == 2:
                                    print(b"\xFF\xF0\x03\xFE")
                                    ser.write(b"\xFF\xF0\x03\xFE")

                                elif key_last[LenList - 1] == 4:
                                    ser.write(b"\xFF\xF0\x04\xFE")

                                    print(b"\xFF\xF0\x04\xFE")

                            print("crash")
                        elif picture == 0:
                            turn = (pathx, pathy) in PathOriginal
                            x, y = list_val(t_last)
                            print((x, y))
                            element = [x, y] in corner
                            if turn == True:
                                ser.write(b"\xFF\xF0\x05\xFE")

                                print(b"\xFF\xF0\x05\xFE")
                            else:
                                if element == False:
                                    if key_last[LenList - 1] == 2:
                                        print(b"\xFF\xF0\x06\xFE")

                                        ser.write(b"\xFF\xF0\x06\xFE")

                                    elif key_last[LenList - 1] == 4:

                                        ser.write(b"\xFF\xF0\x07\xFE")

                                        print(b"\xFF\xF0\x07\xFE")

                                else:
                                    if key_last[LenList - 1] == 2:

                                        ser.write(b"\xFF\xF0\x03\xFE")

                                        print(b"\xFF\xF0\x03\xFE")
                                    elif key_last[LenList - 1] == 4:
                                        ser.write(b"\xFF\xF0\x04\xFE")

                                        print(b"\xFF\xF0\x04\xFE")

                        else:
                            print(b"\xFF\xF0\x05\xFE")
                            ser.write(b"\xFF\xF0\x05\xFE")

                            print("rejected")
                        LenList = len(key)

                        print(pic_list)
                    else:

                        # 预判到宝藏是真宝藏的情况，为什么只要不是0就是真宝藏？
                        # 因为我加了一个bool类型t_visited的推断,不用拍照的话肯定是遍历真宝藏，
                        # 可以看看t_judge函数的实现，改函数里面的算法（dfs，inf）可以先不管
                        # 这一段代码就是除了拍照那里读取的是预判的结果，其他都不变

                        picture = pic_list[t_next]
                        # time.sleep(1)
                        # print(PathOriginal)

                        # dir_last=copy.deepcopy(PathOriginal)
                        t_visited = t_judge(t_visited)
                        print("end")
                        print(PathOriginal)
                        pathx, pathy = PathOriginal[-1]
                        # 这一步读取路径的最后一个点，然后在判断是否需要原来返回
                        key_last = copy.deepcopy(key)
                        print(t_visited)
                        key = search()
                        print(PathOriginal)

                        count = -1
                        print(key)
                        # 以上也是路径的更新
                        if picture == 1:
                            x, y = list_val(t_last)
                            print((x, y))
                            element = [x, y] in corner
                            if element == False:
                                # 拐点，撞击就停在那里不动了，下一个宝藏的行走指令需要改动
                                if key_last[LenList - 1] == 2:
                                    print(b"\xFF\xF0\x06\xFE")
                                    ser.write(b"\xFF\xF0\x06\xFE")

                                elif key_last[LenList - 1] == 4:
                                    print(b"\xFF\xF0\x07\xFE")
                                    ser.write(b"\xFF\xF0\x07\xFE")

                                # 看看是距离几个格子，然后发对应的指令

                                print((pathx, pathy))
                                turn = (pathx, pathy) in PathOriginal
                                # 判断，如果原路返回，还是绕行，

                                # if turn==True:
                                #     key.insert(0,3)

                                # else:
                                #     # 如果绕行，就要根据新的路径里面的经过的第一点的坐标来运算
                                #     deltax,deltay= PathOriginal[0]
                                #     if deltax-pathx+deltay-pathy==0:
                                #         key.insert(0,2)
                                #     else:
                                #         key.insert(0,1)

                            else:
                                # 胡同点，正常走，下一个宝藏的行走指令不需要改动
                                if key_last[LenList - 1] == 2:
                                    print(b"\xFF\xF0\x03\xFE")
                                    ser.write(b"\xFF\xF0\x03\xFE")

                                elif key_last[LenList - 1] == 4:
                                    ser.write(b"\xFF\xF0\x04\xFE")

                                    print(b"\xFF\xF0\x04\xFE")

                            print("crash")
                        elif picture == 0:
                            # 识别不到，但是当作是墙开启了，

                            turn = (pathx, pathy) in PathOriginal

                            print((x, y))

                            # 不管是否在胡同点还是拐点，我只要判断下一步要不要掉头就行
                            if turn == True:
                                # 如果下一段路径需要掉头，就当作是假宝藏掉头，

                                ser.write(b"\xFF\xF0\x05\xFE")
                                print(b"\xFF\xF0\x05\xFE")
                            else:
                                # 如果下一段路径不需要掉头，当作是拐点真宝藏绕过去

                                if key_last[LenList - 1] == 2:
                                    print(b"\xFF\xF0\x06\xFE")
                                    ser.write(b"\xFF\xF0\x06\xFE")

                                elif key_last[LenList - 1] == 4:
                                    ser.write(b"\xFF\xF0\x07\xFE")

                                    print(b"\xFF\xF0\x07\xFE")

                        else:
                            print(b"\xFF\xF0\x05\xFE")
                            ser.write(b"\xFF\xF0\x05\xFE")

                            print("rejected")
                        LenList = len(key)

                        print(pic_list)

                    if count == 0 and end == False:
                        break
                    ser.flushInput()

                    while True:
                        size = ser.inWaiting()

                        if size != 0:
                            res = ser.read(3)
                            print(res)
                            if res == b"\x01\x01\x01" or res == b"\x03\x03\x03":
                                ser.flushInput()
                                break

                            # if res != b"\x01\x01\x01":
                            #
                            #     continue

                else:
                    ser.write(b"\xFF")
                    ser.write(b"\xFF")

                    if key[count] == 0:
                        print("is0")
                        ser.write(b"\x00")

                    elif key[count] == 1:
                        print("is1")
                        ser.write(b"\x01")

                    elif key[count] == 2:
                        print("is2")
                        ser.write(b"\x02")

                    elif key[count] == 3:
                        print("is3")
                        ser.write(b"\x03")
                    ser.write(b"\xFE")

                    ser.flushInput()

                    while True:

                        size = ser.inWaiting()

                        if size != 0:
                            res = ser.read(3)
                            print(res)

                            # if res != b"\x01\x01\x01":
                            #
                            #     continue
                            if res == b"\x01\x01\x01" or res == b"\x03\x03\x03":
                                ser.flushInput()
                                break

                if res == b"\x03\x03\x03":
                    break
            if res == b"\x03\x03\x03":
                continue

            if count == 0 and end == False:
                break

    except KeyboardInterrupt:
        ser.close()
