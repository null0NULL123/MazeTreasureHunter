import numpy as np
import copy
import ast
import heapq

key_code = 0
frame_count = 1
direct_parameter = 0.4
cross_parameter = 0
capture_parameter = 5
maze_parameter = 206
photo_number = 3
size = 0
begin = True
end = True
matrix = 0
mat_list = []
mat_id = 2
picture = 0
pic_error = 0
pic_list = []
PathOriginal = []
t_path = []
t_path_corner = []
pathx = 0
pathy = 0
# 0未识别，正负1 2表示真假，正负3从镜像可以判断颜色而不知真伪，
# 4识别不出，56789可以自行设置功能
# 以此对于dfs的最好的结果进行筛选
treasure = []
t_len = len(treasure)
t_tranform = []
t_visited = []
t_distances = []
t_direction = []
path_temp = []
t_wall = []
t_next = -1
t_last = -1
# 用来判断复位和起点
t_score = 0
t_dir = 0
t_temp = copy.deepcopy(t_last)
t_first1 = True
t_first2 = True
t_fake = False
# t_true1=True
# t_true2=True
t_error = 0
t_startx = 0
t_starty = 0
t_endx = 18
t_endy = 0

t_self1 = 0
t_self2 = 0
t_score = 0
# t_traversal=0

# 0未知，1己方，-1对方,标号表示的是访问的象限及其对面
# True表示识别到假宝藏，False表示未识别到

# from main import temp
startx = 0
starty = 0
maze_endx = 18
maze_endy = 0
maze_direct = []
predecessors = []
distances = []

dfs_p = []
dfs_distances = float("inf")
dfs_index = 0
dfs_visited = []
dfs_last = t_last
inf_shortest = True
inf_parameter = 0
inf_p = []
inf_distances = []
inf_index = []
inf_visited = []
permutations = []

maze = np.loadtxt("maze.txt", dtype=int)
m = len(maze)
n = len(maze[0])

maze_visited = maze.copy()
t_pos = [
    (0, 2),
    (0, 4),
    (0, 5),
    (0, 9),
    (1, 3),
    (1, 6),
    (1, 7),
    (1, 8),
    (2, 2),
    (2, 5),
    (2, 8),
    (2, 9),
    (3, 0),
    (3, 3),
    (3, 4),
    (3, 8),
    (3, 9),
    (4, 7),
    (4, 8),
    (9, 7),
    (9, 5),
    (9, 4),
    (9, 0),
    (8, 6),
    (8, 3),
    (8, 2),
    (8, 1),
    (7, 7),
    (7, 4),
    (7, 1),
    (7, 0),
    (6, 9),
    (6, 6),
    (6, 5),
    (6, 1),
    (6, 0),
    (5, 2),
    (5, 1),
]
# t_wall_pos=[[0, 5], [1, 3], [1, 6], [1, 7], [2, 6], [2, 8], [3, 3], [3, 8], [3, 9], [9, 4], [8, 6], [8, 3], [8, 2], [7, 3], [7, 1], [6, 6], [6, 1], [6, 0]]
t_wall_maze = [
    [18, 10],
    [16, 6],
    [16, 12],
    [16, 14],
    [14, 10],
    [14, 12],
    [14, 16],
    [12, 6],
    [12, 16],
    [12, 18],
    [0, 8],
    [2, 12],
    [2, 6],
    [2, 4],
    [4, 8],
    [4, 6],
    [4, 2],
    [6, 12],
    [6, 2],
    [6, 0],
]
corner = [
    [16, 16],
    [14, 4],
    [14, 18],
    [12, 0],
    [12, 8],
    [10, 16],
    [2, 2],
    [4, 14],
    [4, 0],
    [6, 18],
    [6, 10],
    [8, 2],
]
cross = [
    (10, 0),
    (0, 2),
    (4, 4),
    (12, 4),
    (10, 4),
    (0, 6),
    (6, 6),
    (8, 6),
    (10, 6),
    (14, 6),
    (2, 8),
    (2, 10),
    (16, 8),
    (16, 10),
    (4, 12),
    (8, 12),
    (10, 12),
    (12, 12),
    (18, 12),
    (8, 14),
    (6, 14),
    (14, 14),
    (18, 16),
    (8, 18),
]
# 已更新


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
        content = file.read()  # 读取文本文件内容
        my_list = ast.literal_eval(content)  # 使用ast模块解析文本中的列表格式
    return my_list


def MatrixTransform(item):
    for i in range(len(item)):
        item[i][0] = 18 - (2 * item[i][0])
        item[i][1] = 2 * item[i][1]
    return item


def ListTransform(item):
    item[0] = 18 - (2 * item[0])
    item[1] = 2 * item[1]
    return item


# key_code = 0
# frame_count = 1
# direct_parameter=0.5
# cross_parameter=0
# capture_parameter=5
# maze_parameter=206
# photo_number=3
# size=0
# begin=True
# end=True
# matrix=0
# mat_list=[]
# mat_id=2
# picture=0
# pic_error=0
# pic_list=[]

# # 0未识别，正负1 2表示真假，正负3从镜像可以判断颜色而不知真伪，
# # 4识别不出，56789可以自行设置功能
# # 以此对于dfs的最好的结果进行筛选
# treasure = []
# t_len=len(treasure)
# t_tranform=[]
# t_visited=[]
# t_distances=[]
# t_direction=[]
# t_lastx=0
# t_lasty=0

# t_wall=[]
# t_next=-1
# t_last=-1
# t_temp=t_last
# # 用来判断复位和起点
# t_score=0
# t_dir=0

# t_first1=True
# t_first2=True
# t_fake=False
# # t_true1=True
# # t_true2=True
# t_error=0


# t_self1=0
# t_self2=0
# t_score=0
# # t_traversal=0

# # 0未知，1己方，-1对方,标号表示的是访问的象限及其对面
# # True表示识别到假宝藏，False表示未识别到

# # from main import temp

# maze_direct = []
# predecessors=[]
# distances = []

# dfs_p=[]
# dfs_distances=float('inf')
# dfs_index=0
# dfs_visited=[]
# dfs_last=t_last
# inf_shortest=True
# inf_parameter=0
# inf_p=[]
# inf_distances=[]
# inf_index=[]
# inf_visited=[]
# inf_wall=[]

# maze = np.loadtxt("maze.txt", dtype=int)
# m = len(maze)
# n = len(maze[0])
# maze[14][10]=1
# maze[4][8]=1
# maze_visited = maze.copy()
# t_pos=[
#     (0, 2), (0, 4), (0, 5), (0, 9), (1, 3), (1, 6), (1, 7), (1, 8), (2, 2), (2, 5), (2, 8), (2, 9), (3, 0), (3, 3), (3, 4), (3, 8), (3, 9), (4, 7), (4, 8), (9, 7), (9, 5), (9, 4), (9, 0), (8, 6), (8, 3), (8, 2), (8, 1), (7, 7), (7, 4), (7, 1), (7, 0), (6, 9), (6, 6), (6, 5), (6, 1), (6, 0), (5, 2), (5, 1)]
# # t_wall_pos=[[0, 5], [1, 3], [1, 6], [1, 7], [2, 6], [2, 8], [3, 3], [3, 8], [3, 9], [9, 4], [8, 6], [8, 3], [8, 2], [7, 3], [7, 1], [6, 6], [6, 1], [6, 0]]
# t_wall_maze=[[18, 10], [16, 6], [16, 12], [16, 14], [14, 12], [14, 16], [12, 6], [12, 16], [12, 18], [0, 8], [2, 12], [2, 6], [2, 4], [4, 6], [4, 2], [6, 12], [6, 2], [6, 0]]
# corner=[
#     (0,2),
#     (0,4),

#     (0,9),
#     (1,8),
#     (2,2),
#     (2,5),

#     (2,9),
#     (3,0),

#     (3,4),

#     (4,7),
#     (4,8),


# ]
# cross = [
#     (10, 0),
#     (0, 2),
#     (4, 4),
#     (12, 4),
#     (10, 4),
#     (0, 6),
#     (6, 6),
#     (8, 6),
#     (10, 6),
#     (14, 6),
#     (2, 8),
#     (2, 10),
#     (16, 8),
#     (16, 10),
#     (4, 12),
#     (8, 12),
#     (10, 12),
#     (12, 12),
#     (18, 12),
#     (8, 14),
#     (6, 14),
#     (14, 14),
#     (18, 16),
#     (8, 18),
# ]


# def trd_grd_inf():
#     pass
# def trd_grd_inf_dfs():
#     global t_next
#     if t_first1==False and t_first2==False and t_fake==True:

#         t_next=dfs_ccl(t_visited)


#     elif t_first1==True and t_first2==True:
#         t_next=greedy(-1)

#     else:
#         t_next=inf_ccl()
#     return t_next
# def trd_grd_dfs():
#     pass
# def trd_inf_dfs():

#     global t_next
#     if t_first1==False and t_first2==False and t_fake==True:

#         t_next=dfs_ccl(t_visited)
#     else:
#         t_next=inf_ccl()
#     return t_next
# def trd_dfs():
#     global t_next
#     t_next=dfs_ccl(t_visited)

#     return t_next

# def inf_traversal():
#     fake=1
#     if t_fake==True:
#         fake=0
#     return 5-abs(t_self1)-abs(t_self2)-t_score+fake
# def inf_error():
#     pass

# def inf_traversal():
#     fake=1
#     if t_fake==True:
#         fake=0
#     return 5-abs(t_self1)-abs(t_self2)-t_score+fake
# def inf_error():
#     pass
# def inf(visited):
#     visited_list=[]
#     # if t_last==3:
#     #     print("!")
#     # visited_shortest=[]

#     for inf in range(4):
#         for fake in range(1,5):
#             #  inf 从0到3，表示全错，1对2错，1对2错，全对，
#             # 二进制转化后可以发现规律

#             visited_temp=copy.deepcopy(visited)

#             # 表示的是否为该象限中第一个访问点
#             true1=True
#             true2=True
#             if inf==0:
#                 if t_self1==1 or t_self2==1:
#                     continue
#                 true1=False
#                 true2=False
#             elif inf ==1:
#                 if t_self1==1 or t_self2==-1:
#                     continue
#                 true1=False
#             elif inf==2:
#                 if t_self1==-1 or t_self2==1:
#                     continue
#                 true2=False
#             elif inf==2:
#                 if t_self1==-1 or t_self2==-1:
#                     continue
#             if t_fake==True:
#                 pass
#             else:
#                 visited_temp[2*fake-2]=False
#                 visited_temp[2*fake-1]=False
#             if t_first1==True:
#                 if true1==True:
#                     visited_temp[1]=False
#                     visited_temp[7]=False
#                 else:
#                     visited_temp[0]=False
#                     visited_temp[6]=False
#             if t_first2==True:
#                 if true2==True:
#                     visited_temp[3]=False
#                     visited_temp[4]=False
#                 else:
#                     visited_temp[2]=False
#                     visited_temp[5]=False


#             count = visited_temp.count(True)+t_score
#             element=visited_temp in visited_list
#             if element==False and count==3:

#                 visited_list.append(visited_temp)
#     return visited_list
# def greedy(pos):
#     short_dis=float('inf')

#     for i in range(t_len):
#         if t_visited[i]==False:
#             continue

#         if t_last==-1:
#             dis=t_distances[i][i]
#         else:
#             dis=t_distances[pos][i]
#         if dis<short_dis:
#             short_dis = dis
#             pos=i

#     return pos


# def dfs_ccl(item):
#     global dfs_distances
#     global dfs_index
#     global dfs_visited
#     global dfs_p
#     dfs_visited=copy.deepcopy(item)
#     dfs_distances=float('inf')
#     dfs_p=[]
#     dfs_index=-1
#     dfs(0)

#     return dfs_index


# def inf_ccl():
# global inf_visited
# global inf_distances
# global inf_index
# global dfs_distances
# global dfs_index
# global dfs_p
# global dfs_visited
# inf_index=[0 for _ in range(t_len)]
# inf_distances=[0 for _ in range(t_len)]


# inf_visited=inf(t_visited)
# for item in inf_visited:
#     dfs_index=dfs_ccl(item)
#     if dfs_index>0:

#         inf_index[dfs_index]+=1
#         inf_distances[dfs_index]+=dfs_distances
# p_count=-1
# pos=-1

# for i in range(t_len):
#     if inf_index[i]>p_count:
#         p_count=inf_index[i]
#         pos=i
#     elif inf_index[i]==p_count:
#         if inf_distances[i]<inf_distances[pos]:
#             pos=i
# return pos
#

# def t_judge(visited):
#     global t_self1,t_self2
#     global t_fake,t_score,t_error
#     global t_first1,t_first2
#     global t_last,t_next
#     global dfs_last
#     global maze
#     global t_wall
#     t_last=copy.deepcopy(t_next)
#     dfs_last=t_last
#     visited[t_last]=False
#     if picture==1:
#         t_score+=1
#         x,y=list_val(t_last)
#         maze[x][y]=0
#         element=[x,y]in t_wall
#         if element==True:
#             t_wall.remove([x,y])
#     elif abs(picture)==2:
#         t_fake=True
#         visited[t_len-1-t_last] = False
#     elif picture==0:
#         t_error+=1

#     if t_score == 3:
#         for i in range(len(visited)):
#             visited[i] = False
#     elif picture  > 0:
#         # 我方宝藏
#         visited[t_len-1-t_last] = False
#         for i in range(t_len):
#             if mat_list[i]==mat_list[t_last]:
#                 visited[i] = False
#         if mat_list[t_last]==1 or mat_list[t_last]==4:
#             t_self1=1
#             t_first1=False
#         else:
#             t_self2=1
#             t_first2=False
#     elif picture  < 0:
#         for i in range(t_len):
#             if mat_list[i]+mat_list[t_last]==5 and i!=t_len-1-t_last:
#                 visited[i] = False
#         if mat_list[t_last]==1 or mat_list[t_last]==4:
#             t_self1=-1
#             t_first1=False
#         else:
#             t_self2=-1
#             t_first2=False


#     return visited


# def t_traversal():
#     pass

# def load_float_list_from_file(filename):
#     with open(filename, 'r') as file:
#         content = file.read()  # 读取文本文件内容
#         my_list = ast.literal_eval(content)  # 使用ast模块解析文本中的列表格式
#     return my_list

# def MatrixTransform(item):
#     for i in range(len(item)):
#         item[i][0] = 18 - (2 * item[i][0])
#         item[i][1] = 2 * item[i][1]
#     return item
# def ListTransform(item):
#     item[0] = 18 - (2 * item[0])
#     item[1] = 2 * item[1]
#     return item


# def crush(ind):
#     global maze
#     global t_wall_maze
#     x=t_tranform[ind][0]
#     y=t_tranform[ind][1]
#     element=[x,y] in t_wall_maze
#     if element ==True:
#         t_wall.remove([x,y])

#         maze[x][y]=0
#     return t_lastx,t_lasty
# def list_val(count):
#     if count==-1:
#         return 18,0
#     else:
#         return t_tranform[count][0],t_tranform[count][1]

# def grd_ccl_time():
#     count=copy.deepcopy(t_last)

#     x0,y0=list_val(count)
#     element=[x0,y0] in t_wall
#     if  element==True:
#         x0,y0=crush(count)
#     ans=grd_time(count)
#     x,y=list_val(ans)
#     dij_time((x0,y0),(x,y))
#     dir=DirectTransform((x0,y0),(x,y))
#     return dir

# def dir_not_crush(i):
#     count=copy.deepcopy(i)
#     x,y=list_val(count)
# def trd_grd_inf():
#     pass
# def trd_grd_inf_dfs():
#     global t_next
#     if t_first1==False and t_first2==False and t_fake==True:

#         t_next=dfs_ccl(t_visited)


#     elif t_first1==True and t_first2==True:
#         t_next=greedy(-1)

#     else:
#         t_next=inf_ccl()
#     return t_next
# def trd_grd_dfs_1():
#     global t_next


#     if t_first1==True and t_first2==True:
#         t_next=greedy(-1)

#     else:
#         t_next=dfs_ccl(t_visited)
#     return t_next
# def trd_inf_dfs():

#     global t_next
#     if t_first1==False and t_first2==False and t_fake==True:

#         t_next=dfs_ccl(t_visited)
#     else:
#         t_next=inf_ccl()
#     return t_next


# dir=[[[0,1],[5,6]],[[1,5],[9,6]],]
# print(dir)
# dir[0][0]=[1,0]
# print(dir)
