import heapq
import numpy as np
from typing import List, Tuple, Dict, Set, Optional
import copy
from config.settings import GameSettings

class DijkstraStrategy:
    """Dijkstra路径查找策略"""
    
    def __init__(self, settings: GameSettings) -> None:
        """
        初始化Dijkstra策略
        
        Args:
            settings: 游戏设置
        """
        self.settings = settings
        # 移动代价参数
        self.direct_parameter = settings.direct_parameter
        self.cross_parameter = settings.cross_parameter
        self.turn_parameter = settings.turn_parameter
        self.capture_parameter = settings.capture_parameter
    
    def find_path(self, maze: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        使用Dijkstra算法查找从起点到终点的最短路径
        
        Args:
            maze: 迷宫数组
            start: 起点坐标 (x, y)
            end: 终点坐标 (x, y)
            
        Returns:
            包含路径坐标的列表，如果没有找到路径则返回空列表
        """
        # 保存原始迷宫，防止临时修改影响后续操作
        original_maze = copy.deepcopy(maze)
        m, n = len(maze), len(maze[0])
        
        # 确保起点和终点在迷宫内且不是墙
        startx, starty = start
        endx, endy = end
        
        # 临时清除起点和终点的墙（如果有的话）
        is_start_wall = maze[startx][starty] == 1
        is_end_wall = maze[endx][endy] == 1
        maze[startx][starty] = 0
        maze[endx][endy] = 0
        
        # 初始化距离矩阵和前驱矩阵
        distances = [[float("inf")] * n for _ in range(m)]
        predecessors = [[None] * n for _ in range(m)]
        distances[startx][starty] = 0
        
        # 初始化已访问集合和优先队列
        visited = set()
        heap = [(0, start)]
        
        while heap:
            dist, (x, y) = heapq.heappop(heap)
            
            if (x, y) in visited:
                continue
            
            visited.add((x, y))
            
            if (x, y) == end:
                break
            
            # 探索四个方向
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                
                # 检查边界和墙
                if nx < 0 or nx >= m or ny < 0 or ny >= n or (nx, ny) in visited or maze[nx][ny] == 1:
                    continue
                
                # 计算新距离
                # 根据运动方向和前一个位置决定代价
                if predecessors[x][y] is not None:
                    px, py = predecessors[x][y]
                    if x - px == dx and y - py == dy:
                        # 直线运动
                        new_dist = dist + self.direct_parameter
                    elif x - px == -dx and y - py == -dy:
                        # 掉头
                        new_dist = dist + self.turn_parameter
                    else:
                        # 转弯
                        new_dist = dist + self.cross_parameter
                else:
                    # 第一步移动
                    new_dist = dist + 1
                
                # 更新距离和前驱
                if new_dist < distances[nx][ny]:
                    distances[nx][ny] = new_dist
                    predecessors[nx][ny] = (x, y)
                    heapq.heappush(heap, (new_dist, (nx, ny)))
        
        # 恢复原始迷宫
        if is_start_wall:
            maze[startx][starty] = 1
        if is_end_wall:
            maze[endx][endy] = 1
        
        # 如果没有找到路径
        if predecessors[endx][endy] is None:
            return []
        
        # 从终点回溯到起点，构建路径
        path = []
        current = end
        while current != start:
            path.append(current)
            x, y = current
            current = predecessors[x][y]
        
        path.append(start)
        path.reverse()
        
        return path
    
    def get_distances(self, maze: np.ndarray, start: Tuple[int, int]) -> List[List[float]]:
        """
        计算从起点到所有其他点的距离
        
        Args:
            maze: 迷宫数组
            start: 起点坐标 (x, y)
            
        Returns:
            距离矩阵
        """
        m, n = len(maze), len(maze[0])
        startx, starty = start
        
        # 初始化距离矩阵和前驱矩阵
        distances = [[float("inf")] * n for _ in range(m)]
        predecessors = [[None] * n for _ in range(m)]
        distances[startx][starty] = 0
        
        # 初始化已访问集合和优先队列
        visited = set()
        heap = [(0, start)]
        
        while heap:
            dist, (x, y) = heapq.heappop(heap)
            
            if (x, y) in visited:
                continue
            
            visited.add((x, y))
            
            # 探索四个方向
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                
                # 检查边界和墙
                if nx < 0 or nx >= m or ny < 0 or ny >= n or (nx, ny) in visited or maze[nx][ny] == 1:
                    continue
                
                # 计算新距离
                if predecessors[x][y] is not None:
                    px, py = predecessors[x][y]
                    if x - px == dx and y - py == dy:
                        # 直线运动
                        new_dist = dist + self.direct_parameter
                    elif x - px == -dx and y - py == -dy:
                        # 掉头
                        new_dist = dist + self.turn_parameter
                    else:
                        # 转弯
                        new_dist = dist + self.cross_parameter
                else:
                    # 第一步移动
                    new_dist = dist + 1
                
                # 更新距离和前驱
                if new_dist < distances[nx][ny]:
                    distances[nx][ny] = new_dist
                    predecessors[nx][ny] = (x, y)
                    heapq.heappush(heap, (new_dist, (nx, ny)))
        
        return distances


class DFSStrategy:
    """深度优先搜索路径查找策略"""
    
    def __init__(self) -> None:
        """初始化DFS策略"""
        self.path = []
    
    def find_path(self, maze: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        使用DFS算法查找从起点到终点的路径
        
        Args:
            maze: 迷宫数组
            start: 起点坐标 (x, y)
            end: 终点坐标 (x, y)
            
        Returns:
            包含路径坐标的列表，如果没有找到路径则返回空列表
        """
        m, n = len(maze), len(maze[0])
        visited = set()
        self.path = []
        
        def dfs(x: int, y: int) -> bool:
            # 如果到达终点
            if (x, y) == end:
                self.path.append((x, y))
                return True
            
            # 标记为已访问
            visited.add((x, y))
            
            # 尝试四个方向
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                
                # 检查边界、墙和已访问
                if nx < 0 or nx >= m or ny < 0 or ny >= n or (nx, ny) in visited or maze[nx][ny] == 1:
                    continue
                
                # 递归探索
                if dfs(nx, ny):
                    self.path.append((x, y))
                    return True
            
            return False
        
        # 从起点开始DFS
        dfs(*start)
        
        # 反转路径顺序（因为是从终点回溯到起点）
        self.path.reverse()
        
        return self.path
    
    def get_distances(self, maze: np.ndarray, start: Tuple[int, int]) -> List[List[float]]:
        """
        使用DFS计算从起点到所有其他点的距离
        
        Args:
            maze: 迷宫数组
            start: 起点坐标 (x, y)
            
        Returns:
            距离矩阵（在DFS中不太适用，返回近似值）
        """
        m, n = len(maze), len(maze[0])
        distances = [[float("inf")] * n for _ in range(m)]
        visited = set()
        
        def dfs(x: int, y: int, depth: int) -> None:
            # 更新距离
            distances[x][y] = min(distances[x][y], depth)
            
            # 标记为已访问
            visited.add((x, y))
            
            # 尝试四个方向
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                
                # 检查边界、墙和已访问
                if nx < 0 or nx >= m or ny < 0 or ny >= n or (nx, ny) in visited or maze[nx][ny] == 1:
                    continue
                
                # 递归探索
                dfs(nx, ny, depth + 1)
            
            # 回溯时取消访问标记，允许其他路径重新访问
            visited.remove((x, y))
        
        # 从起点开始DFS
        startx, starty = start
        distances[startx][starty] = 0
        dfs(startx, starty, 0)
        
        return distances


class PathFinder:
    """路径查找器，封装和协调不同的路径查找策略"""
    
    def __init__(self, settings: GameSettings) -> None:
        """
        初始化路径查找器
        
        Args:
            settings: 游戏设置
        """
        self.settings = settings
        self.strategy = DijkstraStrategy(settings)  # 默认使用Dijkstra策略
    
    def set_strategy(self, strategy_name: str) -> None:
        """
        设置路径查找策略
        
        Args:
            strategy_name: 策略名称（"dijkstra"或"dfs"）
        """
        if strategy_name.lower() == "dijkstra":
            self.strategy = DijkstraStrategy(self.settings)
        elif strategy_name.lower() == "dfs":
            self.strategy = DFSStrategy()
    
    def find_path(self, maze: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        使用当前策略查找路径
        
        Args:
            maze: 迷宫数组
            start: 起点坐标 (x, y)
            end: 终点坐标 (x, y)
            
        Returns:
            路径坐标列表
        """
        return self.strategy.find_path(maze, start, end)
    
    def get_distances(self, maze: np.ndarray, start: Tuple[int, int]) -> List[List[float]]:
        """
        使用当前策略计算距离
        
        Args:
            maze: 迷宫数组
            start: 起点坐标 (x, y)
            
        Returns:
            距离矩阵
        """
        return self.strategy.get_distances(maze, start) 