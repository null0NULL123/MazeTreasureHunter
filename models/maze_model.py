import numpy as np
from typing import List, Tuple, Optional, Set
import copy
from config.maze_config import MazeConfig

class MazeModel:
    """迷宫数据模型，管理迷宫的状态和操作"""
    
    def __init__(self, config: MazeConfig) -> None:
        """
        初始化迷宫模型
        
        Args:
            config: 迷宫配置对象
        """
        self.config = config
        self.maze: np.ndarray = self._load_maze()
        self.height: int = len(self.maze)
        self.width: int = len(self.maze[0])
        self.visited: np.ndarray = np.zeros_like(self.maze)
        self.path: List[Tuple[int, int]] = []
        
    def _load_maze(self) -> np.ndarray:
        """从文件加载迷宫数据"""
        try:
            return np.loadtxt(self.config.maze_file, dtype=int)
        except Exception as e:
            # 如果加载失败，返回一个默认的小迷宫
            print(f"加载迷宫文件失败: {e}")
            return np.zeros((10, 10), dtype=int)
    
    def is_wall(self, x: int, y: int) -> bool:
        """检查指定位置是否为墙"""
        if not self.is_valid_position(x, y):
            return True
        return self.maze[x][y] == 1
    
    def is_valid_position(self, x: int, y: int) -> bool:
        """检查位置是否在迷宫范围内"""
        return 0 <= x < self.height and 0 <= y < self.width
    
    def get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """获取邻近可通行位置"""
        neighbors: List[Tuple[int, int]] = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 右、下、左、上
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_valid_position(nx, ny) and not self.is_wall(nx, ny):
                neighbors.append((nx, ny))
        
        return neighbors
    
    def mark_visited(self, x: int, y: int) -> None:
        """标记位置为已访问"""
        if self.is_valid_position(x, y):
            self.visited[x][y] = 1
    
    def is_visited(self, x: int, y: int) -> bool:
        """检查位置是否已访问"""
        if not self.is_valid_position(x, y):
            return False
        return self.visited[x][y] == 1
    
    def reset_visited(self) -> None:
        """重置所有访问状态"""
        self.visited = np.zeros_like(self.maze)
    
    def add_to_path(self, x: int, y: int) -> None:
        """添加位置到路径"""
        self.path.append((x, y))
    
    def get_path(self) -> List[Tuple[int, int]]:
        """获取当前路径"""
        return copy.deepcopy(self.path)
    
    def reset_path(self) -> None:
        """重置路径"""
        self.path = []
    
    def set_wall(self, x: int, y: int, is_wall: bool = True) -> None:
        """设置或清除墙壁"""
        if self.is_valid_position(x, y):
            self.maze[x][y] = 1 if is_wall else 0
    
    def get_maze_copy(self) -> np.ndarray:
        """获取迷宫数据的副本"""
        return copy.deepcopy(self.maze) 