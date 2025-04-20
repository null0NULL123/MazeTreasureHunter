from typing import List, Tuple, Optional, Protocol, Any
import numpy as np

class PathStrategy(Protocol):
    """路径查找策略接口"""
    
    def find_path(self, maze: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        查找从起点到终点的路径
        
        Args:
            maze: 迷宫数组
            start: 起点坐标 (x, y)
            end: 终点坐标 (x, y)
            
        Returns:
            包含路径坐标的列表，如果没有找到路径则返回空列表
        """
        pass
    
    def get_distances(self, maze: np.ndarray, start: Tuple[int, int]) -> List[List[float]]:
        """
        计算从起点到所有其他点的距离
        
        Args:
            maze: 迷宫数组
            start: 起点坐标 (x, y)
            
        Returns:
            距离矩阵
        """
        pass


class TreasureStrategy(Protocol):
    """宝藏收集策略接口"""
    
    def find_optimal_path(self, distances: List[List[float]], start_id: int, treasures_to_visit: List[int]) -> List[int]:
        """
        找出收集宝藏的最优路径
        
        Args:
            distances: 宝藏之间的距离矩阵
            start_id: 起始宝藏ID
            treasures_to_visit: 需要访问的宝藏ID列表
            
        Returns:
            收集宝藏的顺序（ID列表）
        """
        pass


class InferenceStrategy(Protocol):
    """推理策略接口"""
    
    def infer_fake_treasures(self, treasures: List[Any], discovered_info: List[int]) -> List[int]:
        """
        推断假宝藏
        
        Args:
            treasures: 宝藏列表
            discovered_info: 已发现的宝藏信息
            
        Returns:
            推断为假宝藏的ID列表
        """
        pass 