from typing import List, Tuple, Dict, Set, Optional, Any
import copy
import random
from models.treasure_model import Treasure

class InferenceEngine:
    """推理引擎，用于推断宝藏的真假"""
    
    def __init__(self) -> None:
        """初始化推理引擎"""
        self.fake_probability: Dict[int, float] = {}  # 宝藏是假的概率
        self.real_probability: Dict[int, float] = {}  # 宝藏是真的概率
    
    def infer_fake_treasures(self, treasures: List[Treasure], discovered_info: List[int]) -> List[int]:
        """
        推断假宝藏
        
        Args:
            treasures: 宝藏列表
            discovered_info: 已发现的宝藏信息（状态列表）
            
        Returns:
            推断为假宝藏的ID列表
        """
        # 初始化概率
        self._initialize_probabilities(treasures)
        
        # 更新概率
        self._update_probabilities(treasures, discovered_info)
        
        # 返回概率超过阈值的假宝藏ID
        fake_treasures = []
        for treasure in treasures:
            if treasure.treasure_id in self.fake_probability and self.fake_probability[treasure.treasure_id] > 0.7:
                fake_treasures.append(treasure.treasure_id)
        
        return fake_treasures
    
    def _initialize_probabilities(self, treasures: List[Treasure]) -> None:
        """初始化宝藏真假概率"""
        n = len(treasures)
        for treasure in treasures:
            # 初始时，假设一半宝藏是真的，一半是假的
            self.fake_probability[treasure.treasure_id] = 0.5
            self.real_probability[treasure.treasure_id] = 0.5
    
    def _update_probabilities(self, treasures: List[Treasure], discovered_info: List[int]) -> None:
        """
        根据已发现的信息更新概率
        
        Args:
            treasures: 宝藏列表
            discovered_info: 已发现的宝藏状态（与宝藏列表对应）
                状态说明：
                0: 未知
                1: 确认为真
                -1: 确认为假
                2: 可能为真
                -2: 可能为假
        """
        # 总宝藏数和真宝藏数
        n = len(treasures)
        real_count = n // 2
        fake_count = n - real_count
        
        # 已确认的真宝藏和假宝藏数量
        confirmed_real = discovered_info.count(1)
        confirmed_fake = discovered_info.count(-1)
        
        # 如果所有真宝藏或假宝藏都已确认，则剩余的宝藏状态确定
        if confirmed_real == real_count:
            for i, treasure in enumerate(treasures):
                if discovered_info[i] == 0:
                    self.fake_probability[treasure.treasure_id] = 1.0
                    self.real_probability[treasure.treasure_id] = 0.0
        elif confirmed_fake == fake_count:
            for i, treasure in enumerate(treasures):
                if discovered_info[i] == 0:
                    self.fake_probability[treasure.treasure_id] = 0.0
                    self.real_probability[treasure.treasure_id] = 1.0
        else:
            # 基于已知信息更新概率
            for i, treasure in enumerate(treasures):
                if discovered_info[i] == 1:  # 确认为真
                    self.fake_probability[treasure.treasure_id] = 0.0
                    self.real_probability[treasure.treasure_id] = 1.0
                elif discovered_info[i] == -1:  # 确认为假
                    self.fake_probability[treasure.treasure_id] = 1.0
                    self.real_probability[treasure.treasure_id] = 0.0
                elif discovered_info[i] == 2:  # 可能为真
                    self.fake_probability[treasure.treasure_id] *= 0.3
                    self.real_probability[treasure.treasure_id] = 1 - self.fake_probability[treasure.treasure_id]
                elif discovered_info[i] == -2:  # 可能为假
                    self.fake_probability[treasure.treasure_id] *= 1.5
                    if self.fake_probability[treasure.treasure_id] > 1.0:
                        self.fake_probability[treasure.treasure_id] = 1.0
                    self.real_probability[treasure.treasure_id] = 1 - self.fake_probability[treasure.treasure_id]
    
    def get_treasure_probabilities(self) -> Dict[int, Dict[str, float]]:
        """获取宝藏真假概率"""
        probabilities = {}
        for treasure_id in self.fake_probability:
            probabilities[treasure_id] = {
                "fake": self.fake_probability.get(treasure_id, 0.5),
                "real": self.real_probability.get(treasure_id, 0.5)
            }
        return probabilities


class GreedyTreasureStrategy:
    """贪心宝藏收集策略"""
    
    def find_optimal_path(self, distances: List[List[float]], start_id: int, treasures_to_visit: List[int]) -> List[int]:
        """
        使用贪心算法找出收集宝藏的最优路径
        
        Args:
            distances: 宝藏之间的距离矩阵
            start_id: 起始宝藏ID
            treasures_to_visit: 需要访问的宝藏ID列表
            
        Returns:
            收集宝藏的顺序（ID列表）
        """
        if not treasures_to_visit:
            return []
        
        current_id = start_id
        path = [current_id]
        remaining = treasures_to_visit.copy()
        
        while remaining:
            # 找出距离当前位置最近的宝藏
            min_dist = float('inf')
            next_id = -1
            
            for treasure_id in remaining:
                if current_id < len(distances) and treasure_id < len(distances[0]):
                    dist = distances[current_id][treasure_id]
                    if dist < min_dist:
                        min_dist = dist
                        next_id = treasure_id
            
            if next_id == -1:
                break
            
            # 移动到下一个宝藏
            current_id = next_id
            path.append(current_id)
            remaining.remove(current_id)
        
        return path


class DFSTreasureStrategy:
    """DFS宝藏收集策略，尝试找出最优路径"""
    
    def __init__(self) -> None:
        """初始化DFS策略"""
        self.best_path = []
        self.best_distance = float('inf')
    
    def find_optimal_path(self, distances: List[List[float]], start_id: int, treasures_to_visit: List[int]) -> List[int]:
        """
        使用DFS算法找出收集宝藏的最优路径
        
        Args:
            distances: 宝藏之间的距离矩阵
            start_id: 起始宝藏ID
            treasures_to_visit: 需要访问的宝藏ID列表
            
        Returns:
            收集宝藏的顺序（ID列表）
        """
        if not treasures_to_visit:
            return []
        
        self.best_path = []
        self.best_distance = float('inf')
        
        # 创建已访问标记
        visited = [False] * len(treasures_to_visit)
        
        def dfs(current_id: int, path: List[int], total_dist: float) -> None:
            # 如果所有宝藏都已访问
            if all(visited):
                if total_dist < self.best_distance:
                    self.best_distance = total_dist
                    self.best_path = path.copy()
                return
            
            # 尝试访问每个未访问的宝藏
            for i, treasure_id in enumerate(treasures_to_visit):
                if not visited[i]:
                    # 计算移动到下一个宝藏的距离
                    next_dist = 0
                    if current_id < len(distances) and treasure_id < len(distances[0]):
                        next_dist = distances[current_id][treasure_id]
                    
                    # 标记为已访问
                    visited[i] = True
                    
                    # 递归探索
                    dfs(treasure_id, path + [treasure_id], total_dist + next_dist)
                    
                    # 回溯
                    visited[i] = False
        
        # 从起点开始DFS
        dfs(start_id, [start_id], 0)
        
        return self.best_path 