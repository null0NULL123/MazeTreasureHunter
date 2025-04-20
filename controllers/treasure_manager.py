from typing import List, Tuple, Dict, Optional
import numpy as np

from models.treasure_model import TreasureModel, Treasure
from models.maze_model import MazeModel
from algorithms.path_finder import PathFinder
from algorithms.inference import InferenceEngine, GreedyTreasureStrategy, DFSTreasureStrategy

class TreasureManager:
    """宝藏管理器，处理宝藏相关的逻辑"""
    
    def __init__(self, treasure_model: TreasureModel, maze_model: MazeModel, path_finder: PathFinder) -> None:
        """
        初始化宝藏管理器
        
        Args:
            treasure_model: 宝藏模型
            maze_model: 迷宫模型
            path_finder: 路径查找器
        """
        self.treasure_model = treasure_model
        self.maze_model = maze_model
        self.path_finder = path_finder
        self.inference_engine = InferenceEngine()
        self.treasure_strategy = GreedyTreasureStrategy()  # 默认使用贪心策略
        
        # 状态记录
        self.current_treasure_id: int = -1
        self.next_treasure_id: int = -1
        self.treasure_path: List[int] = []
        self.discovered_info: List[int] = []  # 与宝藏列表对应的状态信息
    
    def set_treasure_strategy(self, strategy_name: str) -> None:
        """
        设置宝藏收集策略
        
        Args:
            strategy_name: 策略名称（"greedy"或"dfs"）
        """
        if strategy_name.lower() == "greedy":
            self.treasure_strategy = GreedyTreasureStrategy()
        elif strategy_name.lower() == "dfs":
            self.treasure_strategy = DFSTreasureStrategy()
    
    def discover_treasure(self, treasure_id: int, status: int) -> None:
        """
        发现宝藏
        
        Args:
            treasure_id: 宝藏ID
            status: 宝藏状态（0=未知，1=真，-1=假，2=可能为真，-2=可能为假）
        """
        # 更新宝藏模型
        self.treasure_model.discover_treasure(treasure_id)
        self.treasure_model.set_treasure_status(treasure_id, status)
        
        # 更新状态信息
        self._update_discovered_info()
    
    def collect_treasure(self, treasure_id: int) -> None:
        """
        收集宝藏
        
        Args:
            treasure_id: 宝藏ID
        """
        self.treasure_model.collect_treasure(treasure_id)
        self.current_treasure_id = treasure_id
    
    def _update_discovered_info(self) -> None:
        """更新已发现的宝藏信息"""
        treasures = self.treasure_model.get_all_treasures()
        self.discovered_info = [treasure.status for treasure in treasures]
    
    def calculate_treasure_distances(self) -> None:
        """计算宝藏之间的距离"""
        treasures = self.treasure_model.get_all_treasures()
        n = len(treasures)
        
        # 初始化距离矩阵
        distances = [[float('inf')] * n for _ in range(n)]
        
        # 计算每对宝藏之间的距离
        for i in range(n):
            treasure_i = treasures[i]
            for j in range(n):
                if i == j:
                    distances[i][j] = 0
                    continue
                
                treasure_j = treasures[j]
                path = self.path_finder.find_path(
                    self.maze_model.get_maze_copy(), 
                    treasure_i.position, 
                    treasure_j.position
                )
                
                # 距离是路径长度
                if path:
                    distances[i][j] = len(path) - 1
        
        # 更新宝藏模型的距离矩阵
        self.treasure_model.set_distances(distances)
    
    def find_optimal_treasure_path(self, start_id: int = -1) -> List[int]:
        """
        找出最佳宝藏收集路径
        
        Args:
            start_id: 起始宝藏ID，如果为-1则使用当前宝藏ID
            
        Returns:
            收集宝藏的顺序（ID列表）
        """
        # 如果没有指定起始ID，使用当前宝藏ID
        if start_id == -1:
            start_id = self.current_treasure_id
        
        # 如果当前没有宝藏，使用第一个未收集的宝藏
        if start_id == -1:
            treasures = self.treasure_model.get_undiscovered_treasures()
            if treasures:
                start_id = treasures[0].treasure_id
            else:
                return []
        
        # 获取未收集的宝藏ID
        treasures = self.treasure_model.get_undiscovered_treasures()
        treasures_to_visit = [t.treasure_id for t in treasures]
        
        # 如果没有需要访问的宝藏，返回空路径
        if not treasures_to_visit:
            return []
        
        # 如果没有距离矩阵，计算距离
        if not self.treasure_model.distances:
            self.calculate_treasure_distances()
        
        # 使用策略找出最佳路径
        self.treasure_path = self.treasure_strategy.find_optimal_path(
            self.treasure_model.distances,
            start_id,
            treasures_to_visit
        )
        
        return self.treasure_path
    
    def get_next_treasure(self) -> Optional[Treasure]:
        """
        获取下一个要收集的宝藏
        
        Returns:
            下一个宝藏，如果没有则返回None
        """
        # 如果没有路径，计算一个
        if not self.treasure_path:
            self.find_optimal_treasure_path()
        
        # 如果路径为空，返回None
        if not self.treasure_path or len(self.treasure_path) <= 1:
            return None
        
        # 获取路径中的下一个宝藏ID
        self.next_treasure_id = self.treasure_path[1]
        
        # 返回宝藏对象
        return self.treasure_model.get_treasure_by_id(self.next_treasure_id)
    
    def get_path_to_next_treasure(self) -> List[Tuple[int, int]]:
        """
        获取到下一个宝藏的路径
        
        Returns:
            路径坐标列表
        """
        next_treasure = self.get_next_treasure()
        if not next_treasure:
            return []
        
        # 获取当前位置
        current_treasure = self.treasure_model.get_treasure_by_id(self.current_treasure_id)
        start_pos = (0, 0)  # 默认起点
        if current_treasure:
            start_pos = current_treasure.position
        
        # 查找路径
        path = self.path_finder.find_path(
            self.maze_model.get_maze_copy(),
            start_pos,
            next_treasure.position
        )
        
        return path
    
    def infer_fake_treasures(self) -> List[int]:
        """
        推断假宝藏
        
        Returns:
            推断为假宝藏的ID列表
        """
        # 使用推理引擎推断假宝藏
        treasures = self.treasure_model.get_all_treasures()
        if not self.discovered_info:
            self._update_discovered_info()
        
        return self.inference_engine.infer_fake_treasures(treasures, self.discovered_info)
    
    def get_treasure_status(self) -> Dict[int, Dict[str, float]]:
        """
        获取宝藏状态
        
        Returns:
            宝藏状态字典，键为宝藏ID，值为包含fake和real概率的字典
        """
        return self.inference_engine.get_treasure_probabilities()
    
    def reset(self) -> None:
        """重置宝藏管理器状态"""
        self.current_treasure_id = -1
        self.next_treasure_id = -1
        self.treasure_path = []
        self.discovered_info = [] 