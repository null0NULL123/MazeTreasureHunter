from typing import List, Tuple, Dict, Set, Optional
import random
import copy
from config.maze_config import MazeConfig
from config.settings import GameSettings
from utils.matrix_utils import MatrixTransform
class Treasure:
    """宝藏类，表示单个宝藏"""
    
    def __init__(self, position: Tuple[int, int], treasure_id: int, is_fake: bool = False) -> None:
        """
        初始化宝藏
        
        Args:
            position: 宝藏位置坐标 (x, y)
            treasure_id: 宝藏ID
            is_fake: 是否为假宝藏
        """
        self.position: Tuple[int, int] = position
        self.treasure_id: int = treasure_id
        self.is_fake: bool = is_fake
        self.is_discovered: bool = False
        self.is_collected: bool = False
        
        # 宝藏状态：0=未识别，1=真宝藏，-1=假宝藏，2=可能是真的，-2=可能是假的
        self.status: int = 0
    
    def discover(self) -> None:
        """发现宝藏"""
        self.is_discovered = True
    
    def collect(self) -> None:
        """收集宝藏"""
        self.is_collected = True
    
    def set_status(self, status: int) -> None:
        """设置宝藏状态"""
        self.status = status
    
    def __str__(self) -> str:
        return f"Treasure(id={self.treasure_id}, position={self.position}, fake={self.is_fake}, status={self.status})"


class TreasureModel:
    """宝藏数据模型，管理所有宝藏及其状态"""
    
    def __init__(self, config: MazeConfig, settings: GameSettings) -> None:
        """
        初始化宝藏模型
        
        Args:
            config: 迷宫配置
            settings: 游戏设置
        """
        self.config: MazeConfig = config
        self.settings: GameSettings = settings
        self.treasures: List[Treasure] = []
        self.treasure_set_index: int = 0
        self.distances: List[List[float]] = []  # 保存宝藏之间的距离
        
        # 初始化
        self._initialize_treasures()
    
    def _initialize_treasures(self) -> None:
        """初始化宝藏"""
        # 随机选择一个宝藏集合
        self.treasure_set_index = random.randint(0, len(self.config.treasure_sets) - 1)
        treasure_set = self.config.treasure_sets[self.treasure_set_index]
        treasure_set = MatrixTransform.transform_coordinates(treasure_set)

        # 清空之前的宝藏
        self.treasures = []
        
        # 创建宝藏对象
        for i, position in enumerate(treasure_set):
            # 随机决定是否为假宝藏 (每个集合中一半是假的)
            is_fake = i >= len(treasure_set) // 2
            treasure = Treasure(position, i, is_fake)
            self.treasures.append(treasure)
        
        # 打乱宝藏顺序
        random.shuffle(self.treasures)
        
        # 重新分配ID
        for i, treasure in enumerate(self.treasures):
            treasure.treasure_id = i
    
    def get_treasure_by_id(self, treasure_id: int) -> Optional[Treasure]:
        """根据ID获取宝藏"""
        for treasure in self.treasures:
            if treasure.treasure_id == treasure_id:
                return treasure
        return None
    
    def get_treasure_by_position(self, position: Tuple[int, int]) -> Optional[Treasure]:
        """根据位置获取宝藏"""
        for treasure in self.treasures:
            if treasure.position == position:
                return treasure
        return None
    
    def get_all_treasures(self) -> List[Treasure]:
        """获取所有宝藏"""
        return copy.deepcopy(self.treasures)
    
    def get_discovered_treasures(self) -> List[Treasure]:
        """获取已发现的宝藏"""
        return [t for t in self.treasures if t.is_discovered]
    
    def get_undiscovered_treasures(self) -> List[Treasure]:
        """获取未发现的宝藏"""
        return [t for t in self.treasures if not t.is_discovered]
    
    def get_collected_treasures(self) -> List[Treasure]:
        """获取已收集的宝藏"""
        return [t for t in self.treasures if t.is_collected]
    
    def get_real_treasures(self) -> List[Treasure]:
        """获取真宝藏"""
        return [t for t in self.treasures if not t.is_fake]
    
    def get_fake_treasures(self) -> List[Treasure]:
        """获取假宝藏"""
        return [t for t in self.treasures if t.is_fake]
    
    def discover_treasure(self, treasure_id: int) -> None:
        """发现宝藏"""
        treasure = self.get_treasure_by_id(treasure_id)
        if treasure:
            treasure.discover()
    
    def collect_treasure(self, treasure_id: int) -> None:
        """收集宝藏"""
        treasure = self.get_treasure_by_id(treasure_id)
        if treasure:
            treasure.collect()
    
    def set_treasure_status(self, treasure_id: int, status: int) -> None:
        """设置宝藏状态"""
        treasure = self.get_treasure_by_id(treasure_id)
        if treasure:
            treasure.set_status(status)
    
    def set_distances(self, distances: List[List[float]]) -> None:
        """设置宝藏之间的距离矩阵"""
        self.distances = distances
    
    def get_distance(self, from_id: int, to_id: int) -> float:
        """获取两个宝藏之间的距离"""
        if not self.distances or from_id >= len(self.distances) or to_id >= len(self.distances[0]):
            return float('inf')
        return self.distances[from_id][to_id]
    
    def reset(self) -> None:
        """重置宝藏模型"""
        self._initialize_treasures()
        self.distances = [] 