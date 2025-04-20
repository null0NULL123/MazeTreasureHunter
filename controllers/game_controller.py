from typing import List, Tuple, Dict, Optional, Any, Callable
import time

from config.settings import GameSettings
from config.maze_config import MazeConfig
from models.maze_model import MazeModel
from models.treasure_model import TreasureModel
from algorithms.path_finder import PathFinder
from controllers.treasure_manager import TreasureManager

class GameController:
    """游戏控制器，协调游戏的各个组件和逻辑"""
    
    def __init__(self) -> None:
        """初始化游戏控制器"""
        # 加载配置
        self.settings = GameSettings()
        self.maze_config = MazeConfig()
        
        # 初始化模型
        self.maze_model = MazeModel(self.maze_config)
        self.treasure_model = TreasureModel(self.maze_config, self.settings)
        
        # 初始化算法组件
        self.path_finder = PathFinder(self.settings)
        
        # 初始化控制器组件
        self.treasure_manager = TreasureManager(self.treasure_model, self.maze_model, self.path_finder)
        
        # 游戏状态
        self.player_position: Tuple[int, int] = (self.settings.start_x, self.settings.start_y)
        self.score: int = 0
        self.moves: int = 0
        self.game_over: bool = False
        self.end_position: Tuple[int, int] = (self.settings.end_x, self.settings.end_y)
        self.current_path: List[Tuple[int, int]] = []
        
        # 回调函数
        self.on_player_move: Optional[Callable[[Tuple[int, int]], None]] = None
        self.on_treasure_collect: Optional[Callable[[int, bool], None]] = None
        self.on_game_over: Optional[Callable[[int], None]] = None
    
    def initialize_game(self) -> None:
        """初始化游戏状态"""
        # 重置模型
        self.treasure_model.reset()
        
        # 重置控制器
        self.treasure_manager.reset()
        
        # 重置游戏状态
        self.player_position = (self.settings.start_x, self.settings.start_y)
        self.end_position = (self.settings.end_x, self.settings.end_y)
        self.score = 0
        self.moves = 0
        self.game_over = False
        self.current_path = []
        
        # 预计算宝藏之间的距离
        self.treasure_manager.calculate_treasure_distances()
    
    def set_player_move_callback(self, callback: Callable[[Tuple[int, int]], None]) -> None:
        """设置玩家移动回调"""
        self.on_player_move = callback
    
    def set_treasure_collect_callback(self, callback: Callable[[int, bool], None]) -> None:
        """设置宝藏收集回调"""
        self.on_treasure_collect = callback
    
    def set_game_over_callback(self, callback: Callable[[int], None]) -> None:
        """设置游戏结束回调"""
        self.on_game_over = callback
    
    def move_player(self, direction: str) -> bool:
        """
        移动玩家
        
        Args:
            direction: 移动方向 ("up", "down", "left", "right")
            
        Returns:
            移动是否成功
        """
        if self.game_over:
            return False
        
        x, y = self.player_position
        
        # 计算新位置
        if direction.lower() == "up":
            x -= 1
        elif direction.lower() == "down":
            x += 1
        elif direction.lower() == "left":
            y -= 1
        elif direction.lower() == "right":
            y += 1
        else:
            return False
        
        # 检查新位置是否可行
        if self.maze_model.is_wall(x, y):
            return False
        
        # 更新位置
        new_position = (x, y)
        self.player_position = new_position
        self.moves += 1
        
        # 调用回调
        if self.on_player_move:
            self.on_player_move(new_position)
        
        # 检查是否有宝藏
        self._check_treasure_at_current_position()
        
        # 检查游戏是否结束
        self._check_game_state()
        
        return True
    
    def move_player_along_path(self) -> bool:
        """
        沿着预设路径移动玩家
        
        Returns:
            移动是否成功
        """
        if not self.current_path or len(self.current_path) <= 1:
            return False
        
        # 获取下一个位置
        next_pos = self.current_path[1]
        self.current_path = self.current_path[1:]
        
        # 更新位置
        new_position = next_pos
        self.player_position = new_position
        self.moves += 1
        
        # 调用回调
        if self.on_player_move:
            self.on_player_move(new_position)
        
        # 检查是否有宝藏
        self._check_treasure_at_current_position()
        
        # 检查游戏是否结束
        self._check_game_state()
        
        return True
    
    def _check_treasure_at_current_position(self) -> None:
        """检查当前位置是否有宝藏"""
        x, y = self.player_position
        treasure = self.treasure_model.get_treasure_by_position((x, y))
        
        if treasure and not treasure.is_collected:
            # 收集宝藏
            self.treasure_manager.collect_treasure(treasure.treasure_id)
            
            # 更新分数
            if not treasure.is_fake:
                self.score += 10  # 真宝藏加10分
            else:
                self.score -= 5   # 假宝藏减5分
            
            # 调用回调
            if self.on_treasure_collect:
                self.on_treasure_collect(treasure.treasure_id, treasure.is_fake)
    
    def _check_game_state(self) -> None:
        """检查游戏状态"""
        # 如果所有真宝藏都被收集，游戏结束
        real_treasures = self.treasure_model.get_real_treasures()
        collected_real = [t for t in real_treasures if t.is_collected]
        
        if self.player_position == self.end_position:
            self.game_over = True
            if self.on_game_over:
                print(f"游戏结束，得分: {self.score}")
                self.on_game_over(self.score)
    
    def find_path_to_next_treasure(self) -> List[Tuple[int, int]]:
        """
        查找到下一个宝藏的路径
        
        Returns:
            路径坐标列表
        """
        # 获取下一个要访问的宝藏
        next_treasure = self.treasure_manager.get_next_treasure()
        if not next_treasure:
            return []
        
        # 查找路径
        path = self.path_finder.find_path(
            self.maze_model.get_maze_copy(),
            self.player_position,
            next_treasure.position
        )
        
        self.current_path = path
        return path
    
    def find_path_to_end(self) -> List[Tuple[int, int]]:
        """
        查找到终点的路径
        
        Returns:
            路径坐标列表
        """
        # 查找路径
        path = self.path_finder.find_path(
            self.maze_model.get_maze_copy(),
            self.player_position,
            self.end_position
        )
        
        self.current_path = path
        return path
    
    def get_treasure_info(self) -> Dict[int, Dict[str, Any]]:
        """
        获取宝藏信息
        
        Returns:
            宝藏信息字典，键为宝藏ID
        """
        treasures = self.treasure_model.get_all_treasures()
        treasure_info = {}
        
        for treasure in treasures:
            treasure_info[treasure.treasure_id] = {
                "position": treasure.position,
                "is_fake": treasure.is_fake,
                "is_discovered": treasure.is_discovered,
                "is_collected": treasure.is_collected,
                "status": treasure.status
            }
        
        return treasure_info
    
    def get_game_state(self) -> Dict[str, Any]:
        """
        获取游戏状态
        
        Returns:
            游戏状态字典
        """
        return {
            "player_position": self.player_position,
            "end_position": self.end_position,
            "score": self.score,
            "moves": self.moves,
            "game_over": self.game_over,
            "treasures_collected": len(self.treasure_model.get_collected_treasures()),
            "real_treasures_total": len(self.treasure_model.get_real_treasures()),
            "fake_treasures_total": len(self.treasure_model.get_fake_treasures())
        }
    
    def get_inferred_fake_treasures(self) -> List[int]:
        """
        获取推断为假的宝藏
        
        Returns:
            推断为假宝藏的ID列表
        """
        return self.treasure_manager.infer_fake_treasures() 