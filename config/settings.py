from typing import Dict, Any

class GameSettings:
    """游戏参数配置类，管理所有游戏相关设置"""
    
    def __init__(self) -> None:
        # 路径算法参数
        self.direct_parameter: float = 1.0
        self.cross_parameter: float = 2.0
        self.turn_parameter: float = 3.0
        self.capture_parameter: float = 4.0
        
        # 游戏核心参数
        self.maze_parameter: int = 206
        self.photo_number: int = 3
        
        # 玩家起始和终点坐标
        self.start_x: int = 18
        self.start_y: int = 0
        self.end_x: int = 0
        self.end_y: int = 18
        
    def to_dict(self) -> Dict[str, Any]:
        """将配置转换为字典格式"""
        return {
            "direct_parameter": self.direct_parameter,
            "cross_parameter": self.cross_parameter,
            "turn_parameter": self.turn_parameter,
            "capture_parameter": self.capture_parameter,
            "maze_parameter": self.maze_parameter,
            "photo_number": self.photo_number,
            "start_x": self.start_x,
            "start_y": self.start_y,
            "end_x": self.end_x,
            "end_y": self.end_y
        }
    
    def from_dict(self, config_dict: Dict[str, Any]) -> None:
        """从字典加载配置"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value) 