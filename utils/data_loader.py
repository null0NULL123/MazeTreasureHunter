import numpy as np
import ast
import json
from typing import List, Tuple, Dict, Any, Optional

class DataLoader:
    """数据加载工具类"""
    
    @staticmethod
    def load_maze(file_path: str) -> np.ndarray:
        """
        从文件加载迷宫数据
        
        Args:
            file_path: 迷宫文件路径
            
        Returns:
            迷宫数组
        """
        try:
            return np.loadtxt(file_path, dtype=int)
        except Exception as e:
            print(f"加载迷宫文件失败: {e}")
            # 返回一个默认的小迷宫
            return np.zeros((10, 10), dtype=int)
    
    @staticmethod
    def load_list_from_file(file_path: str) -> List[Any]:
        """
        从文件加载列表数据
        
        Args:
            file_path: 列表文件路径
            
        Returns:
            列表数据
        """
        try:
            with open(file_path, 'r') as file:
                content = file.read()
                return ast.literal_eval(content)
        except Exception as e:
            print(f"加载列表文件失败: {e}")
            return []
    
    @staticmethod
    def load_json_from_file(file_path: str) -> Dict[str, Any]:
        """
        从文件加载JSON数据
        
        Args:
            file_path: JSON文件路径
            
        Returns:
            字典数据
        """
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except Exception as e:
            print(f"加载JSON文件失败: {e}")
            return {}
    
    @staticmethod
    def save_list_to_file(data: List[Any], file_path: str) -> bool:
        """
        将列表数据保存到文件
        
        Args:
            data: 列表数据
            file_path: 保存路径
            
        Returns:
            是否保存成功
        """
        try:
            with open(file_path, 'w') as file:
                file.write(str(data))
            return True
        except Exception as e:
            print(f"保存列表到文件失败: {e}")
            return False
    
    @staticmethod
    def save_json_to_file(data: Dict[str, Any], file_path: str) -> bool:
        """
        将字典数据保存为JSON文件
        
        Args:
            data: 字典数据
            file_path: 保存路径
            
        Returns:
            是否保存成功
        """
        try:
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)
            return True
        except Exception as e:
            print(f"保存JSON到文件失败: {e}")
            return False
    
    @staticmethod
    def save_maze_to_file(maze: np.ndarray, file_path: str) -> bool:
        """
        将迷宫数据保存到文件
        
        Args:
            maze: 迷宫数组
            file_path: 保存路径
            
        Returns:
            是否保存成功
        """
        try:
            np.savetxt(file_path, maze, fmt='%d')
            return True
        except Exception as e:
            print(f"保存迷宫到文件失败: {e}")
            return False 