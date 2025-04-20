import numpy as np
from typing import List, Tuple, Any

class MatrixTransform:
    """矩阵变换工具类"""
    
    @staticmethod
    def transform_coordinates(coords: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        坐标变换，用于在不同坐标系统间转换
        
        Args:
            coords: 坐标列表
            
        Returns:
            变换后的坐标列表
        """
        result = []
        for coord in coords:
            result.append((18 - (2 * coord[0]), 2 * coord[1]))
        return result
    
    @staticmethod
    def transform_coordinate(coord: Tuple[int, int]) -> Tuple[int, int]:
        """
        单个坐标变换
        
        Args:
            coord: 坐标
            
        Returns:
            变换后的坐标
        """
        return (18 - (2 * coord[0]), 2 * coord[1])
    
    @staticmethod
    def matrix_to_position(coords: List[List[int]]) -> List[List[int]]:
        """
        矩阵坐标转换为位置坐标
        
        Args:
            coords: 矩阵坐标列表
            
        Returns:
            位置坐标列表
        """
        result = []
        for coord in coords:
            # 复制坐标以避免修改原始数据
            new_coord = coord.copy()
            new_coord[0] = int(9 - new_coord[0] / 2)
            new_coord[1] = int(new_coord[1] / 2)
            result.append(new_coord)
        return result
    
    @staticmethod
    def negative_matrix(matrix: List[List[float]]) -> List[List[float]]:
        """
        矩阵取负（将矩阵中的所有元素取负值）
        
        Args:
            matrix: 输入矩阵
            
        Returns:
            取负后的矩阵
        """
        matrix_np = np.array(matrix)
        negative_np = -matrix_np
        return negative_np.tolist()
    
    @staticmethod
    def rotate_matrix(matrix: List[List[Any]], direction: int = 1) -> List[List[Any]]:
        """
        旋转矩阵
        
        Args:
            matrix: 输入矩阵
            direction: 旋转方向，1表示顺时针，-1表示逆时针
            
        Returns:
            旋转后的矩阵
        """
        matrix_np = np.array(matrix)
        if direction == 1:  # 顺时针90度
            rotated = np.rot90(matrix_np, k=3)
        else:  # 逆时针90度
            rotated = np.rot90(matrix_np)
        return rotated.tolist()
    
    @staticmethod
    def flip_matrix(matrix: List[List[Any]], axis: int = 0) -> List[List[Any]]:
        """
        翻转矩阵
        
        Args:
            matrix: 输入矩阵
            axis: 翻转轴，0表示水平翻转，1表示垂直翻转
            
        Returns:
            翻转后的矩阵
        """
        matrix_np = np.array(matrix)
        flipped = np.flip(matrix_np, axis=axis)
        return flipped.tolist()
    
    @staticmethod
    def transpose_matrix(matrix: List[List[Any]]) -> List[List[Any]]:
        """
        转置矩阵
        
        Args:
            matrix: 输入矩阵
            
        Returns:
            转置后的矩阵
        """
        matrix_np = np.array(matrix)
        transposed = matrix_np.T
        return transposed.tolist() 