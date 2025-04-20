"""
迷宫寻宝游戏主程序

这是一个迷宫寻宝游戏，玩家需要在迷宫中寻找宝藏。
"""

from controllers.game_controller import GameController
from ui.gui import GUI

def main() -> None:
    """主程序入口"""
    # 创建游戏控制器
    game_controller = GameController()
    
    # 初始化游戏
    game_controller.initialize_game()
    
    # 创建并初始化图形界面
    gui = GUI(game_controller)
    gui.initialize()
    
    # 启动游戏界面
    gui.run()

if __name__ == "__main__":
    main() 