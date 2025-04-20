import tkinter as tk
from typing import Tuple
from controllers.game_controller import GameController

class GUI(tk.Tk):
    """游戏图形界面类"""
    
    def __init__(self, game_controller: GameController) -> None:
        """
        初始化图形界面
        
        Args:
            game_controller: 游戏控制器
        """
        super().__init__()
        
        self.controller = game_controller
        self.maze_model = game_controller.maze_model
        self.treasure_model = game_controller.treasure_model
        
        # 设置窗口
        self.title("迷宫寻宝游戏")
        self.configure(bg="white")
        
        # 设置网格单元大小
        self.UNIT = 30
        self.MAZE_R = self.maze_model.height
        self.MAZE_C = self.maze_model.width
        
        # 创建画布
        canvas_height = self.MAZE_R * self.UNIT
        canvas_width = self.MAZE_C * self.UNIT
        self.canvas = tk.Canvas(self, bg="white", height=canvas_height, width=canvas_width)
        self.canvas.pack(side=tk.LEFT)
        
        # 创建信息面板
        self.info_panel = tk.Frame(self, bg="white", width=200)
        self.info_panel.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 创建分数标签
        self.score_label = tk.Label(self.info_panel, text="分数: 0", font=("Arial", 14), bg="white")
        self.score_label.pack(pady=10)
        
        # 创建移动次数标签
        self.moves_label = tk.Label(self.info_panel, text="移动次数: 0", font=("Arial", 14), bg="white")
        self.moves_label.pack(pady=10)
        
        # 创建宝藏信息标签
        self.treasures_label = tk.Label(self.info_panel, text="宝藏: 0/0", font=("Arial", 14), bg="white")
        self.treasures_label.pack(pady=10)
        
        # 创建状态标签
        self.status_label = tk.Label(self.info_panel, text="游戏中", font=("Arial", 14), bg="white")
        self.status_label.pack(pady=10)
        
        # 创建按钮
        self.restart_button = tk.Button(self.info_panel, text="重新开始", command=self.restart_game)
        self.restart_button.pack(pady=10)
        
        self.auto_move_button = tk.Button(self.info_panel, text="自动寻宝", command=self.auto_move)
        self.auto_move_button.pack(pady=5)
        
        self.goto_end_button = tk.Button(self.info_panel, text="前往终点", command=self.go_to_end)
        self.goto_end_button.pack(pady=5)
        
        # 玩家图形对象
        self.player = None
        
        # 宝藏图形对象字典
        self.treasure_objects = {}
        
        # 绑定键盘事件
        self.bind("<KeyPress>", self.on_key_press)
        
        # 设置回调
        self.controller.set_player_move_callback(self.update_player_position)
        self.controller.set_treasure_collect_callback(self.update_treasure_status)
        self.controller.set_game_over_callback(self.game_over)
        
        # 自动移动标志
        self.auto_moving = False
        
        # 移动模式：0=寻宝，1=前往终点
        self.move_mode = 0
    
    def initialize(self) -> None:
        """初始化界面"""
        self.draw_maze()
        self.draw_end_position()
        self.draw_treasures()
        self.draw_player()
        self.update_info()
    
    def draw_maze(self) -> None:
        """绘制迷宫"""
        self.canvas.delete("all")
        
        # 绘制网格
        for i in range(self.MAZE_R):
            for j in range(self.MAZE_C):
                # 绘制单元格
                x1 = j * self.UNIT
                y1 = i * self.UNIT
                x2 = x1 + self.UNIT
                y2 = y1 + self.UNIT
                
                # 墙壁为黑色，通道为白色
                if self.maze_model.is_wall(i, j):
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="gray")
                else:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="gray")
    
    def draw_end_position(self) -> None:
        """绘制终点位置"""
        x, y = self.controller.end_position
        
        # 创建终点图形 - 使用红色方块
        x1 = y * self.UNIT
        y1 = x * self.UNIT
        x2 = x1 + self.UNIT
        y2 = y1 + self.UNIT
        
        # 绘制终点背景和标记
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="light green", outline="black")
        
        # 添加终点标志 - 使用红色旗帜图案
        flag_x1 = x1 + self.UNIT // 4
        flag_y1 = y1 + self.UNIT // 4
        flag_x2 = flag_x1 + self.UNIT // 2
        flag_y2 = flag_y1 + self.UNIT // 2
        
        # 绘制旗杆
        self.canvas.create_line(flag_x1, flag_y1, flag_x1, flag_y2, width=2, fill="brown")
        # 绘制旗帜
        self.canvas.create_polygon(
            flag_x1, flag_y1, 
            flag_x2, flag_y1 + self.UNIT // 4, 
            flag_x1, flag_y1 + self.UNIT // 2,
            fill="red", outline="black"
        )
        
        # 添加"终点"文字
        self.canvas.create_text(x1 + self.UNIT // 2, y1 + self.UNIT // 2 + self.UNIT // 4, 
                               text="终点", fill="black", font=("Arial", 9, "bold"))
    
    def draw_treasures(self) -> None:
        """绘制宝藏"""
        treasures = self.treasure_model.get_all_treasures()
        
        for treasure in treasures:
            x, y = treasure.position
            
            # 创建宝藏图形
            x1 = y * self.UNIT + self.UNIT // 4
            y1 = x * self.UNIT + self.UNIT // 4
            x2 = x1 + self.UNIT // 2
            y2 = y1 + self.UNIT // 2
            
            # 未被收集的宝藏为黄色，已收集的宝藏用不同颜色表示真假
            if not treasure.is_collected:
                obj = self.canvas.create_oval(x1, y1, x2, y2, fill="gold", outline="black")
            else:
                if treasure.is_fake:
                    obj = self.canvas.create_oval(x1, y1, x2, y2, fill="red", outline="black")
                else:
                    obj = self.canvas.create_oval(x1, y1, x2, y2, fill="green", outline="black")
            
            # 存储图形对象ID
            self.treasure_objects[treasure.treasure_id] = obj
            
            # 添加宝藏编号
            self.canvas.create_text(x1 + self.UNIT // 4, y1 + self.UNIT // 4, text=str(treasure.treasure_id+1), fill="black")
    
    def draw_player(self) -> None:
        """绘制玩家"""
        x, y = self.controller.player_position
        
        # 创建玩家图形
        x1 = y * self.UNIT + self.UNIT // 6
        y1 = x * self.UNIT + self.UNIT // 6
        x2 = x1 + self.UNIT * 2 // 3
        y2 = y1 + self.UNIT * 2 // 3
        
        self.player = self.canvas.create_oval(x1, y1, x2, y2, fill="blue", outline="black")
    
    def update_player_position(self, position: Tuple[int, int]) -> None:
        """
        更新玩家位置
        
        Args:
            position: 玩家新位置
        """
        if not self.player:
            return
        
        # 获取新位置
        x, y = position
        
        # 计算当前位置，需要根据当前画布中的玩家位置计算
        current_coords = self.canvas.coords(self.player)
        current_x = current_coords[0] - self.UNIT // 6
        current_y = current_coords[1] - self.UNIT // 6
        
        # 计算移动量（目标位置 - 当前位置）
        dx = (y * self.UNIT + self.UNIT // 6) - current_x
        dy = (x * self.UNIT + self.UNIT // 6) - current_y
        
        # 移动玩家图形
        self.canvas.move(self.player, dx, dy)
        print(f"玩家移动到: {x}, {y}")
        
        # 更新信息面板
        self.update_info()
        
        # 刷新画布
        self.update()
    
    def update_treasure_status(self, treasure_id: int, is_fake: bool) -> None:
        """
        更新宝藏状态
        
        Args:
            treasure: 被收集的宝藏
        """
        # 获取宝藏图形对象
        obj = self.treasure_objects.get(treasure_id)
        if not obj:
            return
        
        # 更新宝藏颜色
        if is_fake:
            self.canvas.itemconfig(obj, fill="red")
        else:
            self.canvas.itemconfig(obj, fill="green")
        
        # 更新信息面板
        self.update_info()
    
    def update_info(self) -> None:
        """更新信息面板"""
        # 获取游戏状态
        state = self.controller.get_game_state()
        
        # 更新标签
        self.score_label.config(text=f"分数: {state['score']}")
        self.moves_label.config(text=f"移动次数: {state['moves']}")
        self.treasures_label.config(text=f"宝藏: {state['treasures_collected']}/{state['real_treasures_total'] + state['fake_treasures_total']}")
        
        # 更新状态
        if state['game_over']:
            self.status_label.config(text="游戏结束", fg="red")
        else:
            self.status_label.config(text="游戏中", fg="green")
    
    def game_over(self, score: int) -> None:
        """
        游戏结束处理
        
        Args:
            score: 最终分数
        """
        self.update_info()
        self.status_label.config(text=f"游戏结束\n最终分数: {score}", fg="red")
        
        # 停止自动移动
        self.auto_moving = False
    
    def on_key_press(self, event) -> None:
        """
        处理键盘事件
        
        Args:
            event: 键盘事件
        """
        key = event.keysym.lower()
        
        # 移动方向映射
        direction_map = {
            "up": "up",
            "down": "down",
            "left": "left",
            "right": "right",
            "w": "up",
            "s": "down",
            "a": "left",
            "d": "right"
        }
        
        # 如果是方向键，移动玩家
        if key in direction_map:
            self.controller.move_player(direction_map[key])
        
        # 如果是空格键，自动移动切换
        elif key == "space":
            self.auto_move()
    
    def restart_game(self) -> None:
        """重新开始游戏"""
        # 停止自动移动
        self.auto_moving = False
        
        # 初始化游戏
        self.controller.initialize_game()
        
        # 重新绘制界面
        self.initialize()
    
    def go_to_end(self) -> None:
        """前往终点"""
        # 停止当前自动移动
        self.auto_moving = False
        self.auto_move_button.config(text="自动寻宝")
        
        # 设置移动模式为前往终点
        self.move_mode = 1
        
        # 查找到终点的路径
        path = self.controller.find_path_to_end()
        print(f"前往终点路径: {path}")
        
        if path:
            # 开始自动移动
            self.auto_moving = True
            self.goto_end_button.config(text="停止移动")
            self._auto_move_step()
        else:
            self.goto_end_button.config(text="无法到达终点")
    
    def auto_move(self) -> None:
        """自动移动玩家寻找宝藏"""
        # 切换自动移动状态
        self.auto_moving = not self.auto_moving
        
        # 设置移动模式为寻宝
        self.move_mode = 0
        
        if self.auto_moving:
            self.auto_move_button.config(text="停止寻宝")
            self._auto_move_step()
        else:
            self.auto_move_button.config(text="自动寻宝")
            self.goto_end_button.config(text="前往终点")
    
    def _auto_move_step(self) -> None:
        """执行一步自动移动"""
        if not self.auto_moving or self.controller.game_over:
            self.auto_moving = False
            self.auto_move_button.config(text="自动寻宝")
            self.goto_end_button.config(text="前往终点")
            return
        
        # 如果没有当前路径，根据模式获取路径
        if not self.controller.current_path:
            if self.move_mode == 0:  # 寻宝模式
                self.controller.find_path_to_next_treasure()
            else:  # 前往终点模式
                self.controller.find_path_to_end()
            
            print(f"路径: {self.controller.current_path}")
        
        # 如果有路径，沿着路径移动
        if self.controller.current_path:
            self.controller.move_player_along_path()
            
            # 延迟一段时间后继续移动
            self.after(10, self._auto_move_step)
        else:
            # 如果没有路径，停止自动移动
            self.auto_moving = False
            if self.move_mode == 0:
                self.auto_move_button.config(text="自动寻宝")
            else:
                self.goto_end_button.config(text="前往终点")
    
    def run(self) -> None:
        """运行游戏界面"""
        self.mainloop() 