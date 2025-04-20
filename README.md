# MazeTreasureHunter

这是一个用于探索迷宫中宝藏的系统，通过图像处理（OpenCV/YOLO）识别宝藏，通过Dijkstra算法搜索不同节点之间的最短路径，结合推理判断通过贪心/DFS算法选择最优路径。

本项目基于2023年光电设计大赛作品，时隔两年，重构完善代码。

## 使用方法

### 环境要求

- Python 3.10
- UV包管理工具

### 环境配置

- 安装UV工具：

```Powershell
pip install uv
```

- 创建Python 3.10虚拟环境：

```Powershell
uv venv --python=3.10
```

- 激活虚拟环境：

   Windows:

   ```Powershell
   .\.venv\Scripts\activate
   ```

   Linux/Mac:

   ```bash
   source .venv/bin/activate
   ```

- 安装项目依赖：

```Powershell
# 从requirements.txt安装依赖
uv pip install -r requirements.txt
```

- 验证Python版本：

```Powershell
python --version
```

确保输出为Python 3.10.x

### 运行程序

```bash
python main.py
```
