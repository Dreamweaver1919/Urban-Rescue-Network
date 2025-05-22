
import numpy as np

class Rescuer:
    def __init__(self, rescuer_id, grid_size):
        """
        更新后的救援人员类，包含边界检查和移动优化
        :param rescuer_id: 救援人员唯一标识
        :param grid_size: 网格尺寸（确保与Environment一致）
        """
        self.rescuer_id = rescuer_id
        self.grid_size = grid_size
        
        # 位置属性
        self.x = np.random.randint(0, grid_size)
        self.y = np.random.randint(0, grid_size)
        self.destination_x = self.x  # 新增目标位置跟踪
        self.destination_y = self.y
        
        # 移动状态
        self.direction_x = True
        self.direction_y = True
        self.moving_x = False
        self.moving_y = False
        
        # 任务状态
        self.busy = False
        self.task = None
        
        # 统计指标
        self.moved_distance = 0
        self.completed_tasks = 0  # 新增任务完成计数

    def move(self, destination_x, destination_y):
        """更新移动方法：添加边界约束和状态重置"""
        # 限制目标位置在网格范围内
        self.destination_x = np.clip(int(destination_x), 0, self.grid_size-1)
        self.destination_y = np.clip(int(destination_y), 0, self.grid_size-1)
        
        # 更新移动方向状态
        self.moving_x = (self.x != self.destination_x)
        self.moving_y = (self.y != self.destination_y)
        
        # 计算移动方向
        self.direction_x = (self.destination_x > self.x)
        self.direction_y = (self.destination_y > self.y)

    def update(self, current_time):
        """
        更新方法：与Environment.py中的任务管理逻辑对齐
        添加任务有效性检查和状态同步
        """
        prev_pos = (self.x, self.y)
        
        if not self.busy:
            # 轴对齐移动逻辑
            if self.moving_x:
                self.x += 1 if self.direction_x else -1
                # 到达目标位置后停止
                if (self.direction_x and self.x >= self.destination_x) or \
                   (not self.direction_x and self.x <= self.destination_x):
                    self.x = self.destination_x
                    self.moving_x = False
                    
            elif self.moving_y:
                self.y += 1 if self.direction_y else -1
                if (self.direction_y and self.y >= self.destination_y) or \
                   (not self.direction_y and self.y <= self.destination_y):
                    self.y = self.destination_y
                    self.moving_y = False
        
        # 更新移动距离（曼哈顿距离）
        self.moved_distance += abs(self.x - prev_pos[0]) + abs(self.y - prev_pos[1])
        
        # 任务到达检测（添加任务有效性检查）
        if self.task is not None:
            # 检查任务是否仍然有效
            if not hasattr(self.task, 'x') or not hasattr(self.task, 'y'):
                self._release_task()
                return
                
            # 到达检测（允许1个单位的误差）
            if (abs(self.x - self.task.x) <= 1) and (abs(self.y - self.task.y) <= 1):
                self.x = self.task.x  # 精确对齐坐标
                self.y = self.task.y
                if not self.busy:
                    try:
                        self.task.add_rescuer(self,current_time)
                        self.completed_tasks += 1  # 统计任务参与数
                    except AttributeError:
                        self._release_task()

    def _release_task(self):
        """内部方法：安全释放任务引用"""
        self.task = None
        self.busy = False
        self.moving_x = False
        self.moving_y = False

    def __repr__(self):
        return (f"Rescuer#{self.rescuer_id} [{'Busy' if self.busy else 'Free'}] "
                f"Pos:({self.x},{self.y}) Dist:{self.moved_distance} "
                f"Tasks:{self.completed_tasks}")