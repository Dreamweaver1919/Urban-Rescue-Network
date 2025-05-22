import random

class Task:
    def __init__(self, task_id, grid_size, x=None, y=None, victim=10, deadline=100):
        """
        :param task_id: 任务编号
        :param grid_size: 地图网格边长（用于随机坐标生成）
        :param x, y: 任务位置（若未提供则在 grid_size * grid_size 范围内随机生成）
        :param victim: 初始需要被救援的人数（重命名为initial_victim保持一致性）
        :param deadline: 任务截止时间
        """
        self.task_id = task_id
        self.grid_size = grid_size
        self.x = x if x is not None else random.randint(0, grid_size - 1)
        self.y = y if y is not None else random.randint(0, grid_size - 1)
        self.initial_victim = victim  # 统一使用initial_victim
        self.rescued_victim = 0
        self.deadline = deadline
        self.assigned_rescuers = []
        self.completed = False  # 新增完成状态

    def add_rescuer(self, rescuer, current_time):
        if rescuer in self.assigned_rescuers:
            return
        if current_time <= self.deadline:
            self.assigned_rescuers.append(rescuer)
            rescuer.busy = True
            rescuer.task = self
            print(f"DEBUG_ADD: Task{self.task_id}.assigned_rescuers = {[r.rescuer_id for r in self.assigned_rescuers]}")

    def remove_rescuer(self, rescuer):
        """增加状态同步：更新救援者位置和任务状态"""
        if rescuer in self.assigned_rescuers:
            self.assigned_rescuers.remove(rescuer)
            rescuer.busy = False
            rescuer.task = None
            # 重置救援者位置到任务点（避免移动中状态不一致）
            rescuer.x = self.x
            rescuer.y = self.y

    def update(self, current_time):
        print(f"DEBUG_UPDATE_START: Task{self.task_id}.assigned_rescuers = {[r.rescuer_id for r in self.assigned_rescuers]}")
        if current_time > self.deadline:
            for r in self.assigned_rescuers[:]:
                self.remove_rescuer(r)
            self.completed = True
            return 0

        before = self.rescued_victim
        active = [
            r for r in self.assigned_rescuers
            if abs(r.x - self.x) <= 1 and abs(r.y - self.y) <= 1
        ]
        self.rescued_victim = min(self.rescued_victim + len(active), self.initial_victim)
        delta = self.rescued_victim - before
        print(f"DEBUG: 任务{self.task_id} 本步救援+{delta}")
        return delta
    

    def __repr__(self):
        """显示关键状态：任务位置、救援进度、剩余时间"""
        time_left = max(self.deadline - (getattr(self, 'env', None).current_time if hasattr(self, 'env') else 0), 0)
        return (f"Task#{self.task_id} @({self.x},{self.y}) | "
                f"Rescued: {self.rescued_victim}/{self.initial_victim} | "
                f"DL: {time_left}/{self.deadline} | "
                f"Agents: {len(self.assigned_rescuers)}")