class Task:
    def __init__(self, task_id, arrive_time, node_id, victim, deadline):
        # 任务的唯一标识符
        self.task_id = task_id
        # 任务到达的时间
        self.arrive_time = arrive_time
        # 任务所在的节点ID
        self.node_id = node_id
        # 任务中的受害者数量
        self.victim = victim
        # 已救援的受害者数量
        self.rescued_victim = 0
        # 初始的受害者数量
        self.initial_victim = victim
        # 任务的截止时间
        self.deadline = deadline
        # 分配给该任务的救援人员列表
        self.assigned_rescuers = []

    def add_rescuer(self, rescuer):
        # 为任务分配救援人员
        if rescuer not in self.assigned_rescuers:
            self.assigned_rescuers.append(rescuer)
            rescuer.busy = True
            rescuer.task = self
            if self not in rescuer.participated_tasks:
                rescuer.participated_tasks.append(self)

    def remove_rescuer(self, rescuer):
        # 从任务中移除救援人员
        if rescuer in self.assigned_rescuers:
            self.assigned_rescuers.remove(rescuer)
        rescuer.busy = False
        rescuer.task = None
        rescuer.path = []
        rescuer.task_locked = False

    def update(self, current_time):
        if current_time <= self.deadline:
            arrived_rescuers = [r for r in self.assigned_rescuers if r._get_current_node() == self.node_id]
            total_capacity = sum(r.rescue_capacity for r in arrived_rescuers)
            print(
                f"任务#{self.task_id} 在时间{current_time}: 到达救援人员={len(arrived_rescuers)}, 总救援能力={total_capacity}")

            self.rescued_victim += total_capacity
            if self.rescued_victim > self.victim:
                self.rescued_victim = self.victim
                print(f"任务#{self.task_id} 已完成")

        # 任务完成或过期时，清理救援人员
        if self.rescued_victim >= self.victim or current_time >= self.deadline:
            print(f"任务#{self.task_id} 状态: {'已完成' if self.rescued_victim >= self.victim else '已过期'}")
            for r in self.assigned_rescuers[:]:
                self.remove_rescuer(r)