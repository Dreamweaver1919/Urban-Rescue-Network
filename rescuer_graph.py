class Rescuer:
    def __init__(self, rescuer_id, start_position, rescue_capacity, graph, speed=1):
        self.rescuer_id = rescuer_id
        self.start_position = start_position  # 起始位置（节点或边）
        self.graph = graph
        self.rescue_capacity = rescue_capacity  # 救援能力（必需参数）
        self.speed = speed
        self.busy = False
        self.task = None
        self.moved_distance = 0
        self.participated_tasks = []
        self.task_locked = False
        self.path = []
        self.current_position = None
        self._init_position()  # 初始化位置

    def _init_position(self):
        # 根据初始位置的类型，初始化当前位置
        if self.start_position[0] == 'node':
            node_id = self.start_position[1]
            self.current_position = ('node', node_id)
        else:
            u, v, distance_from_u = self.start_position[1], self.start_position[2], self.start_position[3]
            edge_length = self.graph.nodes[u].neighbors[v]
            # 确保距离在合法范围内
            distance_from_u = max(0, min(distance_from_u, edge_length))
            self.current_position = ('edge', u, v, distance_from_u)

    def current_node(self):
        """获取当前所在节点ID（边的起点或当前节点）"""
        if self.current_position[0] == 'node':
            return self.current_position[1]
        elif self.current_position[0] == 'edge':
            return self.current_position[1]  # 使用边的起点u作为当前节点（或根据进度选择更近的节点）
        else:
            raise ValueError("无效的位置格式，必须为 ('node', id) 或 ('edge', u, v, progress)")


    def _get_current_node(self):
        """获取当前所在的节点ID（用于路径规划起点）"""
        if self.current_position[0] == 'node':
            return self.current_position[1]
        else:
            # 处于边上时，路径规划起点为边的起点或终点（根据移动方向）
            u, v, d = self.current_position[1], self.current_position[2], self.current_position[3]
            edge_length = self.graph.nodes[u].neighbors[v]
            # 若已移动超过边长度的50%，则当前节点为v，否则为u
            return v if d > edge_length / 2 else u

    def move_to_task(self, target_node):
        # 规划救援人员到目标节点的路径
        current_node = self._get_current_node()
        # 如果当前节点就是目标节点，清空路径
        if current_node == target_node:
            self.path = []
            return
        # 使用Dijkstra算法计算最短路径
        full_path = self.graph.dijkstra(current_node, target_node)
        if not full_path:
            print(f"Rescuer#{self.rescuer_id} 到目标节点{target_node}无有效路径")
            self.path = []
            return
        # 关键修改：移除路径中的当前节点，仅保留后续节点
        self.path = full_path[1:]  # 例如原路径 [A,B,C] → 修正为 [B,C]
        print(f"Rescuer#{self.rescuer_id} 路径: {current_node} -> {self.path}")

    def update_position(self):
        # 更新救援人员的位置（添加日志）
        if self.current_position[0] == 'edge':
            # 处理在边上的移动
            u, v, d = self.current_position[1], self.current_position[2], self.current_position[3]
            edge_length = self.graph.nodes[u].neighbors[v]
            move_dist = self.speed
            print(f"[Rescuer#{self.rescuer_id}] 当前在边{u}-{v}（已移动{d}/{edge_length}），速度{move_dist}")

            if d + move_dist <= edge_length:
                # 未到达终点，继续在边上移动
                self.current_position = ('edge', u, v, d + move_dist)
                self.moved_distance += move_dist
                print(f"→ 移动到边{u}-{v}的{d+move_dist}位置，剩余路径：{self.path}")
            else:
                # 到达终点节点v
                self.moved_distance += (edge_length - d)
                self.current_position = ('node', v)
                print(f"→ 到达节点{v}，剩余路径（原）：{self.path}")
                # 移除路径中已到达的节点（仅当路径第一个节点是v时）
                if self.path and self.path[0] == v:
                    self.path.pop(0)
                    print(f"→ 移除路径中的节点{v}，剩余路径：{self.path}")

        elif self.path:
            # 处理从节点出发的移动
            current_node = self._get_current_node()
            next_node = self.path[0] if self.path else None
            if not next_node:
                return

            edge_length = self.graph.nodes[current_node].neighbors.get(next_node, 0)
            if edge_length == 0:
                print(f"[错误] 无效边: {current_node}->{next_node}，路径：{self.path}")
                return

            move_dist = self.speed
            print(f"[Rescuer#{self.rescuer_id}] 在节点{current_node}，目标节点{next_node}（边长度{edge_length}），速度{move_dist}")

            if move_dist <= edge_length:
                # 开始在边上移动（未到达终点）
                self.current_position = ('edge', current_node, next_node, move_dist)
                self.moved_distance += move_dist
                print(f"→ 进入边{current_node}-{next_node}的{move_dist}位置，剩余路径：{self.path}")
            else:
                # 直接到达下一个节点
                self.moved_distance += edge_length
                self.current_position = ('node', next_node)
                self.path.pop(0)  # 移除已到达的节点
                print(f"→ 到达节点{next_node}，剩余路径：{self.path}")

    def update(self, current_time):
        # 检查任务是否已完成或过期
        if self.task is not None:
            if self.task.rescued_victim >= self.task.victim or current_time >= self.task.deadline:
                print(f"救援人员#{self.rescuer_id} 任务已完成或过期，释放")
                self.task.remove_rescuer(self)  # 确保从任务中移除
                self.busy = False
                self.task = None
                self.path = []
                self.task_locked = False  # 解锁任务

        # 如果没有任务，直接返回
        if self.task is None:
            return

        # 检查是否需要更新路径（例如，任务节点改变）
        if self.task and self.path and self.path[-1] != self.task.node_id:
            print(f"救援人员#{self.rescuer_id} 更新路径到任务#{self.task.task_id}")
            self.move_to_task(self.task.node_id)

        self.update_position()

        current_node = self._get_current_node()
        if current_node == self.task.node_id and not self.busy:
            self.task.add_rescuer(self)
            self.busy = True
            print(f"救援人员#{self.rescuer_id} 到达任务#{self.task.task_id} 并开始执行")