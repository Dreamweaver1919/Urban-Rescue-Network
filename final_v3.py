import random
import heapq
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

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
        node_id = self.start_position[1]
        self.current_position = ('node', node_id)

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
            u, v, d = self.current_position[1], self.current_position[2], self.current_position[3]
            edge_length = self.graph.nodes[u].neighbors[v]
            return v if d > edge_length / 2 else u

    def move_to_task(self, target_node):
        current_node = self._get_current_node()
        full_path = self.graph.dijkstra(current_node, target_node)
        if full_path and full_path[0] == current_node:
            self.path = full_path[1:]  # 关键修正：移除当前节点
        else:
            self.path = []
        print(f"Rescuer#{self.rescuer_id} 路径: {current_node} -> {self.path}")

    def update_position(self):
        if self.current_position[0] == 'edge':
            u, v, d = self.current_position[1], self.current_position[2], self.current_position[3]
            edge_length = self.graph.nodes[u].neighbors[v]
            move_dist = self.speed

            # 计算剩余可移动距离
            remaining_dist = edge_length - d
            if move_dist <= remaining_dist:
                # 未到达终点，继续在边上移动
                self.current_position = ('edge', u, v, d + move_dist)
                self.moved_distance += move_dist
            else:
                # 到达终点节点v
                self.current_position = ('node', v)
                self.moved_distance += remaining_dist
                # 如果路径中下一个节点是v，则移除
                if self.path and self.path[0] == v:
                    self.path.pop(0)
        elif self.path:
            current_node = self._get_current_node()
            next_node = self.path[0]
            if next_node in self.graph.nodes[current_node].neighbors:
                edge_length = self.graph.nodes[current_node].neighbors[next_node]
                # 进入边移动状态
                self.current_position = ('edge', current_node, next_node, self.speed)
                self.moved_distance += self.speed

    def update(self, current_time):
        # 检查任务状态
        if self.task and (self.task.rescued_victim >= self.task.victim or current_time >= self.task.deadline):
            self.task.remove_rescuer(self)
            self.busy = False
            self.task = None
            self.path = []
            return

        # 更新位置并检查是否到达
        self.update_position()

        # 关键修改：只有当当前位置是节点且匹配任务节点时，才触发救援
        if (self.task is not None and
                self.current_position[0] == 'node' and
                self.current_position[1] == self.task.node_id and
                not self.busy):
            self.task.add_rescuer(self)
            self.busy = True
            print(f"Rescuer#{self.rescuer_id} 已到达任务#{self.task.task_id}并开始救援")


SEED = 50
random.seed(SEED)
np.random.seed(SEED)

class Task:
    def __init__(self, task_id, arrive_time, node_id, victim, deadline, weibull_shape=2.0):
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
        self.survivors_not_rescued = victim
        # 任务的截止时间
        self.deadline = deadline
        # 分配给该任务的救援人员列表
        self.assigned_rescuers = []
        # Weibull 衰减参数
        self.weibull_shape = weibull_shape
        T = max(1, self.deadline - self.arrive_time)
        self.weibull_scale = T / (np.log(10) ** (1.0 / self.weibull_shape))

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
        # 离散 Weibull 衰减（死亡），至少保留 1 人
        t = current_time - self.arrive_time
        if 1 <= t < (self.deadline - self.arrive_time):
            delta = ((t / self.weibull_scale) ** self.weibull_shape
                     - ((t - 1) / self.weibull_scale) ** self.weibull_shape)
            h = 1 - np.exp(-delta)
            N = self.survivors_not_rescued
            if N > 1:
                deaths = np.random.binomial(N, h)
                deaths = min(deaths, N - 1)
                self.survivors_not_rescued -= deaths
        if current_time >= self.deadline:
            self.survivors_not_rescued = 0

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

# 并查集类，用于生成图时确保图的连通性
class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        fx = self.find(x)
        fy = self.find(y)
        if fx != fy:
            self.parent[fy] = fx
            return True
        return False


class GraphNode:
    def __init__(self, node_id, x, y):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.neighbors = {}


class RescueGraph:
    def __init__(self):
        self.nodes = {}

    def add_node(self, node_id, x, y):
        self.nodes[node_id] = GraphNode(node_id, x, y)

    def add_edge(self, node1, node2, distance):
        self.nodes[node1].neighbors[node2] = distance
        self.nodes[node2].neighbors[node1] = distance

    def dijkstra(self, start_id, end_id):
        if start_id not in self.nodes or end_id not in self.nodes:
            print(f"[Dijkstra错误] 起点 {start_id} 或终点 {end_id} 不存在")
            return []

        print(f"[Dijkstra开始] 计算 {start_id} → {end_id} 的最短路径...")
        distances = {node: float('inf') for node in self.nodes}
        prev_nodes = {node: None for node in self.nodes}
        distances[start_id] = 0
        priority_queue = [(0, start_id)]

        while priority_queue:
            current_dist, current_node = heapq.heappop(priority_queue)

            # 提前终止：找到终点
            if current_node == end_id:
                break

            # 跳过非最短路径的旧记录
            if current_dist > distances[current_node]:
                continue

            # 遍历邻居更新距离
            for neighbor, edge_dist in self.nodes[current_node].neighbors.items():
                new_dist = current_dist + edge_dist
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    prev_nodes[neighbor] = current_node
                    heapq.heappush(priority_queue, (new_dist, neighbor))

        # 回溯路径
        path = []
        current = end_id
        while current is not None:
            path.append(current)
            current = prev_nodes.get(current)  # 防止KeyError

        # 验证路径有效性（起点是否在路径末尾）
        if not path or path[-1] != start_id:
            print(f"[Dijkstra警告] {start_id} → {end_id} 无有效路径")
            return []

        # 反向路径得到正确顺序
        path.reverse()
        total_distance = sum(self.nodes[path[i]].neighbors[path[i + 1]] for i in range(len(path) - 1))
        print(f"[Dijkstra成功] 路径：{' -> '.join(map(str, path))}，总距离：{total_distance}")
        return path

    def shortest_path_length(self, start_id, end_id):
        path = self.dijkstra(start_id, end_id)
        if not path:
            return float('inf')
        total_distance = 0
        for i in range(len(path) - 1):
            total_distance += self.nodes[path[i]].neighbors[path[i + 1]]
        return total_distance




class RescueEnvCore:
    def __init__(self, graph, tasks, rescuers, max_time=200):
        # 救援网络图
        self.graph = graph
        # 初始任务列表的深拷贝
        self.initial_tasks = deepcopy(tasks)
        # 初始救援人员列表的深拷贝
        self.initial_rescuers = deepcopy(rescuers)
        # 最大模拟时间
        self.max_time = max_time
        # 当前任务列表
        self.tasks = []
        # 救援人员列表
        self.rescuers = rescuers
        # 当前时间
        self.current_time = 0
        # 模拟是否结束的标志
        self.done = False
        # 按任务到达时间对初始任务列表进行排序
        self.initial_tasks.sort(key=lambda t: t.arrive_time)
        # 计算总受害者数量
        self.total_victims = sum(task.victim for task in self.initial_tasks)
        # 重置环境
        self.reset()
        self.exclude_tasks = []  # 新增排除任务列表

    def reset(self):
        # 重置环境的状态
        self.tasks = []
        self.rescuers = deepcopy(self.initial_rescuers)
        self.current_time = 0
        self.done = False
        self.exclude_tasks = []  # 重置排除列表
        # 返回当前环境的状态
        return self._get_state()

    def step(self):
        # 执行一个时间步的模拟
        new_tasks = []
        for task in self.initial_tasks:
            if task.arrive_time == self.current_time:
                # 如果任务在当前时间到达，添加到当前任务列表
                self.tasks.append(task)
                new_tasks.append(task)
                print(f"[时间 {self.current_time}] 新任务#{task.task_id} 在节点{task.node_id}")

        # 使用最近任务策略分配任务
        actions = self._nearest_task_strategy_v2()
        # 分配任务给救援人员
        self._assign_actions(actions)

        # 更新每个救援人员的状态
        for rescuer in self.rescuers:
            rescuer.update(self.current_time)

        # 更新每个任务的状态
        for task in self.tasks:
            task.update(self.current_time)

        # 时间推进一个单位
        self.current_time += 1

        print(f"\n=== 时间步 {self.current_time} ===")
        # 打印当前环境的状态
        self.print_status()

        # 判断模拟是否结束
        self.done = self.current_time >= self.max_time
        # 返回当前环境的状态和模拟是否结束的标志
        return self._get_state(), self.done

    def _assign_actions(self, actions):
        # 根据任务分配结果，为救援人员分配任务
        for rescuer_id, task_id in actions.items():
            rescuer = self.rescuers[rescuer_id]
            if rescuer.task_locked:
                print(f"救援人员#{rescuer_id} 任务锁定中")
                continue
            target_task = self._find_task_by_id(task_id)
            if not target_task:
                print(f"任务 {task_id} 未找到")
                continue
            if self.current_time > target_task.deadline:
                print(f"任务 {task_id} 已过期")
                continue
            if not rescuer.busy:
                rescuer.task = target_task
                print(f"分配任务#{task_id} 给救援人员#{rescuer_id}")
                rescuer.move_to_task(target_task.node_id)

    def _find_task_by_id(self, task_id):
        # 根据任务ID查找任务
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        return None

    def _get_state(self):
        # 获取当前环境的状态
        state = []
        for r in self.rescuers:
            current_node = r._get_current_node()
            node = self.graph.nodes[current_node]
            # 添加救援人员的位置、忙碌状态和任务ID到状态列表
            state.extend([node.x, node.y, int(r.busy), r.task.task_id if r.task else -1])
        for t in self.tasks:
            node = self.graph.nodes[t.node_id]
            # 添加任务的位置、已救援人数和剩余时间到状态列表
            state.extend([node.x, node.y, t.rescued_victim, t.deadline - self.current_time])
        return state

    #baseline
    def _nearest_task_strategy_base(self):
        actions = {}
        free_rescuers = [r for r in self.rescuers if not r.busy and not r.task_locked]
        available_tasks = [t for t in self.tasks if t.survivors_not_rescued > 0 and self.current_time <= t.deadline]

        # 打印调试信息
        print(f"\n[任务分配策略] 时间: {self.current_time}")
        print(f"空闲救援人员: {len(free_rescuers)}")
        for r in free_rescuers:
            print(f"  救援人员#{r.rescuer_id} 当前位置: {r.current_position}")

        print(f"可用任务: {len(available_tasks)}")
        for t in available_tasks:
            print(
                f"  任务#{t.task_id} 在节点{t.node_id}, 受害者: {t.survivors_not_rescued}, 截止时间: {t.deadline}")

        # 检查是否有救援人员需要重新分配
        for rescuer in self.rescuers:
            if rescuer.task and (
                    rescuer.task.survivors_not_rescued <= 0 or self.current_time > rescuer.task.deadline):
                print(f"  救援人员#{rescuer.rescuer_id} 任务已完成或过期，重新分配")
                rescuer.task.remove_rescuer(rescuer)
                free_rescuers.append(rescuer)

        task_priority = []
        for task in available_tasks:
            time_left = task.deadline - self.current_time

            # 计算最近救援人员到达所需时间
            required_time = float('inf')
            closest_rescuer = None
            for r in free_rescuers:
                current_node = r._get_current_node()
                dist = self.graph.shortest_path_length(current_node, task.node_id)
                if dist == float('inf'):
                    print(
                        f"  救援人员#{r.rescuer_id} 到任务#{task.task_id} 无路径（节点{current_node}→{task.node_id}）")  # 更详细的路径日志
                else:
                    if dist < required_time:
                        required_time = dist
                        closest_rescuer = r

            # 打印任务优先级计算信息
            if closest_rescuer:
                print(
                    f"  任务#{task.task_id}: 剩余时间={time_left}, 所需时间={required_time}, 救援人员#{closest_rescuer.rescuer_id}")

            if required_time > time_left:
                print(f"  任务#{task.task_id} 无法按时完成，跳过")
                continue

            # 计算紧急度，优先处理时间紧迫且受害者多的任务
            urgency = (task.survivors_not_rescued) / (time_left - required_time + 1e-5)
            print(f"  任务#{task.task_id} 紧急度: {urgency}")
            task_priority.append((-urgency, task.task_id, task, closest_rescuer))

        task_priority.sort()

        # 打印任务优先级排序结果
        print("\n任务优先级排序:")
        for u, tid, _, _ in task_priority:
            print(f"  任务#{tid} 紧急度: {-u}")

        # 为每个任务分配最合适的救援人员
        assigned_rescuers = set()
        for _, _, task, rescuer in task_priority:
            if rescuer and rescuer.rescuer_id not in assigned_rescuers:
                print(f"  分配任务#{task.task_id} 给救援人员#{rescuer.rescuer_id}")
                actions[rescuer.rescuer_id] = task.task_id
                assigned_rescuers.add(rescuer.rescuer_id)

        return actions

    # V1按需分配
    def _nearest_task_strategy_v1(self):
        import math
        actions = {}
        available_tasks = [task for task in self.tasks
                           if task.rescued_victim < task.victim
                           and self.current_time <= task.deadline]
        free_rescuers = [r for r in self.rescuers
                         if not r.busy and not r.task_locked]

        # 步骤1：计算每个任务所需救援能力
        task_needs = {}
        for task in available_tasks:
            remaining_time = task.deadline - self.current_time
            if remaining_time <= 0:
                continue

            # 计算已分配救援人员的有效救援量
            assigned_effective = 0
            for rescuer in task.assigned_rescuers:
                current_node = rescuer.current_node()
                distance = self.graph.shortest_path_length(current_node, task.node_id)
                if distance == float('inf'):
                    continue
                arrival_time = math.ceil(distance / rescuer.speed)
                effective_time = remaining_time - arrival_time
                if effective_time > 0:
                    assigned_effective += rescuer.rescue_capacity * effective_time

            remaining_needed = max(task.survivors_not_rescued - assigned_effective, 0)
            if remaining_needed <= 0:
                continue

            # 收集可用救援人员的有效救援量
            valid_rescuers = []
            for rescuer in free_rescuers:
                current_node = rescuer.current_node()
                distance = self.graph.shortest_path_length(current_node, task.node_id)
                if distance == float('inf'):
                    continue
                arrival_time = math.ceil(distance / rescuer.speed)
                effective_time = remaining_time - arrival_time
                if effective_time <= 0:
                    continue
                effective_rescue = rescuer.rescue_capacity * effective_time
                valid_rescuers.append((effective_rescue, rescuer))

            # 按救援量降序排序
            valid_rescuers.sort(reverse=True, key=lambda x: x[0])

            # 计算所需救援人员数
            required = 0
            current_rescue = 0
            for eff, r in valid_rescuers:
                current_rescue += eff
                required += 1
                if current_rescue >= remaining_needed:
                    break

            task_needs[task] = required

        # 步骤2：任务排序（剩余时间少、需求高优先）
        sorted_tasks = sorted(
            [t for t in available_tasks if task_needs.get(t, 0) > 0],
            key=lambda t: (
                (t.deadline - self.current_time),
                -task_needs[t]
            )
        )

        # 步骤3：收集空闲救援人员（按救援能力降序）
        free_rescuers_sorted = sorted(
            free_rescuers,
            key=lambda r: -r.rescue_capacity
        )

        # 步骤4：分配救援人员
        for task in sorted_tasks:
            needed = task_needs.get(task, 0)
            if needed <= 0 or not free_rescuers_sorted:
                continue

            # 生成候选列表（按到达时间和能力排序）
            candidates = []
            for rescuer in free_rescuers_sorted:
                current_node = rescuer.current_node()
                distance = self.graph.shortest_path_length(current_node, task.node_id)
                if distance == float('inf'):
                    continue
                arrival_time = math.ceil(distance / rescuer.speed)
                candidates.append((
                    arrival_time,
                    -rescuer.rescue_capacity,  # 能力降序
                    rescuer
                ))

            # 按到达时间升序，能力降序排序
            candidates.sort(key=lambda x: (x[0], x[1]))

            # 分配前needed个救援人员
            assigned = 0
            for cand in candidates:
                if assigned >= needed:
                    break
                _, _, rescuer = cand
                if rescuer in free_rescuers_sorted:
                    actions[rescuer.rescuer_id] = task.task_id
                    free_rescuers_sorted.remove(rescuer)
                    assigned += 1

        return actions

    #v2撤离法
    def _nearest_task_strategy_v2(self):
        """基于最短路径的任务分配策略"""
        actions = {}
        self.exclude_tasks = []
        free_rescuers = [r for r in self.rescuers if not r.busy and not r.task_locked]

        # 第一阶段：移除冗余救援人员（保持原逻辑）
        for task in self.tasks:
            if task.survivors_not_rescued <= 0 or \
                    self.current_time >= task.deadline or \
                    task in self.exclude_tasks:
                continue

            current_assigned = sorted(task.assigned_rescuers, key=lambda r: r.rescue_capacity)
            removed = False
            while len(current_assigned) > 1:
                candidate = current_assigned[0]
                remaining = current_assigned[1:]
                time_left = task.deadline - self.current_time
                total_capacity = sum(r.rescue_capacity for r in remaining)

                if total_capacity * time_left >= task.survivors_not_rescued:
                    task.remove_rescuer(candidate)
                    current_assigned = remaining
                    removed = True
                else:
                    break
            if removed:
                self.exclude_tasks.append(task)

        # 第二阶段：生成有效任务列表
        available_tasks = [
            t for t in self.tasks
            if t.survivors_not_rescued > 0 and
               self.current_time < t.deadline and
               t not in self.exclude_tasks
        ]

        # 第三阶段：基于最短路径分配任务
        for rescuer in free_rescuers:
            if not available_tasks:
                break

            current_node = rescuer.current_node()
            min_distance = float('inf')
            nearest_task = None

            # 遍历所有可用任务寻找最近路径
            for task in available_tasks:
                # 跳过不可达任务
                if not self.graph.nodes.get(task.node_id) or \
                        task.node_id not in self.graph.nodes[current_node].neighbors:
                    continue

                # 获取最短路径长度
                path_length = self.graph.shortest_path_length(current_node, task.node_id)

                if path_length < min_distance and path_length != float('inf'):
                    min_distance = path_length
                    nearest_task = task

            if nearest_task:
                actions[rescuer.rescuer_id] = nearest_task.task_id
                # 更新救援人员路径
                rescuer.move_to_task(nearest_task.node_id)
                print(f"Rescuer#{rescuer.rescuer_id} 路径: {rescuer.path}")

        return actions


    def print_status(self):
        # 打印当前环境的状态信息
        rescued = sum(t.rescued_victim for t in self.tasks)
        active_tasks = sum(1 for t in self.tasks if t.rescued_victim < t.victim and self.current_time <= t.deadline)
        print(f"[状态] 时间: {self.current_time}/{self.max_time}")
        print(f"已救援: {rescued}/{self.total_victims}")
        print(f"活跃任务: {active_tasks}, 空闲救援人员: {sum(1 for r in self.rescuers if not r.busy)}")
        print("----")

def generate_random_graph(num_nodes=10, connect_prob=0.05):  # 提高初始边概率
    graph = RescueGraph()
    for i in range(num_nodes):
        x = random.randint(-25, 25)
        y = random.randint(-25, 25)
        graph.add_node(i, x, y)

    # 生成最小生成树确保基础连通性
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            distance = int(np.hypot(graph.nodes[i].x - graph.nodes[j].x, graph.nodes[i].y - graph.nodes[j].y))
            edges.append((i, j, distance))
    edges.sort(key=lambda x: x[2])
    uf = UnionFind(num_nodes)
    edges_added = 0
    for i, j, d in edges:
        if uf.union(i, j):
            graph.add_edge(i, j, d)
            edges_added += 1
            if edges_added == num_nodes - 1:
                break

    # 添加额外边提高连通性
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if j not in graph.nodes[i].neighbors and random.random() < connect_prob:
                distance = int(np.hypot(graph.nodes[i].x - graph.nodes[j].x, graph.nodes[i].y - graph.nodes[j].y))
                graph.add_edge(i, j, distance)

    # 增强孤立节点处理（连接最近节点）
    for node_id in list(graph.nodes.keys()):
        if not graph.nodes[node_id].neighbors:
            # 计算到所有其他节点的距离
            distances = []
            for other_id in graph.nodes:
                if other_id == node_id:
                    continue
                dist = np.hypot(graph.nodes[node_id].x - graph.nodes[other_id].x,
                                graph.nodes[node_id].y - graph.nodes[other_id].y)
                distances.append((other_id, dist))
            # 选择最近的节点连接
            closest_id, closest_dist = min(distances, key=lambda x: x[1])
            graph.add_edge(node_id, closest_id, int(closest_dist))
            print(f"修复孤立节点 {node_id}，连接到 {closest_id}（距离 {closest_dist}）")

    # 强制确保图连通（关键修改）
    while not is_graph_connected(graph):
        print("检测到图不连通，尝试添加连接边...")
        # 使用BFS找到不连通分量
        start_node = next(iter(graph.nodes.keys()))
        visited = set()
        queue = [start_node]
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            for neighbor in graph.nodes[node].neighbors:
                if neighbor not in visited:
                    queue.append(neighbor)
        # 找到未访问的节点
        unvisited = [n for n in graph.nodes if n not in visited]
        if not unvisited:
            break
        # 随机连接两个分量的节点
        u = random.choice(list(visited))
        v = random.choice(unvisited)
        distance = int(np.hypot(graph.nodes[u].x - graph.nodes[v].x, graph.nodes[u].y - graph.nodes[v].y))
        graph.add_edge(u, v, distance)
        print(f"添加跨分量边 {u}-{v}（距离 {distance}）")

    return graph

def is_graph_connected(graph):
    if not graph.nodes:
        return True
    start_node = next(iter(graph.nodes.keys()))
    visited = set()
    queue = [start_node]
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        for neighbor in graph.nodes[node].neighbors:
            if neighbor not in visited:
                queue.append(neighbor)
    return len(visited) == len(graph.nodes)


def generate_random_tasks(n, graph):
    tasks = []
    node_ids = list(graph.nodes.keys())
    for i in range(n):
        arrive_time = random.randint(0, 60)
        node_id = random.choice(node_ids)
        victim = random.randint(20, 100)
        deadline = arrive_time + 4500 // victim
        tasks.append(Task(i, arrive_time, node_id, victim, deadline))
    return tasks

# 初始化救援人员的函数
def init_rescuers(num_rescuers, graph, speed):
    rescuers = []
    node_ids = list(graph.nodes.keys())
    for i in range(num_rescuers):
        start_node = random.choice(node_ids)
        start_position = ('node', start_node)  # 起始位置为随机节点
        rescue_capacity = random.randint(1, 3)  # 救援能力在1-3之间随机生成
        rescuers.append(Rescuer(i, start_position, rescue_capacity, graph, speed))
    return rescuers

def Visualize(env):
    """优化后的图结构可视化函数"""
    # 初始化图形界面
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(16, 8))
    grid = plt.GridSpec(1, 2, width_ratios=[3, 1])

    # 创建子图
    ax_main = fig.add_subplot(grid[0, 0])  # 主视图
    ax_progress = fig.add_subplot(grid[0, 1])  # 进度视图

    # ================== 图结构初始化 ==================
    # 获取所有节点坐标
    node_coords = {nid: (node.x, node.y) for nid, node in env.graph.nodes.items()}
    node_x = [x for x, _ in node_coords.values()]
    node_y = [y for _, y in node_coords.values()]

    # 绘制所有边（静态元素）
    for node in env.graph.nodes.values():
        for neighbor_id in node.neighbors:
            neighbor = env.graph.nodes[neighbor_id]
            ax_main.plot([node.x, neighbor.x], [node.y, neighbor.y],
                         color='grey', linewidth=0.8, alpha=0.4, zorder=1)

    # ================== 动态元素初始化 ==================
    # 救援人员（三角形标记）
    rescuer_scatter = ax_main.scatter(
        [], [],
        c='red',  # 初始颜色
        s=100,  # 标记大小
        marker='^',  # 三角形标记
        edgecolors='k',  # 黑色边框
        zorder=4  # 显示层级
    )

    # 任务（方形标记）
    task_scatter = ax_main.scatter(
        [], [],
        c='orange',  # 初始颜色
        s=120,  # 标记大小
        marker='s',  # 方形标记
        edgecolors='k',  # 黑色边框
        zorder=3  # 显示层级
    )

    # ================== 进度图初始化 ==================
    progress_line, = ax_progress.plot([], [], 'b-', lw=2, label='实时进度')
    target_line = ax_progress.axhline(
        env.total_victims,
        color='r',
        linestyle='--',
        label='总目标'
    )

    # ================== 坐标范围设置 ==================
    ax_main.set(
        xlim=(min(node_x) - 5, max(node_x) + 5),
        ylim=(min(node_y) - 5, max(node_y) + 5),
        xlabel="X 坐标",
        ylabel="Y 坐标",
        title="实时救援路径网络"
    )

    ax_progress.set(
        xlim=(0, env.max_time),
        ylim=(0, env.total_victims * 1.2),
        xlabel="Time Step",
        ylabel="Number of the Rescued",
        title="Rescue Process Tracking"
    )
    ax_progress.legend(loc='lower right')
    ax_progress.grid(True, alpha=0.3)

    # ================== 数据追踪器 ==================
    rescue_data = {
        'time': [],
        'rescued': [],
        'active_tasks': [],
        'idle_rescuers': []
    }

    # ================== 动画更新函数 ==================
    def update(frame):
        if env.done:
            return

        # 执行模拟步骤
        state, done = env.step()

        # === 救援人员位置更新 ===
        rescuer_pos = []
        rescuer_colors = []
        for r in env.rescuers:
            # 位置计算逻辑
            if r.current_position[0] == 'node':
                node = env.graph.nodes[r.current_position[1]]
                x, y = node.x, node.y
            else:  # 处理边上的移动
                u, v, d = r.current_position[1], r.current_position[2], r.current_position[3]
                start_node = env.graph.nodes[u]
                end_node = env.graph.nodes[v]
                total_dist = start_node.neighbors[v]
                ratio = d / total_dist
                x = start_node.x + (end_node.x - start_node.x) * ratio
                y = start_node.y + (end_node.y - start_node.y) * ratio

            # 转换为NumPy数组确保维度正确
            rescuer_pos.append([x, y])
            rescuer_colors.append('red' if r.busy else 'limegreen')

        # 转换为二维数组（重要修复点）
        rescuer_pos = np.array(rescuer_pos).reshape(-1, 2) if rescuer_pos else np.empty((0, 2))
        rescuer_colors = np.array(rescuer_colors, dtype=object)

        # === 任务状态更新 ===
        active_tasks = [t for t in env.tasks
                        if t.rescued_victim < t.victim and env.current_time <= t.deadline]

        # 确保二维数组结构（关键修复）
        if active_tasks:
            task_pos = np.array([[env.graph.nodes[t.node_id].x,
                                  env.graph.nodes[t.node_id].y]
                                 for t in active_tasks])
        else:
            task_pos = np.empty((0, 2))  # 创建空的二维数组

        # 确保颜色数组维度匹配
        task_colors = np.array(
            ['gold' if env.current_time <= t.deadline else 'gray' for t in active_tasks],
            dtype=object
        )

        # === 可视化元素更新 ===
        rescuer_scatter.set_offsets(rescuer_pos)
        rescuer_scatter.set_color(rescuer_colors)
        task_scatter.set_offsets(task_pos)
        task_scatter.set_color(task_colors)

        # === 数据记录 ===
        current_rescued = sum(t.rescued_victim for t in env.tasks)
        rescue_data['time'].append(env.current_time)
        rescue_data['rescued'].append(current_rescued)
        rescue_data['active_tasks'].append(len(active_tasks))
        rescue_data['idle_rescuers'].append(sum(1 for r in env.rescuers if not r.busy))

        # === 进度曲线更新 ===
        progress_line.set_data(rescue_data['time'], rescue_data['rescued'])

        # === 标题更新 ===
        ax_main.set_title(
            f"(Total Rescued:{current_rescued}/{env.total_victims} | "
            f"Progress:{current_rescued / env.total_victims * 100:.1f}%)"
        )

        return rescuer_scatter, task_scatter, progress_line

    # ================== 图例配置 ==================
    legend_elements = [
        Line2D([0], [0], marker='^', color='w', label='rescuer(busy)',
               markerfacecolor='red', markersize=12),
        Line2D([0], [0], marker='^', color='w', label='rescuer(free)',
               markerfacecolor='limegreen', markersize=12),
        Line2D([0], [0], marker='s', color='w', label='active task',
               markerfacecolor='gold', markersize=12),
        Line2D([0], [0], marker='s', color='w', label='overdue task',
               markerfacecolor='gray', markersize=12)
    ]
    ax_main.legend(
        handles=legend_elements,
        loc='upper right',
        framealpha=0.9
    )

    # ================== 运行动画 ==================
    ani = FuncAnimation(
        fig,
        update,
        frames=env.max_time,
        interval=200,
        blit=False,
        repeat=False,
        save_count=env.max_time  # 关键参数：确保帧数正确
    )

    # 保存动画（使用Pillow写入器）
    try:
        ani.save(
            'rescue_simulation.gif',
            writer='pillow',  # 改用更可靠的写入器
            fps=5,
            dpi=100,
            progress_callback=lambda i, n: print(f'\r保存进度: {i + 1}/{n}', end='')
        )
        print("\n动画保存成功!")
    except Exception as e:
        print(f"\n保存失败: {str(e)}")
        print("请尝试安装Pillow库:pip install pillow")


if __name__ == "__main__":
    # 生成连通图（循环直到成功）
    graph = None
    while True:
        graph = generate_random_graph(num_nodes=25)
        if is_graph_connected(graph):
            print("成功生成连通图！")
            break
        print("图不连通，重新生成...")

    tasks = generate_random_tasks(10, graph)
    rescuers = init_rescuers(10, graph, speed=1)
    env = RescueEnvCore(graph, tasks, rescuers, max_time=200)
    Visualize(env)