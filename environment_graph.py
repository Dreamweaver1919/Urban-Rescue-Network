import random
import heapq
from copy import deepcopy

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

    def reset(self):
        # 重置环境的状态
        self.tasks = []
        self.rescuers = deepcopy(self.initial_rescuers)
        self.current_time = 0
        self.done = False
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
        actions = self._nearest_task_strategy()
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

    def _nearest_task_strategy(self):
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

    def print_status(self):
        # 打印当前环境的状态信息
        rescued = sum(t.rescued_victim for t in self.tasks)
        active_tasks = sum(1 for t in self.tasks if t.rescued_victim < t.victim and self.current_time <= t.deadline)
        print(f"[状态] 时间: {self.current_time}/{self.max_time}")
        print(f"已救援: {rescued}/{self.total_victims}")
        print(f"活跃任务: {active_tasks}, 空闲救援人员: {sum(1 for r in self.rescuers if not r.busy)}")
        print("----")