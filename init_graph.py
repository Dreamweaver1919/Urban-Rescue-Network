from environment_graph import RescueGraph,UnionFind
import random
import numpy as np
from task_graph import Task
from rescuer_graph import Rescuer



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
