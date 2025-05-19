from init_graph import generate_random_graph, is_graph_connected,generate_random_tasks, init_rescuers
from environment_graph import RescueEnvCore
from visualizations_graph import Visualize


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
