import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

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
        c='red',        # 初始颜色
        s=100,          # 标记大小
        marker='^',     # 三角形标记
        edgecolors='k', # 黑色边框
        zorder=4        # 显示层级
    )
    
    # 任务（方形标记）
    task_scatter = ax_main.scatter(
        [], [], 
        c='orange',     # 初始颜色
        s=120,          # 标记大小 
        marker='s',     # 方形标记
        edgecolors='k', # 黑色边框
        zorder=3        # 显示层级
    )

    # ================== 进度图初始化 ==================
    progress_line, = ax_progress.plot([], [], 'b-', lw=2, label='Victims Rescued')
    target_line = ax_progress.axhline(
        env.total_victims, 
        color='r', 
        linestyle='--', 
        label='Total Target'
    )
    
    # ================== 坐标范围设置 ==================
    ax_main.set(
        xlim=(min(node_x)-5, max(node_x)+5),
        ylim=(min(node_y)-5, max(node_y)+5),
        xlabel="X",
        ylabel="Y",
        title="Real-Time Urban Rescue Network"
    )

    ax_progress.set(
        xlim=(0, env.max_time),
        ylim=(0, env.total_victims*1.2),
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
            f"Progress:{current_rescued/env.total_victims*100:.1f}%)"
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
            progress_callback=lambda i, n: print(f'\r保存进度: {i+1}/{n}', end='')
        )
        print("\n动画保存成功!")
    except Exception as e:
        print(f"\n保存失败: {str(e)}")
        print("请尝试安装Pillow库:pip install pillow")

    # plt.tight_layout()
    # plt.show()
