import random
from copy import deepcopy
import numpy as np
import matplotlib.animation as animation
import json

SEED = 42  
random.seed(SEED)
np.random.seed(SEED)

# 定义 Rescuer 类，用于表示救援人员
class Rescuer:
    def __init__(self, rescuer_id, x, y, efficiency=1):
        self.rescuer_id = rescuer_id
        self.x = x
        self.y = y
        # 救援人员的效率（每步可以救援的人数）
        self.efficiency = efficiency
        self.direction_x = True
        self.direction_y = True
        self.moving_x = False
        self.moving_y = False
        self.busy = False
        self.task = None
        self.moved_distance = 0
        self.participated_tasks = []
        self.total_distance = 0
        self.task_locked = False

    def move(self, destination_x, destination_y):
        self.total_distance = abs(self.x - destination_x) + abs(self.y - destination_y)
        self.moved_distance = 0
        self.task_locked = False
        self.direction_x = destination_x > self.x
        self.direction_y = destination_y > self.y
        self.moving_x = (self.x != destination_x)
        self.moving_y = (self.y != destination_y)

    def update(self, current_time):
        # 如果当前任务为空或超时，释放救援人员
        if self.task is None or current_time >= self.task.deadline or self.task.survivors_not_rescued == 0:
            self.task_locked = False
            self.busy = False
            self.task = None
            self.moved_distance = 0
            self.total_distance = 0
            self.moving_x = False
            self.moving_y = False
            return

        # 移动逻辑
        dx = self.task.x - self.x
        dy = self.task.y - self.y
        if dx != 0:
            self.moving_x = True
            self.direction_x = dx > 0
            self.moving_y = False
        elif dy != 0:
            self.moving_y = True
            self.direction_y = dy > 0
            self.moving_x = False

        prev_x, prev_y = self.x, self.y
        if self.moving_x:
            self.x += 1 if self.direction_x else -1
        elif self.moving_y:
            self.y += 1 if self.direction_y else -1
        self.moved_distance += abs(self.x - prev_x) + abs(self.y - prev_y)

        # 锁定任务，避免中途切换
        if self.task and not self.task_locked and self.total_distance > 0 and self.moved_distance > self.total_distance * 0.2:
            self.task_locked = True

        # 到达后开始执行任务
        if self.task and self.x == self.task.x and self.y == self.task.y and not self.busy:
            self.task.add_rescuer(self)
            self.move(self.x, self.y)

    def __repr__(self):
        return (f"Rescuer#{self.rescuer_id} busy on task {self.task}." if self.busy else f"Rescuer#{self.rescuer_id} free.")


# 定义 Task 类，用于表示救援任务，加入离散化 Weibull 衰减模型和效率累积
class Task:
    def __init__(self, task_id, arrive_time, x, y, initial_victim, deadline, weibull_shape=2.0):
        self.task_id = task_id
        self.arrive_time = arrive_time
        self.x = x
        self.y = y
        self.initial_victim = initial_victim
        # 当前存活且未获救的幸存者数
        self.survivors_not_rescued = initial_victim
        self.rescued_victim = 0
        self.deadline = deadline
        self.assigned_rescuers = []
        # Weibull 衰减参数
        self.weibull_shape = weibull_shape
        T = max(1, self.deadline - self.arrive_time)
        self.weibull_scale = T / (np.log(10) ** (1.0 / self.weibull_shape))

    def add_rescuer(self, rescuer):
        if rescuer not in self.assigned_rescuers:
            self.assigned_rescuers.append(rescuer)
            rescuer.busy = True
            rescuer.task = self
            if self not in rescuer.participated_tasks:
                rescuer.participated_tasks.append(self)

    def remove_rescuer(self, rescuer):
        if rescuer in self.assigned_rescuers:
            self.assigned_rescuers.remove(rescuer)
        rescuer.busy = False
        rescuer.task = None
        rescuer.moved_distance = 0
        rescuer.total_distance = 0
        rescuer.task_locked = False
        rescuer.moving_x = False
        rescuer.moving_y = False

    def update(self, current_time):
        # 1. 离散 Weibull 衰减（死亡），至少保留 1 人
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

        # 2. 执行救援：根据到达现场的救援人员效率累积
        if current_time <= self.deadline:
            total_eff = sum(r.efficiency for r in self.assigned_rescuers)
            to_rescue = min(total_eff, self.survivors_not_rescued)
            self.rescued_victim += to_rescue
            self.survivors_not_rescued -= to_rescue

        # 3. 若无幸存者或超时，则释放所有救援人员
        if self.survivors_not_rescued <= 0 or current_time >= self.deadline:
            for r in self.assigned_rescuers[:]:
                self.remove_rescuer(r)

    def __repr__(self):
        return (f"Task#{self.task_id}@{self.arrive_time} survivors={self.survivors_not_rescued}, "
                f"rescued={self.rescued_victim}, deadline={self.deadline}")


# 随机生成指定数量的任务
def generate_random_tasks(n):
    tasks = []
    for i in range(n):
        arrive_time = random.randint(0, 60)
        x = random.randint(-20, 20)
        y = random.randint(-20, 20)
        victim = random.randint(20, 100)
        deadline = arrive_time + 4500 // victim
        tasks.append(Task(i, arrive_time, x, y, victim, deadline))
    return tasks


# 初始化指定数量的救援人员
def init_rescuers(n):
    rescuers = []
    for i in range(n):
        x = random.randint(-20, 20)
        y = random.randint(-20, 20)
        efficiency = random.randint(1, 3)
        rescuers.append(Rescuer(i, x, y, efficiency))
    return rescuers


# 定义环境类，用于模拟救援环境
class RescueEnvCore:
    def __init__(self, tasks, rescuers, max_time=200):
        self.initial_tasks = deepcopy(tasks)
        self.initial_rescuers = deepcopy(rescuers)
        self.max_time = max_time
        self.tasks = []
        self.rescuers = rescuers
        self.current_time = 0
        self.done = False
        self.initial_tasks.sort(key=lambda t: t.arrive_time)
        self.total_victims = sum(t.initial_victim for t in tasks)
        self.exclude_tasks = []
        self.reset()

    def reset(self):
        self.tasks = []
        self.rescuers = deepcopy(self.initial_rescuers)
        self.current_time = 0
        self.done = False
        self.exclude_tasks = []
        return self._get_state()

    def step(self):
        for task in self.initial_tasks:
            if task.arrive_time == self.current_time:
                self.tasks.append(task)
                print(f"Time step {self.current_time}: Task#{task.task_id} arrived.")
            if task.deadline == self.current_time:
                for r in self.initial_rescuers:
                    if r.task == task:
                        r.busy = False
        
        actions = self._nearest_task_strategy()
        self._assign_actions(actions)

        for r in self.rescuers:
            r.update(self.current_time)
        for t in self.tasks:
            t.update(self.current_time)
        self.current_time += 1
        self.done = self.current_time >= self.max_time
        return self._get_state(), self.done

    def _assign_actions(self, actions):
        for rescuer_id, task_id in actions.items():
            rescuer = self.rescuers[rescuer_id]
            if rescuer.task and rescuer.task_locked:
                continue
            target = self._find_task_by_id(task_id)
            if not target or self.current_time > target.deadline:
                continue
            if not rescuer.busy:
                rescuer.task = target
                rescuer.move(target.x, target.y)
            else:
                if rescuer.task.task_id != task_id and not rescuer.task_locked:
                    rescuer.task.remove_rescuer(rescuer)
                    rescuer.task = target
                    rescuer.move(target.x, target.y)

    def _find_task_by_id(self, task_id):
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        return None

    def _nearest_task_strategy(self):
        actions = {}
        for task in self.tasks:
            if task.survivors_not_rescued <= 0 or self.current_time >= task.deadline or task in self.exclude_tasks:
                continue
            current = sorted(task.assigned_rescuers, key=lambda r: r.efficiency)
            removed = False
            while len(current) > 1:
                cand = current[0]
                rest = current[1:]
                time_left = task.deadline - self.current_time 
                total_eff = sum(r.efficiency for r in rest)
                if total_eff * time_left >= task.survivors_not_rescued:
                    task.remove_rescuer(cand)
                    current = rest
                    removed = True
                else:
                    break
            if removed and task not in self.exclude_tasks:
                self.exclude_tasks.append(task)
        available = [t for t in self.tasks if t.survivors_not_rescued > 0 and self.current_time < t.deadline and t not in self.exclude_tasks]

        # 3. 计算任务优先级
        if available:
            # 计算每个任务的紧急度（剩余幸存者 / 剩余时间）
            task_priority = []
            for task in available:
                time_left = task.deadline - self.current_time
                urgency = task.survivors_not_rescued / (time_left + 1e-5)  # 避免除以0
                task_priority.append((urgency, task.task_id, task))
        
            # 按紧急度降序排序（最紧急的任务优先）
            task_priority.sort(reverse=True, key=lambda x: x[0])

            # 4. 按优先级分配救援人员（修改部分）
            free_rescuers = [r for r in self.rescuers if not r.busy]
            for _, task_id, task in task_priority:
                if not free_rescuers:
                    break
                # 找出离该任务最近的空闲救援人员
                nearest_rescuer = min(
                    free_rescuers,
                    key=lambda r: abs(r.x - task.x) + abs(r.y - task.y)
                )
                actions[nearest_rescuer.rescuer_id] = task_id
                free_rescuers.remove(nearest_rescuer)  # 标记为已分配

        return actions
    def _get_state(self):
        state = []
        for r in self.rescuers:
            state.extend([r.x, r.y, int(r.busy), r.task.task_id if r.task else -1])
        for t in self.tasks:
            state.extend([t.x, t.y, t.survivors_not_rescued, t.deadline - self.current_time])
        return state

    def print_status(self):
        if self.done:
            total_rescued = sum(t.rescued_victim for t in self.tasks)
            ratio = f"{total_rescued/self.total_victims*100:.2f}%" if self.total_victims>0 else "0.00%"
            print(f"Total rescued / Total victims: {total_rescued}/{self.total_victims} ({ratio})")

    def _get_plot_data(self):
        rescuers_x = [r.x for r in self.rescuers]
        rescuers_y = [r.y for r in self.rescuers]
        colors = ['red' if r.busy else 'green' for r in self.rescuers]
        tasks_x, tasks_y, task_colors, task_sizes = [], [], [], []
        for t in self.tasks:
            if self.current_time > t.deadline:
                task_colors.append('gray')
            elif t.rescued_victim >= t.initial_victim:
                task_colors.append('blue')
            else:
                task_colors.append('orange')
            tasks_x.append(t.x)
            tasks_y.append(t.y)
            task_sizes.append(50 + t.survivors_not_rescued * 2)
        return (np.array(rescuers_x), np.array(rescuers_y), np.array(colors),
                np.array(tasks_x), np.array(tasks_y), np.array(task_colors), np.array(task_sizes))

def Visualize(env):
    """相当于可视化的总函数，在 main 中调用 Visualize(env=env)"""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plt.subplots_adjust(bottom=0.2)

    rescuer_scatter = ax1.scatter([], [], c=[], cmap='RdYlGn', s=100, label='Rescuers')
    task_scatter = ax1.scatter([], [], c=[], s=[], alpha=0.6, cmap='viridis', label='Tasks')

    ax1.set_xlim(-100, 100)
    ax1.set_ylim(-100, 100)
    ax1.grid(True)
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.set_title('Rescue Operation Visualization')

    status_text = ax1.text(0.02, 0.25, '', transform=ax1.transAxes, fontsize=10)
    time_text = ax1.text(0.02, 0.20, '', transform=ax1.transAxes, fontsize=10)

    progress_line, = ax2.plot([], [], 'b-', label='Rescued')
    from matplotlib.collections import LineCollection
    deadline_lines = LineCollection([], colors='gray', linestyles='--', alpha=0.3, label='Task Deadlines')
    ax2.add_collection(deadline_lines)
    ax2.set_xlim(0, env.max_time)
    ax2.set_ylim(0, 1000)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Victims Rescued')
    ax2.set_title('Rescue Progress Over Time')
    ax2.grid(True)
    ax2.legend()

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Free Rescuer',
                   markerfacecolor='green', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Busy Rescuer',
                   markerfacecolor='red', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Active Task',
                   markerfacecolor='orange', markersize=10, alpha=0.6),
        plt.Line2D([0], [0], marker='o', color='w', label='Completed Task',
                   markerfacecolor='blue', markersize=10, alpha=0.6),
        plt.Line2D([0], [0], marker='o', color='w', label='Missed Task',
                   markerfacecolor='gray', markersize=10, alpha=0.6)
    ]
    ax1.legend(handles=legend_elements, bbox_to_anchor=(0, 1), loc='upper left')

    progress_legend = [
        plt.Line2D([0], [0], color='blue', lw=2, label='Rescued Victims'),
        plt.Line2D([0], [0], color='red', lw=2, linestyle='--', label='Total Target'),
        plt.Line2D([0], [0], color='gray', lw=1, alpha=0.3, label='Task Deadlines')
    ]
    ax2.legend(handles=progress_legend, loc='upper left')
    progress_line, = ax2.plot([], [], 'b-', label='Rescued Victims')
    total_target_line, = ax2.plot([], [], 'r--', label='Total Target')

    def update(frame, env, rescuer_scatter, task_scatter, status_text, time_text, progress_line, total_target_line):
        import numpy as np
        state, done = env.step()

        r_x, r_y, r_c, t_x, t_y, t_c, t_s = env._get_plot_data()

        if len(t_x) == 0:
            task_offsets = np.empty((0, 2))
        else:
            task_offsets = np.column_stack([t_x, t_y])

        rescuer_scatter.set_offsets(np.column_stack([r_x, r_y]))
        rescuer_scatter.set_color(r_c)

        task_scatter.set_offsets(task_offsets)
        task_scatter.set_color(t_c)
        task_scatter.set_sizes(t_s)

        rescued = sum(t.rescued_victim for t in env.tasks)
        status_text.set_text(f'Rescued: {rescued}/{env.total_victims}')
        time_text.set_text(f'Time: {env.current_time}/{env.max_time}')

        progress_line.set_data(range(env.current_time + 1),
                               [sum(t.rescued_victim for t in env.tasks[:i + 1])
                                for i in range(env.current_time + 1)])

        total_victims = env.total_victims
        total_target_line.set_data([0, env.max_time], [total_victims, total_victims])

        current_rescued = sum(t.rescued_victim for t in env.tasks)
        progress_line.set_data(range(env.current_time + 1),
                               [sum(t.rescued_victim for t in env.tasks[:i + 1])
                                for i in range(env.current_time + 1)])

        segments = []
        for task in env.tasks:
            segments.append([(task.deadline, 0), (task.deadline, env.total_victims)])
        deadline_lines.set_segments(segments)

        if done:
            ani.event_source.stop()
            env.print_status()

        return (rescuer_scatter, task_scatter, status_text,
                time_text, progress_line, total_target_line, deadline_lines)

    ani = animation.FuncAnimation(fig, update, frames=env.max_time,
                                  interval=200, blit=False, repeat=False, fargs=(
            env, rescuer_scatter, task_scatter, status_text, time_text, progress_line, total_target_line))

    ani.save('Decay_Evacuate.gif', writer='pillow', fps=5, dpi=100)


# 读取保存的模拟数据
def load_simulation_data(filename):
    """读取保存的模拟数据"""
    with open(filename, "r") as f:
        data = json.load(f)
    # 创建任务列表
    tasks = [Task(t['task_id'], t['arrive_time'], t['x'], t['y'], t['victim'], t['deadline']) for t in data['tasks']]
    # 创建救援人员列表
    rescuers = [Rescuer(r['rescuer_id'], r['x'], r['y'], r['efficiency']) for r in data['rescuers']]
    
    return tasks, rescuers


# 示例运行
if __name__ == "__main__":
    """
    num_tasks = 50
    num_rescuers = 80
    tasks = generate_random_tasks(num_tasks)
    rescuers = init_rescuers(num_rescuers)
    """
    tasks, rescuers =load_simulation_data('/Users/zhkevin/Desktop/Urban-Rescue-Network/data2.json')
    env = RescueEnvCore(tasks, rescuers)
    state = env.reset()
    # Visualize(env=env)
    done = False
    while not done:
        state, done = env.step()
    env.print_status()
