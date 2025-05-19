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
        # 救援人员的唯一标识符
        self.rescuer_id = rescuer_id
        # 救援人员当前的 x 坐标
        self.x = x
        # 救援人员当前的 y 坐标
        self.y = y
        # x 方向的移动方向，True 表示正向，False 表示负向
        self.direction_x = True
        # y 方向的移动方向，True 表示正向，False 表示负向
        self.direction_y = True
        # 是否正在 x 方向移动的标志
        self.moving_x = False
        # 是否正在 y 方向移动的标志
        self.moving_y = False
        # 救援人员是否忙碌的标志
        self.busy = False
        # 救援人员当前正在执行的任务
        self.task = None
        # 救援人员移动的总距离（累计）
        self.moved_distance = 0
        # 救援人员参与过的任务列表
        self.participated_tasks = []
        # 新增加：当被分配新任务时，记录从当前位置到任务的总距离
        self.total_distance = 0
        # 新增加：任务锁定标志，若任务被锁定则在到达任务地点前不允许更换任务
        self.task_locked = False
        self.efficiency = efficiency  # 单位时间可救援的受害者数量

    def move(self, destination_x, destination_y):
        # 当分配新的任务时，重新计算 total_distance，并重置 moved_distance 和任务锁定状态
        self.total_distance = abs(self.x - destination_x) + abs(self.y - destination_y)
        self.moved_distance = 0
        self.task_locked = False

        # 根据目标 x 坐标和当前 x 坐标确定 x 方向的移动方向
        self.direction_x = True if destination_x > self.x else False
        # 根据目标 y 坐标和当前 y 坐标确定 y 方向的移动方向
        self.direction_y = True if destination_y > self.y else False

        # 如果当前 x 坐标不等于目标 x 坐标，则需要在 x 方向移动
        if self.x != destination_x:
            self.moving_x = True
        else:
            self.moving_x = False

        # 如果当前 y 坐标不等于目标 y 坐标，则需要在 y 方向移动
        if self.y != destination_y:
            self.moving_y = True
        else:
            self.moving_y = False

    def update(self, current_time):

        # 如果当前任务为空或超时，则解锁并清空任务
        if self.task is None or current_time >= self.task.deadline:
            self.task_locked = False
            self.busy = False
            self.task = None
            self.moved_distance = 0
            self.total_distance = 0
            # 清除移动标志，保持静止等待新任务
            self.moving_x = False
            self.moving_y = False
            return

        # 动态更新移动方向（即使已锁定，确保方向正确）
        if self.task:
            destination_x, destination_y = self.task.x, self.task.y
            dx = destination_x - self.x
            dy = destination_y - self.y

            if dx != 0:
                self.moving_x = True
                self.direction_x = dx > 0
                self.moving_y = False
            elif dy != 0:
                self.moving_y = True
                self.direction_y = dy > 0
                self.moving_x = False

        # 记录上一时刻的位置
        prev_x = self.x
        prev_y = self.y

        if self.moving_x:
            self.x += 1 if self.direction_x else -1
        elif self.moving_y:
            self.y += 1 if self.direction_y else -1

        self.moved_distance += abs(self.x - prev_x) + abs(self.y - prev_y)

        # 使用固定的 total_distance 判断是否锁定任务
        if self.task is not None and not self.task_locked and self.total_distance > 0 and self.moved_distance > self.total_distance * 0.1:
            self.task_locked = True

        # 如果到达任务地点且不忙，则开始执行任务
        if self.task and self.x == self.task.x and self.y == self.task.y and not self.busy:
            self.task.add_rescuer(self)
            # 到达目标后，重置移动状态
            self.move(self.x, self.y)

    def __repr__(self):
        # 返回救援人员的状态信息
        return (f"Rescuer#{self.rescuer_id} busy on task {self.task}."
                if self.busy else f"Rescuer#{self.rescuer_id} free.")


# 定义 Task 类，用于表示救援任务,加入离散化 Weibull 衰减模型和效率累积
class Task:
    def __init__(self, task_id, arrive_time, x, y, initial_victim, deadline, weibull_shape=2.0):
        
        # 任务的唯一标识符
        self.task_id = task_id
        # 任务的到达时间
        self.arrive_time = arrive_time
        # 任务的 x 坐标
        self.x = x
        # 任务的 y 坐标
        self.y = y
        # 初始被困人数
        self.initial_victim = initial_victim
        # 当前存活且未获救的幸存者数
        self.survivors_not_rescued = initial_victim
        # 已经救出的人数
        self.rescued_victim = 0
        # 任务的截止时间
        self.deadline = deadline
        # 分配给该任务的救援人员列表
        self.assigned_rescuers = []
        # Weibull 衰减参数
        self.weibull_shape = weibull_shape
        T = max(1, self.deadline - self.arrive_time)
        self.weibull_scale = T / (np.log(10) ** (1.0 / self.weibull_shape))

    def add_rescuer(self, rescuer):
        # 如果救援人员还未分配到该任务，则将其添加到任务的救援人员列表中
        if rescuer not in self.assigned_rescuers:
            self.assigned_rescuers.append(rescuer)
            # 标记救援人员为忙碌状态
            rescuer.busy = True
            # 将该任务赋值给救援人员的当前任务
            rescuer.task = self
            # 如果该任务不在救援人员参与过的任务列表中，则添加进去
            if self not in rescuer.participated_tasks:
                rescuer.participated_tasks.append(self)

    def remove_rescuer(self, rescuer):
        # 如果救援人员在任务的救援人员列表中，则将其移除
        if rescuer in self.assigned_rescuers:
            self.assigned_rescuers.remove(rescuer)
        # 标记救援人员为空闲状态
        rescuer.busy = False
        # 将救援人员的当前任务置为 None，并重置移动距离信息（确保下次分配新任务时计算正确）
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
            # 计算当前分配的救援人员总能力（原逻辑为len(assigned_rescuers)，现为总能力）
            '''total_Efficiency = sum(rescuer.rescue_efficiency for rescuer in self.assigned_rescuers)
            self.rescued_victim += total_Efficiency  # 按总能力累加
            if self.rescued_victim > self.victim:
                self.rescued_victim = self.victim'''
            # 仅计算已到达的救援人员                                
            total_eff = sum(r.efficiency for r in self.assigned_rescuers)
            to_rescue = min(total_eff, self.survivors_not_rescued)
            self.rescued_victim += to_rescue
            self.survivors_not_rescued -= to_rescue

        # 任务完成或超时后释放救援人员（逻辑不变）
        if self.survivors_not_rescued <= 0 or current_time >= self.deadline:
            for r in self.assigned_rescuers[:]:
                self.remove_rescuer(r)

    def __repr__(self):
        # 返回任务的状态信息
        return (f"Task#{self.task_id} arrived at {self.arrive_time}, "
                f"rescued_victim={self.rescued_victim}/{self.victim}")


# 随机生成指定数量的任务
def generate_random_tasks(n):
    tasks = []
    for i in range(n):
        # 随机生成任务的到达时间，范围是 0 到 60
        arrive_time = random.randint(0, 60)
        # 随机生成任务的 x 坐标，范围是 -20 到 20
        x = random.randint(-75, 75)
        # 随机生成任务的 y 坐标，范围是 -20 到 20
        y = random.randint(-75, 75)
        # 随机生成任务中被困的人数，范围是 20 到 100
        victim = random.randint(20, 100)
        # 随机生成任务的截止时间，截止时间是到达时间加上一个随机值
        # deadline = arrive_time + victim // 5 + random.randint(30, 60)
        deadline = arrive_time + 4500 // victim
        # 创建一个任务对象
        task = Task(i, arrive_time, x, y, victim, deadline)
        # 将任务对象添加到任务列表中
        tasks.append(task)
    # 可以按照任务的到达时间对任务列表进行排序
    # tasks.sort(key=lambda task: task.arrive_time)
    return tasks


# 初始化指定数量的救援人员
def init_rescuers(n):
    rescuers = []
    for i in range(n):
        x = random.randint(-75, 75)
        y = random.randint(-75, 75)
        # 随机生成救援能力（1-5）
        efficiency = random.randint(1, 3)
        rescuer = Rescuer(i, x, y, efficiency)  # 传递能力值
        rescuers.append(rescuer)
    return rescuers


# 定义环境类，用于模拟救援环境
class RescueEnvCore:
    def __init__(self, tasks, rescuers, max_time=200, seed=42):
        self.rng = np.random.RandomState(seed)
        # 保存初始的任务列表，用于重置环境
        self.initial_tasks = deepcopy(tasks)
        # 保存初始的救援人员列表，用于重置环境
        self.initial_rescuers = deepcopy(rescuers)
        # 模拟的最大时间步数
        self.max_time = max_time
        # 当前的任务列表
        self.tasks = []
        # 当前的救援人员列表
        self.rescuers = rescuers
        # 当前的时间步数
        self.current_time = 0
        # 模拟是否结束的标志
        self.done = False
        # 按照任务的到达时间对任务列表进行排序
        self.initial_tasks.sort(key=lambda t: t.arrive_time)
        # 计算所有任务中被困的总人数
        self.total_victims = sum(task.initial_victim for task in tasks)
        # 重置环境
        self.reset()

    def reset(self):
        # 恢复初始的任务列表
        self.tasks = []
        # 恢复初始的救援人员列表
        self.rescuers = deepcopy(self.initial_rescuers)
        # 将当前时间步数重置为 0
        self.current_time = 0
        # 将模拟结束标志重置为 False
        self.done = False
        # 按照任务的到达时间对任务列表进行排序
        self.initial_tasks.sort(key=lambda t: t.arrive_time)
        # 获取当前环境的状态
        return self._get_state()

    def step(self):
        # 检查是否有新的任务到达
        new_tasks = []
        for task in self.initial_tasks:
            if task.arrive_time == self.current_time:
                self.tasks.append(task)
                new_tasks.append(task)
                print(f"Time step {self.current_time}: Task#{task.task_id} has arrived.")
            if task.deadline == self.current_time:
                for rescuers in self.initial_rescuers:
                    if rescuers.task == task:
                        rescuers.busy = False

        # 根据最近任务优先策略生成救援人员的行动
        actions = self._nearest_task_strategy()
        # 根据行动分配救援人员到任务
        self._assign_actions(actions)

        # 更新每个救援人员的状态
        for rescuer in self.rescuers:
            rescuer.update(self.current_time)

        # 更新每个任务的状态
        for task in self.tasks:
            task.update(self.current_time)

        # 时间步数加 1
        self.current_time += 1

        # 判断模拟是否结束
        if self.current_time >= self.max_time:
            self.done = True
        else:
            self.done = False

        # 获取下一步的环境状态
        next_state = self._get_state()
        return next_state, self.done

    def _assign_actions(self, actions):
        # 遍历每个行动：格式为 {救援人员ID: 任务ID}
        for rescuer_id, task_id in actions.items():
            # 根据救援人员的标识符获取救援人员对象
            rescuer = self.rescuers[rescuer_id]
            # 如果该救援人员已有任务且任务已被锁定，则跳过重新分配（解决频繁切换任务的问题）
            if rescuer.task is not None and rescuer.task_locked:
                continue
            # 根据任务的标识符获取任务对象
            target = self._find_task_by_id(task_id)
            # 如果任务对象不存在，则跳过该行动
            if not target:
                continue
            # 如果当前时间超过了任务的截止时间，则跳过该行动
            if self.current_time > target.deadline:
                continue
            # 如果救援人员不忙碌，则将其分配到目标任务
            if not rescuer.busy:
                rescuer.task = target
                rescuer.move(target.x, target.y)

    def _find_task_by_id(self, task_id):
        # 遍历所有任务，查找指定标识符的任务
        for t in self.tasks:
            if t.task_id == task_id:
                return t
        return None

    def _all_tasks_finished(self):
        # 如果所有的任务全部完成，模拟结束
        for task in self.tasks:
            if task.rescued_victim < task.initial_victim and self.current_time <= task.deadline:
                return False
        return True

    def _get_state(self):
        state = []
        # 遍历所有救援人员，将其状态信息添加到状态列表中
        for r in self.rescuers:
            current_task_id = r.task.task_id if r.task else -1
            state.extend([r.x, r.y, int(r.busy), current_task_id])
        # 遍历所有任务，将其状态信息添加到状态列表中
        for t in self.tasks:
            time_left = t.deadline - self.current_time
            state.extend([t.x, t.y, t.rescued_victim, time_left])
        return state

    def _nearest_task_strategy(self):
        actions = {}
        available_tasks = [task for task in self.tasks if
                           task.rescued_victim < task.initial_victim and
                           self.current_time <= task.deadline]

        # 步骤1：计算任务所需救援能力（考虑已分配救援人员的能力）
        task_needs = {}
        for task in available_tasks:
            remaining_time = task.deadline - self.current_time
            if remaining_time <= 0:
                continue

            # 计算已分配救援人员的有效救援量（考虑到达时间）
            assigned_effective = 0
            for rescuer in task.assigned_rescuers:
                arrival_time = abs(rescuer.x - task.x) + abs(rescuer.y - task.y)
                effective_time = remaining_time - arrival_time
                if effective_time > 0:
                    assigned_effective += rescuer.efficiency * effective_time

            remaining_needed = max(task.survivors_not_rescued - assigned_effective, 0)

            # 收集可用救援人员并计算他们的有效救援量
            valid_rescuers = []
            for rescuer in self.rescuers:
                if rescuer.busy or rescuer in task.assigned_rescuers:
                    continue  # 排除已忙碌或已分配的人员
                arrival_time = abs(rescuer.x - task.x) + abs(rescuer.y - task.y)
                effective_time = remaining_time - arrival_time
                if effective_time <= 0:
                    continue
                effective_rescue = rescuer.efficiency * effective_time
                valid_rescuers.append((effective_rescue, rescuer))

            # 按救援量降序排序
            valid_rescuers.sort(reverse=True, key=lambda x: x[0])

            # 计算需要多少救援人员来满足剩余需求
            required = 0
            current_rescue = 0
            for eff, r in valid_rescuers:
                current_rescue += eff
                required += 1
                if current_rescue >= remaining_needed:
                    break

            task_needs[task] = required

        # 步骤2：任务排序（按紧急程度：剩余时间少、需求高优先）
        sorted_tasks = sorted(
            [t for t in available_tasks if task_needs.get(t, 0) > 0],
            key=lambda t: (
                t.deadline - self.current_time,  # 剩余时间越少越紧急
                -task_needs[t]  # 需要救援人员越多越优先
            )
        )

        # 步骤3：收集空闲救援人员（按能力降序）
        free_rescuers = sorted(
            [r for r in self.rescuers if not r.busy],
            key=lambda r: -r.efficiency
        )

        # 步骤4：分配救援人员到任务
        for task in sorted_tasks:
            needed = task_needs.get(task, 0)
            if needed <= 0 or not free_rescuers:
                continue

            # 选择距离最近且能力高的救援人员
            candidates = []
            for rescuer in free_rescuers:
                distance = abs(rescuer.x - task.x) + abs(rescuer.y - task.y)
                candidates.append((distance, -rescuer.efficiency, rescuer))

            # 按距离升序、能力降序排序
            candidates.sort(key=lambda x: (x[0], x[1]))

            # 分配前needed个救援人员
            for _, _, rescuer in candidates[:needed]:
                actions[rescuer.rescuer_id] = task.task_id
                free_rescuers.remove(rescuer)

            if not free_rescuers:
                break

        return actions

    def _get_urgency(self, task):
        if not task:
            return -1
        distance = abs(task.x - self.rescuers[0].x) + abs(task.y - self.rescuers[0].y)
        arrival_time = self.current_time + distance
        return (task.deadline - arrival_time) / (distance + 1)

    
    """
    def print_status(self):
        print(f"Time step: {self.current_time}")
        print("Rescuers:")
        for rescuer in self.rescuers:
            if rescuer.busy:
                task = rescuer.task
                if task.victim == 0:
                    progress = "0.00%"
                else:
                    progress = f"{task.rescued_victim / task.victim * 100:.2f}%"
                print(f"  Rescuer#{rescuer.rescuer_id} is working on Task#{task.task_id} ({progress} completed)")
            else:
                print(f"  Rescuer#{rescuer.rescuer_id} is free.")

        completed_tasks = [task for task in self.tasks if task.rescued_victim >= task.victim]
        incomplete_tasks = [task for task in self.tasks if task.rescued_victim < task.victim]

        print("\nCompleted Tasks:")
        if completed_tasks:
            for task in completed_tasks:
                print(f"  Task#{task.task_id} (arrived at {task.arrive_time}): Rescued {task.victim} victims.")
        else:
            print("  None")

        print("\nIncomplete Tasks:")
        if incomplete_tasks:
            for task in incomplete_tasks:
                if task.victim == 0:
                    progress = "0.00%"
                else:
                    progress = f"{task.rescued_victim / task.victim * 100:.2f}%"
                print(
                    f"  Task#{task.task_id} (arrived at {task.arrive_time}): {task.rescued_victim}/{task.victim} victims rescued ({progress}).")
        else:
            print("  None")

        total_victims = sum(task.victim for task in self.tasks)
        total_rescued = sum(task.rescued_victim for task in self.tasks)
        if total_victims == 0:
            overall_progress = "0.00%"
        else:
            overall_progress = f"{total_rescued / total_victims * 100:.2f}%"
        print(f"\nOverall Rescue Progress: {overall_progress}")

        if self.done:
            total_rescued = sum(task.rescued_victim for task in self.tasks)
            if self.total_victims == 0:
                rescue_ratio = "0.00%"
            else:
                rescue_ratio = f"{total_rescued / self.total_victims * 100:.2f}%"
            print(f"\nTotal rescued / Total victims: {total_rescued}/{self.total_victims} ({rescue_ratio})")
            print("\nRescuers' Participated Tasks:")
            for rescuer in self.rescuers:
                task_ids = [task.task_id for task in rescuer.participated_tasks]
                print(f"  Rescuer#{rescuer.rescuer_id} participated in tasks: {task_ids}")
            print("\nTasks' Final Results:")
            for task in self.tasks:
                print(
                    f"  Task#{task.task_id} (arrived at {task.arrive_time}): Rescued {task.rescued_victim}/{task.victim} victims")
            print("\nTasks' Arrival Times:")
            for task in self.initial_tasks:
                print(f"  Task#{task.task_id} arrived at {task.arrive_time}")
                print(f"  Task#{task.task_id} happens at ({task.x} , {task.y})")
                print(f"  Task#{task.task_id} ends at {task.deadline}")
    """
    def print_status(self):
        if self.done:
            total_rescued = sum(t.rescued_victim for t in self.tasks)
            ratio = f"{total_rescued/self.total_victims*100:.2f}%" if self.total_victims>0 else "0.00%"
            print(f"Total rescued / Total victims: {total_rescued}/{self.total_victims} ({ratio})")

    def _get_plot_data(self):
        """辅助可视化的数据获取"""
        rescuers_x = [r.x for r in self.rescuers]
        rescuers_y = [r.y for r in self.rescuers]
        rescuer_colors = ['red' if r.busy else 'green' for r in self.rescuers]

        tasks_x = []
        tasks_y = []
        task_colors = []
        task_sizes = []
        for task in self.tasks:
            if self.current_time > task.deadline:
                task_colors.append('gray')  # Missed tasks
            elif self.survivors_not_rescued == 0:
                task_colors.append('blue')  # Completed tasks
            else:
                task_colors.append('orange')  # Active tasks
            tasks_x.append(task.x)
            tasks_y.append(task.y)
            task_sizes.append(50 + task.survivors_not_rescued * 2)

        return (
            np.array(rescuers_x),
            np.array(rescuers_y),
            np.array(rescuer_colors),
            np.array(tasks_x),
            np.array(tasks_y),
            np.array(task_colors),
            np.array(task_sizes)
        )


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
    ax2.set_ylim(0, 10000)
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
    # ================== 数据追踪器 ==================
    rescue_data = {
        'time': [],
        'rescued': [],
        'active_tasks': [],
        'idle_rescuers': []
    }
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
        # === 任务状态更新 ===
        active_tasks = [t for t in env.tasks 
                       if t.rescued_victim < t.initial_victim and env.current_time <= t.deadline]


        # === 数据记录 ===
        current_rescued = sum(t.rescued_victim for t in env.tasks)
        rescue_data['time'].append(env.current_time)
        rescue_data['rescued'].append(current_rescued)
        rescue_data['active_tasks'].append(len(active_tasks))
        rescue_data['idle_rescuers'].append(sum(1 for r in env.rescuers if not r.busy))



        rescued = sum(t.rescued_victim for t in env.tasks)
        status_text.set_text(f'Rescued: {rescued}/{env.total_victims}')
        time_text.set_text(f'Time: {env.current_time}/{env.max_time}')

        progress_line.set_data(range(env.current_time + 1),
                               [sum(t.rescued_victim for t in env.tasks[:i + 1])
                                for i in range(env.current_time + 1)])

        total_victims = env.total_victims
        total_target_line.set_data([0, env.max_time], [total_victims, total_victims])

        progress_line.set_data(rescue_data['time'], rescue_data['rescued'])

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

    ani.save('rescue_simulation3.gif', writer='pillow', fps=5, dpi=100)

# 保存随机生成的数据到 JSON 文件
def save_random_data(tasks, rescuers):
    task_data = [{"task_id": t.task_id, "arrive_time": t.arrive_time, "x": t.x, "y": t.y,
                  "victim": t.initial_victim, "deadline": t.deadline} for t in tasks]

    # 新增rescue_efficiency字段
    rescuer_data = [{"rescuer_id": r.rescuer_id, "x": r.x, "y": r.y,
                     "efficiency": r.efficiency} for r in rescuers]

    with open('data2.json', 'w') as f:
        json.dump({"tasks": task_data, "rescuers": rescuer_data}, f, indent=4)


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


# 示例使用
if __name__ == "__main__":
    
    num_tasks = 150
    num_rescuers = 150
    # tasks = generate_random_tasks(num_tasks)
    # rescuers = init_rescuers(num_rescuers)
    # save_random_data(tasks, rescuers)

    tasks, rescuers =load_simulation_data("data2.json")
    env = RescueEnvCore(tasks, rescuers)
    state = env.reset()
    # Visualize(env=env)
    done = False
    while not done:
        state, done = env.step()
        env.print_status()
    
    