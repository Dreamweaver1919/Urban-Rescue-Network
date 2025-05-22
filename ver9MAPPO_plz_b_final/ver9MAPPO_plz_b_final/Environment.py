import numpy as np
import Task as TaskModule
import Rescuer as RescuerModule
from copy import deepcopy
from Reward import calculate_reward

class RescueEnvCore:
    def __init__(self, tasks, rescuers, grid_size, max_time=300, reward_params=None, max_tasks=10):
        self.max_tasks = max_tasks  # 新增参数
        self.initial_tasks = deepcopy(tasks)
        self.initial_rescuers = deepcopy(rescuers)
        self.max_time = max_time
        self.grid_size = grid_size

        default_params = {'alpha': 1.0, 'beta': 0.1, 'gamma': 2.0, 'sigma': 0.1, 'eta': 0.5}
        if reward_params:
            default_params.update(reward_params)
        self.reward_params = default_params

        self.active_tasks = []
        self.rescuers = []
        self.current_time = 0
        self.done = False
        self.prev_total_rescued = 0
        self.interruption_count = 0
        self.strategy_func = None
        self.reset()

    def set_strategy(self, strategy_func):
        self.strategy_func = strategy_func

    def reset(self):
        self.active_tasks = [deepcopy(t) for t in self.initial_tasks]
        self.rescuers = [deepcopy(r) for r in self.initial_rescuers]
        self.current_time = 0
        self.done = False
        self.prev_total_rescued = 0
        self.interruption_count = 0
        return self._get_state()

    def step(self, actions=None):
        # 1) 分配动作
        if actions is None and self.strategy_func is not None:
            actions = self.strategy_func(self)
        self._assign_actions(actions)

        # 2) 更新所有救援者位置
        for rescuer in self.rescuers:
            rescuer.update(self.current_time)

        # 3) 推进时间并统计本步救援增量
        self.current_time += 1
        rescued_this_step = 0

        # 注意：Env 中正确的任务列表属性可能是 self.active_tasks 或 self.initial_tasks
        task_list = getattr(self, 'tasks', None) or getattr(self, 'active_tasks', None) or getattr(self, 'initial_tasks')
        for task in task_list:
            delta = task.update(self.current_time)
            rescued_this_step += delta or 0

        # 4) 计算 reward
        reward, reward_details = calculate_reward(self, rescued_this_step)

        # 5) 更新结束条件
        self.done = self.current_time >= self.max_time or not self.active_tasks

        # 6) 返回 state, reward, done, info
        info = {'rescued_this_step': rescued_this_step}
        return self._get_state(), reward, reward_details, self.done, info
    

    def _assign_actions(self, actions):
        for rescuer_id, task_idx in actions.items():
            if task_idx >= len(self.initial_tasks):  # 考虑课程学习后的最大任务数
                continue
            target_task = next((t for t in self.active_tasks if t.task_id == task_idx), None)
            if not target_task:
                continue

            rescuer = self.rescuers[rescuer_id]
            if rescuer.busy:
                if rescuer.task and rescuer.task.task_id != target_task.task_id:
                    rescuer.task.remove_rescuer(rescuer)
                    self.interruption_count += 1
            if not rescuer.busy:
                # 传递当前时间作为参数
                target_task.add_rescuer(rescuer, self.current_time)  # 新增current_time参数
                rescuer.move(target_task.x, target_task.y)

    # 修改后的_get_state方法
    def _get_state(self):
    # 救援者状态
        agent_states = []
        for r in self.rescuers:
            agent_state = [
                r.x / (self.grid_size - 1),  # 确保在[0,1]范围内
                r.y / (self.grid_size - 1),
                float(r.busy),
                (r.task.task_id + 1)/self.max_tasks if r.task else 0.0
            ]
            agent_states.extend(agent_state)
        
        # 任务状态
        task_features = []
        for i in range(self.max_tasks):
            if i < len(self.initial_tasks):
                task = self.initial_tasks[i]
                for r in self.rescuers:
                    dx = (task.x - r.x)/self.grid_size
                    dy = (task.y - r.y)/self.grid_size
                    time_left = max((task.deadline - self.current_time)/task.deadline, 0.0)
                    remaining_victims = max((task.initial_victim - task.rescued_victim)/task.initial_victim, 0.0)
                    
                    features = [
                        dx, dy,
                        time_left,
                        remaining_victims
                    ]
                    task_features.extend(features)
            else:
                # 填充任务用0.0而不是0
                task_features.extend([0.0] * (len(self.rescuers) * 4))
        
        # 合并并确保没有NaN
        state = np.array(agent_states + task_features, dtype=np.float32)
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        return state