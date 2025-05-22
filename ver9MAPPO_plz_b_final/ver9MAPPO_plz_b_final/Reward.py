import numpy as np
from scipy.special import expit

def calculate_reward(env, rescued_this_step):
    """适配MAPPO的奖励函数，包含多智能体协作要素"""
    # 超参数配置（经过MAPPO调优）
    params = {
        'alpha': 4.0,    # 协作效率奖励
        'beta': 2.0,      # 时间紧迫性奖励
        'gamma': 0.0005,    # 协同移动惩罚
        'delta': 15,     # 即时救援奖励
        'eta': 30.0,       # 任务完成奖励
        'zeta': 0.3,      # 协同探索奖励
        'lambda': 0.02     # 任务切换惩罚
    }
    
    # 全局状态参数
    total_initial = sum(t.initial_victim for t in env.initial_tasks)  # 使用initial_tasks保持维度稳定
    time_elapsed_ratio = env.current_time / env.max_time
    
    # 1. 多智能体协作效率奖励（考虑任务分配均衡性）
    rescue_counts = [len(t.assigned_rescuers) for t in env.active_tasks]
    allocation_score = 1 - np.std(rescue_counts)/(len(env.rescuers)+1e-6)
    R_coop = allocation_score * np.log1p(sum(t.rescued_victim for t in env.active_tasks))
    
    # 2. 动态时间紧迫性奖励（全任务视角）
    urgency_scores = []
    for t in env.active_tasks:
        time_left = max(t.deadline - env.current_time, 0)
        progress = t.rescued_victim / (t.initial_victim + 1e-6)
        urgency = (1 - progress) * expit(2 - 0.05*time_left)
        urgency_scores.append(urgency)
    R_urgency = np.mean(urgency_scores) if urgency_scores else 0
    
    # 3. 协同移动惩罚（考虑路径效率）
    avg_distance = np.mean([r.moved_distance for r in env.rescuers])
    R_move = -params['gamma'] * (avg_distance / 100) * (1 + time_elapsed_ratio)
    
    # 4. 即时救援奖励（标准化处理）
    #R_step = params['delta'] * (rescued_this_step)  # 而非除以总人数
    rescued_ratio = rescued_this_step / (total_initial + 1e-6)
    R_step = params['delta'] * (rescued_this_step ** 1.5)  # 救援越多奖励指数增长

    # 5. 任务完成奖励（兼容课程学习）
    completed_tasks = sum(1 for t in env.initial_tasks if t.rescued_victim >= t.initial_victim)
    R_complete = params['eta'] * (completed_tasks / len(env.initial_tasks)) ** 2
    
    # 6. 区域覆盖奖励（协同探索）
    visited_grids = set((r.x, r.y) for r in env.rescuers)
    coverage_ratio = len(visited_grids) / (env.grid_size**2)
    R_explore = params['zeta'] * np.sqrt(coverage_ratio)
    
    # 7. 策略稳定性惩罚（减少无效切换）
    R_switch = -params['lambda'] * env.interruption_count / len(env.rescuers)
    
    # 综合奖励
    total_reward = (
        params['alpha'] * R_coop +
        params['beta'] * R_urgency +
        R_move +
        R_step +
        R_complete +
        R_explore +
        R_switch
    )
    
    # 诊断信息
    reward_details = {
        'components': {
            'cooperation': R_coop,
            'urgency': R_urgency,
            'movement': R_move,
            'immediate': R_step,
            'completion': R_complete,
            'exploration': R_explore,
            'switching': R_switch
        },
        'weights': params,
        'env_status': {
            'current_time': env.current_time,
            'active_tasks': len(env.active_tasks),
            'total_rescued': sum(t.rescued_victim for t in env.initial_tasks)
        }
    }
    # 在返回前添加保护
    total_reward = np.nan_to_num(total_reward, nan=0.0, posinf=10.0, neginf=-10.0)
    for k, v in reward_details['components'].items():
        reward_details['components'][k] = np.nan_to_num(v, nan=0.0)
    
        # Episode 结束奖励
    if env.current_time >= env.max_time:
        # 没完成所有任务
        total_reward -= 20
    elif all(t.rescued_victim >= t.initial_victim for t in env.initial_tasks):
        total_reward += 50

    return total_reward, reward_details
   