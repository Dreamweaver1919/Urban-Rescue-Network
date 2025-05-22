import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
from Environment import RescueEnvCore
from init import generate_random_tasks, init_rescuers
from MAPPOagent import MAPPO
from Task import Task
from Rescuer import Rescuer
import os

# 训练参数解析
def parse_args():
    """修改后的参数解析函数，适配Jupyter环境"""
    parser = argparse.ArgumentParser(
        description='MAPPO Rescue Training',
        allow_abbrev=False  # 禁止缩写参数
    )
    
    # 添加必需参数
    parser.add_argument('--num_tasks', type=int, default=5, 
                       help='Initial number of tasks')
    parser.add_argument('--num_rescuers', type=int, default=3,
                       help='Number of rescue agents')
    parser.add_argument('--grid_size', type=int, default=100, help='Simulation grid size')
    parser.add_argument('--max_time', type=int, default=300, help='Maximum timesteps per episode')
    parser.add_argument('--episodes', type=int, default=100, help='Total training episodes')
    parser.add_argument('--lr_actor', type=float, default=1e-4, help='Actor learning rate')
    parser.add_argument('--lr_critic', type=float, default=3e-4, help='Critic learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--clip_epsilon', type=float, default=0.1, help='PPO clip parameter')
    parser.add_argument('--save_interval', type=int, default=500, help='Model save interval')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval')
    try:
        from IPython import get_ipython
        if 'IPKernelApp' in get_ipython().config:  # 检测Jupyter环境
            return parser.parse_args(args=[])     # 返回默认参数
    except:
        pass
    
    # 命令行环境正常解析
    args, _ = parser.parse_known_args()
    return args

curriculum_config = {
    'stage': 0,
    'max_tasks': 10,  # 定义最大任务数
    'min_deadline': 50,
    'max_initial_deadline': 200  # 初始最大deadline
}

 

# 训练主函数
def train(args):
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 环境初始化（使用最大可能维度）
    initial_tasks = generate_random_tasks(
        num_tasks=curriculum_config['max_tasks'],
        grid_size=args.grid_size,
        max_deadline=curriculum_config['max_initial_deadline']  # 关键修改点1
    )
    
    env = RescueEnvCore(
        tasks=initial_tasks[:args.num_tasks],  # 初始使用部分任务
        rescuers=init_rescuers(args.num_rescuers, args.grid_size),
        grid_size=args.grid_size,
        max_time=args.max_time,
        max_tasks=curriculum_config['max_tasks']
    )
    
    # 计算状态维度时考虑最大任务数
    state_dim = 4 * args.num_rescuers + 4 * args.num_rescuers * curriculum_config['max_tasks']
   
    agent = MAPPO(
        num_agents=args.num_rescuers,  # 从args获取救援者数量
        state_dim=state_dim,
        action_dim=curriculum_config['max_tasks'],  # 动作空间=最大任务数
        global_state_dim=state_dim,
        max_tasks=curriculum_config['max_tasks'],  # 传递最大任务数
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        clip_epsilon=args.clip_epsilon
    )
    
    # Add these lines to move networks to the device
    for actor in agent.actors:
        actor.to(device)
    agent.critic.to(device)
    
    # 训练指标跟踪
    episode_rewards = []
    episode_rescued = []
    efficiency_metrics = []
    recent_success = deque(maxlen=100)
    
    # 创建模型保存目录
    os.makedirs("models", exist_ok=True)

 # 添加初始状态检查
    initial_state = env._get_state()
    print("Initial state stats:")
    print(f"Min: {initial_state.min()}, Max: {initial_state.max()}")
    print(f"NaN count: {np.isnan(initial_state).sum()}")
    print(f"Inf count: {np.isinf(initial_state).sum()}")
    
    # 如果发现NaN/Inf，立即修复
    if np.isnan(initial_state).any() or np.isinf(initial_state).any():
        print("⚠️ Found invalid values in initial state, applying fixes...")
        initial_state = np.nan_to_num(initial_state, nan=0.0, posinf=1.0, neginf=-1.0)

        
     # ================= 训练主循环 =================
    for episode in range(args.episodes):
        # ===== 课程学习更新（每500个episode调整难度） =====
        if episode % 500 == 0 and episode > 0:
            # 更新课程阶段
            curriculum_config['stage'] += 1  # 关键修改点2
            
            # 动态计算新参数
            new_num_tasks = min(
                args.num_tasks + curriculum_config['stage'],
                curriculum_config['max_tasks']
            )
            new_deadline = max(
                curriculum_config['max_initial_deadline'] - curriculum_config['stage'] * 20,
                curriculum_config['min_deadline']
            )
            
            # 使用动态deadline生成新任务
            updated_tasks = generate_random_tasks(
                num_tasks=curriculum_config['max_tasks'],
                grid_size=args.grid_size,
                max_deadline=new_deadline  # 关键修改点3
            )
            
            # 重置环境
            env = RescueEnvCore(
                tasks=updated_tasks[:new_num_tasks],
                rescuers=init_rescuers(args.num_rescuers, args.grid_size),
                grid_size=args.grid_size,
                max_time=args.max_time,
                max_tasks=curriculum_config['max_tasks']
            )

        # 环境重置
        state = env.reset()
        episode_reward = 0
        total_rescued = 0
        done = False
        
        # 探索率衰减（确保不低于5%）
        epsilon = max(0.3, 0.8 - episode/(args.episodes//3))  # 更激进的探索

        # 单回合训练
        while not done:
            # 将当前状态转换为全局状态张量
            global_state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # 添加batch维度
            # 动作选择
            actions = []
            # 在生成动作的部分应用mask
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                for agent_id in range(args.num_rescuers):
                    if np.random.rand() < epsilon:
                        # 随机选择有效任务
                        valid_tasks = [t.task_id for t in env.active_tasks]
                        if valid_tasks:
                            actions.append(np.random.choice(valid_tasks))
                        else:
                            actions.append(0)
                    else:
                        '''# 在 train 循环里，临时把 else 分支全部注释掉，改成：
                        if np.random.rand() < 1.0:  # 100% 随机
                            valid_tasks = [t.task_id for t in env.active_tasks]
                            action = np.random.choice(valid_tasks) if valid_tasks else 0
                            actions.append(action)'''

                        # else 分支里选动作的部分
                        action_probs = agent.actors[agent_id](state_tensor)            # [1, max_tasks]
                        valid_tasks = [t.task_id for t in env.active_tasks]

                        # 1) 把 max_tasks 缓存成局部变量
                        num_tasks_dim = action_probs.size(-1)

                        # 2) 创建 mask
                        mask = torch.zeros_like(action_probs)
                        if valid_tasks:
                            mask[:, valid_tasks] = 1.0
                        else:
                            mask[:, 0] = 1.0

                        # 3) 计算 raw_masked 和 sum
                        raw_masked = action_probs * mask                              # [1, num_tasks_dim]
                        sum_probs = raw_masked.sum(dim=1, keepdim=True)               # [1,1]

                        # 4) 准备 fallback 均匀分布
                        uniform_fallback = mask / (mask.sum(dim=1, keepdim=True) + 1e-8)

                        # 5) 判断并归一化
                        is_good = (sum_probs > 1e-6).float()                          # [1,1]
                        normalized = raw_masked / (sum_probs + 1e-8)                 # [1, num_tasks_dim]
                        masked_probs = is_good * normalized + (1 - is_good) * uniform_fallback

                        # 6) 把任何 NaN/∞ 都替换成 1/num_tasks_dim
                        masked_probs = torch.nan_to_num(
                            masked_probs,
                            nan=1.0/num_tasks_dim,
                            posinf=1.0,
                            neginf=0.0
                        )
                       

                        #归一化
                        # Step 1: 计算每一行的总和
                        row_sums = masked_probs.sum(dim=-1, keepdim=True)
                        if torch.any(row_sums == 0):
                           print("⚠️ 有全为 0 的 masked_probs 行，自动回退为 uniform.")


                        # Step 2: 检查是否有全为0的行（即被mask后无可选动作）
                        is_zero_row = (row_sums == 0)

                        # Step 3: fallback到均匀概率（只在全0行使用）
                        # 这个大小必须与 masked_probs 的 shape 一致
                        fallback_probs = torch.full_like(masked_probs, 1e-8)

                        # Step 4: 替换全为0的行
                        safe_probs = torch.where(is_zero_row, fallback_probs, masked_probs)

                        # Step 5: 再归一化（确保和为1）
                        safe_probs = safe_probs / safe_probs.sum(dim=-1, keepdim=True)


                        #  替换对 safe_probs 归一化的 assert（之前是 masked_probs）
                        assert torch.allclose(
                            safe_probs.sum(dim=-1), 
                            torch.ones_like(safe_probs.sum(dim=-1)), 
                            atol=1e-4
                        ), f"Safe probs not normalized: {safe_probs}"

                        # 7) 使用categorical
                        m = torch.distributions.Categorical(probs=safe_probs)
                        actions.append(m.sample().item())

                    

            # 在训练循环中生成动作并处理
            # 只保留那些合法的 task_id
            valid = [t.task_id for t in env.active_tasks]
            action_dict = {}
            for i, a in enumerate(actions):
                if a in valid:
                    action_dict[i] = a
                else:
                    action_dict[i] = np.random.choice(valid) if valid else 0

            next_state, reward, _, done, info = env.step(action_dict)

            '''# 打印每个 agent 的动作和移动
            for i, r in enumerate(env.rescuers):
                print(f"Agent {i}: moved to ({r.x:.1f},{r.y:.1f}), last action target {action_dict[i]}")

            # 打印当前所有 active_tasks 的位置和剩余未救援人数
            for t in env.active_tasks:
                print(f" Task {t.task_id}: pos=({t.x},{t.y}), remaining={t.initial_victim - t.rescued_victim}")'''


            next_global_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            rewards_tensor = torch.FloatTensor([reward]).to(device)
            dones_tensor = torch.FloatTensor([done]).to(device)
            
            # 转换数据为张量
            state_tensor = torch.FloatTensor(state).to(device)
            next_state_tensor = torch.FloatTensor(next_state).to(device)
            actions_per_agent = [torch.tensor([action], dtype=torch.long).to(device) for action in action_dict.values()]
            
            # 优势计算
            with torch.no_grad():
                current_v = agent.critic(global_state_tensor)
                next_v = agent.critic(next_global_state_tensor) if not done else 0
                td_target = reward + agent.gamma * next_v * (1 - done)
                advantages = (td_target - current_v).detach()

            # 参数更新
            agent.update(
                states=[global_state_tensor] * args.num_rescuers,
                global_states=[global_state_tensor],
                actions=actions_per_agent,  # 直接传递张量列表
                advantages=advantages,
                next_global_states=next_global_state_tensor,
                rewards=torch.FloatTensor([reward]).to(device),
                dones=torch.FloatTensor([done]).to(device)
            )
            
            # 状态转移
            state = next_state
            episode_reward += reward
            total_rescued += info.get('rescued_this_step', 0)
            global_state_tensor = next_global_state_tensor  # 重要！保持状态连续

        # 记录指标
        episode_rewards.append(episode_reward)
        episode_rescued.append(total_rescued)
        recent_success.append(1 if total_rescued > 0 else 0)
        
        # 效率计算（处理除零错误）
        active_tasks = [t for t in env.active_tasks if t.initial_victim > 0]
        if active_tasks:
            efficiency = sum(t.rescued_victim / t.initial_victim for t in active_tasks) / len(active_tasks)
        else:
            efficiency = 0.0
        efficiency_metrics.append(efficiency)


        
        # 定期保存和日志
        if episode % args.log_interval == 0:
            avg_reward = np.mean(episode_rewards[-args.log_interval:] or [0])
            avg_rescued = np.mean(episode_rescued[-args.log_interval:] or [0])
            success_rate = np.mean(recent_success)*100 if recent_success else 0
            
            print(f"\nEpisode {episode+1}/{args.episodes}")
            print(f"[Curr] Reward: {episode_reward:.1f} | Rescued: {total_rescued}")
            print(f"[Avg]  Reward: {avg_reward:.1f} | Rescued: {avg_rescued:.1f}")
            print(f"Success Rate: {success_rate:.1f}% | Efficiency: {np.mean(efficiency_metrics[-10:]):.2f}")
            print(f"Exploration: {epsilon:.2f} | Curriculum Stage: {curriculum_config['stage']}")
        
        if episode % args.save_interval == 0 and episode > 0:
            save_path = f"models/mappo_ep{episode}.pt"
            torch.save({
                'actors': [a.state_dict() for a in agent.actors],
                'critic': agent.critic.state_dict(),
                'config': vars(args)
            }, save_path)
            print(f"Saved model to {save_path}")

        # After training loop
    final_save_path = "models/mappo_final.pt"
    torch.save({
        'actors': [a.state_dict() for a in agent.actors],
        'critic': agent.critic.state_dict(),
        'config': vars(args)
    }, final_save_path)
    print(f"\nFinal model saved to {final_save_path}")

    # 训练结果可视化
    plt.figure(figsize=(15,6))
    
    # 奖励曲线
    plt.subplot(1,2,1)
    plt.plot(episode_rewards, alpha=0.3, label='Instant')
    plt.plot(np.convolve(episode_rewards, np.ones(100)/100, mode='same'), 
            'r-', label='Moving Avg')
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Training Rewards")
    plt.legend()
    
    # 救援人数曲线
    plt.subplot(1,2,2)
    plt.plot(episode_rescued, alpha=0.3, label='Instant')
    plt.plot(np.convolve(episode_rescued, np.ones(100)/100, mode='same'),
            'g-', label='Moving Avg')
    plt.xlabel("Episodes")
    plt.ylabel("Rescued Victims")
    plt.title("Rescue Performance")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("training_progress.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    train(args)