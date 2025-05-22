import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import sys
class TaskAttention(nn.Module):
    def __init__(self, agent_dim, task_dim):
        super().__init__()
        self.query = nn.Linear(agent_dim, task_dim)
        self.key = nn.Linear(task_dim, task_dim)
        
    def forward(self, agent_feat, tasks_feat):
        Q = self.query(agent_feat).unsqueeze(1)  # [B, 1, task_dim]
        K = self.key(tasks_feat)                 # [B, max_tasks, task_dim]

        # 检查数值稳定性
        score = Q @ K.transpose(1, 2)  # [B, 1, max_tasks]
        if torch.isnan(score).any():
            print("❗Attention score contains NaN")
            print("Q:", Q)
            print("K:", K)
            #sys.exit()
        scale = torch.sqrt(torch.tensor(K.size(-1), dtype=torch.float32, device=K.device))
        attn = torch.softmax(score / (scale + 1e-6), dim=-1)

        # 防止 NaN
        attn = torch.nan_to_num(attn, nan=1e-6)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        return attn.squeeze(1)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, num_rescuers, max_tasks):
        super().__init__()
        self.num_rescuers = num_rescuers
        self.max_tasks = max_tasks  # 使用课程学习配置的最大任务数
        self.task_feat_dim = 4  # 每个任务每个救援者的特征数
        self.agent_feat_dim = 4  # 每个救援者的特征维度
        
        # 自动计算输入维度
        self.total_agent_dim = num_rescuers * self.agent_feat_dim
        self.total_task_dim = max_tasks * num_rescuers * self.task_feat_dim
        
        self.agent_encoder = nn.Sequential(
            nn.Linear(self.agent_feat_dim, 64),
            nn.ReLU()
        )
        
        self.task_attention = TaskAttention(
            agent_dim=64,
            task_dim=self.task_feat_dim
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(64 + max_tasks, 128),  # 使用max_tasks保持维度稳定
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        # 添加层归一化
        self.layer_norm = nn.LayerNorm(64 + max_tasks)

    def forward(self, x):
        # 确保输入没有NaN
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        if torch.isnan(x).any():
            print("❗Input x contains NaN")
            print(x)
            sys.exit()
        batch_size = x.shape[0]
        
        # 分割特征
        agent_feat = x[:, :4*self.num_rescuers]  # [batch, 4*N_rescuer]
        task_feat = x[:, 4*self.num_rescuers:]   # [batch, max_tasks*N_rescuer*4]

        # 处理智能体特征
        agent_feat = agent_feat.view(batch_size, self.num_rescuers, self.agent_feat_dim)
        agent_feat = self.agent_encoder(agent_feat)
        
        # 处理任务特征（考虑最大任务数）
        # 重塑任务特征张量
        task_feat = task_feat.view(
            batch_size, 
            self.max_tasks, 
            self.num_rescuers, 
            self.task_feat_dim
        )  # [batch, max_tasks, N_rescuer, 4]
        task_feat = task_feat.mean(dim=2)  # [batch, max_tasks, 4]
        
        # 注意力机制
        attn_weights = self.task_attention(agent_feat.mean(dim=1), task_feat)
        
        # 合并特征
        combined = torch.cat([agent_feat.mean(dim=1), attn_weights], dim=1)
        logits = self.decoder(combined)

         # 在合并特征后添加层归一化
        combined = torch.cat([agent_feat.mean(dim=1), attn_weights], dim=1)
        combined = self.layer_norm(combined)
        
        # 添加梯度裁剪
        logits = self.decoder(combined)
        logits = torch.clamp(logits, min=-10, max=10)  # 限制logits范围
        
        # 更稳定的softmax计算
        max_logits = logits.max(dim=-1, keepdim=True).values
        stable_logits = logits - max_logits
        probs = F.softmax(stable_logits, dim=-1)
        
        # 确保概率有效
        probs = torch.nan_to_num(probs, nan=1.0/self.max_tasks)
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        return probs


    
class Critic(nn.Module):
    def __init__(self, global_state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)

class MAPPO:
    def __init__(self, num_agents, state_dim, action_dim, global_state_dim,
                 max_tasks,  # 新增参数：最大任务数
                 lr_actor=3e-4, lr_critic=1e-3, gamma=0.99, clip_epsilon=0.2):
        
        # 添加类属性存储参数
        self.num_agents = num_agents
        self.max_tasks = max_tasks
        
        # 修正后的Actor初始化
        self.actors = [
            Actor(
                state_dim=state_dim,
                action_dim=action_dim,  # 使用传入的action_dim参数
                num_rescuers=num_agents,  # 使用类参数num_agents
                max_tasks=self.max_tasks  # 使用类属性
            ) for _ in range(num_agents)  # 使用类参数num_agents
        ]
        
        self.critic = Critic(global_state_dim)
        self.optimizers = [optim.Adam(actor.parameters(), lr=lr_actor) for actor in self.actors]
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self._init_weights()  # 新增此行

    def _init_weights(self):
        for actor in self.actors:
            for layer in actor.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('relu'))  # 调整增益
                    nn.init.constant_(layer.bias, 0)
        # Critic使用更稳定的初始化
        nn.init.orthogonal_(self.critic.net[0].weight, gain=1.0)
        nn.init.constant_(self.critic.net[0].bias, 0)

    def update(self, states, global_states, actions, advantages, next_global_states=None, rewards=None, dones=None):
        # 转换为张量并确保维度正确
        states = torch.stack(states)  # [num_agents, batch, state_dim]
        global_states = torch.stack(global_states)  # [batch, global_state_dim]
        actions = torch.stack(actions)  # [num_agents, batch]

        # Critic更新（使用更稳定的TD目标计算）
        with torch.no_grad():
            next_v = self.critic(next_global_states).squeeze() if next_global_states is not None else 0
            td_target = rewards + self.gamma * next_v * (1 - dones)
        
        current_v = self.critic(global_states).squeeze()
        critic_loss = F.mse_loss(current_v, td_target)
        
        # 反向传播Critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optim.step()

        # 更新每个Actor
        for agent_idx in range(self.num_agents):
            agent_states = states[agent_idx]  # [batch, state_dim]
            agent_actions = actions[agent_idx]  # [batch]

            # 计算新旧策略概率
            probs = self.actors[agent_idx](agent_states)
            dist = Categorical(probs)
            old_probs = dist.log_prob(agent_actions).exp().detach()
            
            # PPO损失计算
            ratio = (dist.log_prob(agent_actions).exp() / old_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 熵正则化
            entropy = dist.entropy().mean()
            actor_loss -= 0.01 * entropy

            # 反向传播并应用梯度裁剪
            self.optimizers[agent_idx].zero_grad()
            actor_loss.backward(retain_graph=(agent_idx < self.num_agents-1))
            torch.nn.utils.clip_grad_norm_(self.actors[agent_idx].parameters(), 1.0)
            self.optimizers[agent_idx].step()