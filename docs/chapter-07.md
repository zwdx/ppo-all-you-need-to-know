[上一章](chapter-06.md) · [返回目录](../README.md) · [阅读路线与符号表](README.md)

---

<a name="chapter-7"></a>

# 第 7 章 工程实现与实践

**本章目录**

- [7.1 环境接口与 Gymnasium](#s7-1)
- [7.2 向量环境（Vectorized Environments）](#s7-2)
- [7.3 网络结构设计的数学考量](#s7-3)
- [7.4 PPO 完整参考实现](#s7-4)
- [7.5 数值稳定性与实现细节的数学解释](#s7-5)
- [7.6 训练诊断指标与调参方法论](#s7-6)
- [7.7 常见坑清单](#s7-7)
- [7.8 推荐学习资源与进阶路径](#s7-8)
- [7.9 本章小结与下一步](#s7-9)

前六章完成了 PPO 从数学基础到算法本体的全部理论推导。本章的目标是将这些公式逐行落地为可运行的 PyTorch 代码，并明确指出每一处实现与理论之间的差异、近似和数值稳定性处理。读者在阅读本章时，应随时对照前文公式，理解"代码为什么这样写"而非仅仅记住模板。

<a name="s7-1"></a>

## 7.1 环境接口与 Gymnasium

<a name="s7-1-1"></a>

### 7.1.1 Gymnasium API 概述

Gymnasium 是 OpenAI Gym 的官方继任者，提供了标准化的强化学习环境接口。核心 API 有两个：

- `env.reset(seed=None)` → `(obs, info)`：重置环境并返回初始观测与辅助信息字典。
- `env.step(action)` → `(obs, reward, terminated, truncated, info)`：执行动作后返回五元组。

其中 `terminated` 与 `truncated` 的语义区分是 PPO 正确实现的关键，也是旧版 Gym（仅返回单一 `done`）最常见的 bug 来源。

<a name="s7-1-2"></a>

### 7.1.2 terminated 与 truncated 的语义区别

**`terminated=True`** 表示环境因内在逻辑真正结束了一个 episode（例如 CartPole 中杆子倒下、LunarLander 中着陆器坠毁）。此时状态 $s_{t+1}$ 是吸收态（absorbing state），其价值 $V(s_{t+1}) = 0$，在 GAE 计算中**不应 bootstrap**。

**`truncated=True`** 表示环境仅因外部限制（通常是达到最大步数上限）而中断，episode 在 MDP 意义上并未结束。此时 $s_{t+1}$ 仍然是一个有效的非终止状态，**必须用 $V(s_{t+1})$ 进行 bootstrap**，否则会将"时间到"错误地当作"失败"，导致价值函数系统性低估。

对应第 4 章的 TD 误差 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$，正确的写法应为：

$$
\delta_t = r_t + \gamma \cdot (1 - \text{terminated}_t) \cdot V(s_{t+1}) - V(s_t)
$$

注意这里只屏蔽 `terminated`，不屏蔽 `truncated`。一个常见的错误写法是 `nextnonterminal = 1.0 - dones[t+1]`，其中 `dones = terminated | truncated`——这会在 truncated 时也把 bootstrap 项置零，等价于把截断误认为终止。**这是 PPO 实现中最隐蔽且影响最大的 bug 之一。**

下面给出正确处理两者的 GAE 计算片段（完整版本见 7.4.3 节）：

```python
# 对应第 4 章递推式 Â_t = δ_t + γλÂ_{t+1}
# b_obs, b_rewards, b_dones, b_values, b_nextvalues 均为 rollout buffer 中的数组
# b_dones 仅记录 terminated（不含 truncated）
# b_truncateds 单独记录 truncated 标志
advantages = np.zeros_like(b_rewards)
lastgaelam = 0.0
for t in reversed(range(num_steps)):
    if t == num_steps - 1:
        # 最后一步：若 terminated 则 next_value=0，否则用存储的 next_value
        nextnonterminal = 1.0 - b_dones[t]  # 仅看 terminated！
        nextvalues = b_nextvalues[t]
    else:
        # 中间步骤：下一步是否因 terminated 而结束
        nextnonterminal = 1.0 - b_dones[t + 1]
        nextvalues = b_values[t + 1]
    delta = b_rewards[t] + gamma * nextnonterminal * nextvalues - b_values[t]
    advantages[t] = lastgaelam = delta + gamma * lam * nextnonterminal * lastgaelam
```

关键区别：`nextnonterminal` 只依赖 `b_dones`（即 `terminated`），不包含 `truncated`。当 `truncated=True` 但 `terminated=False` 时，bootstrap 仍然生效。

<a name="s7-2"></a>

## 7.2 向量环境（Vectorized Environments）

<a name="s7-2-1"></a>

### 7.2.1 为什么 PPO 必须用向量环境

第 6 章指出 PPO 每次更新需要 $N \times T$ 个样本（$N$ 个并行环境各采集 $T$ 步）。单环境采集效率极低，且批量过小会导致：(1) 优势估计方差过大；(2) mini-batch 更新不稳定。向量环境通过并行化解决这两个问题。

<a name="s7-2-2"></a>

### 7.2.2 SyncVectorEnv 与 AsyncVectorEnv

Gymnasium 提供两种向量环境封装：

- **`SyncVectorEnv`**：在同一进程中顺序执行 $N$ 个环境。优点是共享内存、无序列化开销；缺点是总耗时为 $N$ 倍单环境耗时。适合轻量级环境（如 CartPole）。
- **`AsyncVectorEnv`**：每个环境运行在独立子进程中，通过管道通信。真正的并行加速，适合计算密集型环境。代价是进程间数据序列化开销和更复杂的调试体验。

<a name="s7-2-3"></a>

### 7.2.3 Rollout Buffer 的形状约定

推荐约定：buffer 形状为 $(T, N, \ldots)$，即第一维是时间步、第二维是环境索引。填充完毕后 flatten 前两维得到 $(T \times N, \ldots)$ 用于 mini-batch 采样：

```python
# 采集阶段：shape = (num_steps, num_envs, ...)
obs_buffer = np.zeros((num_steps, num_envs) + obs_shape, dtype=np.float32)
# ... 填充循环 ...

# 更新阶段：flatten 为 (num_steps * num_envs, ...)
obs_flat = obs_buffer.reshape(-1, *obs_shape)
actions_flat = actions_buffer.reshape(-1, *action_shape)
# 后续按 mini_batch_size 随机切分
```

<a name="s7-2-4"></a>

### 7.2.4 异步环境的自动 reset 语义

向量环境中，当某个子环境 terminated 或 truncated 时，会**自动调用 `reset()` 并将新 episode 的首个观测写入返回值**。这意味着 rollout buffer 中相邻时间步可能属于不同 episode。GAE 计算时必须依靠 `terminated` 标志来正确断开 episode 边界，而不能假设连续的时间步属于同一 episode。这也是为什么 7.1.2 节中 `nextnonterminal` 的处理如此重要。

<a name="s7-3"></a>

## 7.3 网络结构设计的数学考量

<a name="s7-3-1"></a>

### 7.3.1 MLP 主干配置

对于 CartPole 等低维任务，两层 $64 \times 64$ + `tanh` 激活即可；对于 LunarLander 及以上复杂度的任务，推荐 $256 \times 256$ + `ReLU`。激活函数的选择影响梯度传播：`tanh` 输出有界 $[-1,1]$，适合小网络；`ReLU` 避免饱和但可能产生死神经元，配合正交初始化可缓解。

<a name="s7-3-2"></a>

### 7.3.2 初始化对策略梯度的特殊影响

PPO 对网络初始化极其敏感。**最后一层输出必须用小权重初始化**，使初始策略接近均匀分布（离散）或窄高斯（连续）。数学解释如下：

- **离散动作空间**：初始 logits 接近 $0$ → softmax 输出接近均匀分布 → 熵 $H[\pi]$ 最大 → 探索最充分。若初始 logits 绝对值过大，策略过早偏向某些动作，熵崩塌，难以恢复。
- **连续动作空间**：初始均值接近 $0$、标准差适中（如 $\sigma \approx 1$）→ 避免一开始就过度自信地集中在某个动作附近。

PyTorch 中使用正交初始化（orthogonal initialization）：

```python
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """正交初始化，对应第 1 章提到的参数初始化方法。"""
    torch.nn.init.orthogonal_(layer.weight, gain=std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

# 隐层：gain=sqrt(2)，补偿 ReLU/tanh 的方差缩放
layer_init(nn.Linear(obs_dim, 64), std=np.sqrt(2))
layer_init(nn.Linear(64, 64), std=np.sqrt(2))
# Actor 输出层：gain=0.01，使初始 logits 接近 0
layer_init(nn.Linear(64, action_dim), std=0.01)
# Critic 输出层：gain=1.0，价值预测无需特殊缩小
layer_init(nn.Linear(64, 1), std=1.0)
```

<a name="s7-3-3"></a>

### 7.3.3 连续动作空间的对数标准差参数化

对于对角 Gaussian 策略（对应第 1 章和第 5 章），标准差 $\sigma$ 的参数化有两种选择：

1. **独立参数**：`log_std = nn.Parameter(torch.zeros(action_dim))`，与状态无关。简单、参数少，适合大多数连续控制任务。
2. **网络输出**：由主干网络的额外头输出，与状态相关。表达力更强但训练更不稳定。

无论哪种方式，**必须约束 `log_std` 的范围**，通常 clip 到 $[-20, 2]$（对应 $\sigma \in [2 \times 10^{-9}, 7.39]$）或更保守的 $[-5, 2]$。原因：

- `log_std` 过小 → $\sigma \to 0$ → 策略退化为确定性策略 → 熵崩塌 → 无法再探索，且 KL 散度计算可能出现数值异常。
- `log_std` 过大 → $\sigma$ 爆炸 → 动作采样范围不合理 → 奖励信号被噪声淹没。

```python
class ContinuousActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
        )
        self.mean_head = layer_init(nn.Linear(64, action_dim), std=0.01)
        # 独立 log_std 参数，对应第 1 章对角 Gaussian 的参数化
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def get_action_and_value(self, x, action=None):
        features = self.backbone(x)
        mean = self.mean_head(features)
        # 约束 log_std 范围，防止 std 塌缩或爆炸
        log_std = self.log_std.clamp(min=-20.0, max=2.0)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        # 对应第 1 章：对角 Gaussian 的 log_prob = Σ_i log N(a_i; μ_i, σ_i)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return action, log_prob, entropy, mean
```

<a name="s7-3-4"></a>

### 7.3.4 Actor 与 Critic 共享主干 vs 分离主干

**共享主干**节省计算量（特征提取只做一次前向），但存在损失干扰问题（对应第 6 章讨论）：critic 的价值回归梯度会通过共享层反向传播到 actor 的策略输出层，可能破坏策略更新。代码层面体现为：

```python
# 共享主干写法
class SharedActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.backbone = nn.Sequential(...)  # 共享
        self.actor_head = layer_init(nn.Linear(64, action_dim), std=0.01)
        self.critic_head = layer_init(nn.Linear(64, 1), std=1.0)

    def get_action_and_value(self, x, action=None):
        features = self.backbone(x)
        # actor 和 critic 共用 features，梯度会互相影响
        # 对应第 6 章：此时 c_1 权重的选择更加敏感
        ...
```

**分离主干**各自独立前向，互不干扰，但计算量翻倍。实践中两者效果差异不大，初学者建议先用分离主干以避免调参复杂度。若使用共享主干，建议将 $c_1$ 设为较小值（如 0.5），并在监控 `explained_variance` 确认 critic 收敛后再考虑增大。

<a name="s7-4"></a>

## 7.4 PPO 完整参考实现

以下是一份可直接运行于 CartPole-v1 的 PPO 实现，每段代码均标注对应的理论公式出处。

<a name="s7-4-1"></a>

### 7.4.1 策略网络与价值网络

```python
import torch
import torch.nn as nn
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """正交初始化。隐层 gain=√2 补偿激活函数的方差缩放；
    输出层 gain=0.01 使初始策略接近均匀/窄高斯。"""
    torch.nn.init.orthogonal_(layer.weight, gain=std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class DiscreteActorCritic(nn.Module):
    """离散动作空间的 Actor-Critic 网络。
    对应第 1 章 Categorical 分布 + 第 3 章 Actor-Critic 架构。"""

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        # 分离主干，避免第 6 章提到的损失干扰
        self.actor_backbone = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
        )
        self.actor_head = layer_init(nn.Linear(64, action_dim), std=0.01)

        self.critic_backbone = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
        )
        self.critic_head = layer_init(nn.Linear(64, 1), std=1.0)

    def get_value(self, x):
        """对应第 2 章 V(s) 的函数近似。"""
        return self.critic_head(self.critic_backbone(x)).squeeze(-1)

    def get_action_and_value(self, x, action=None):
        """返回 (action, logprob, entropy, value) 四个量。
        对应第 1 章 Categorical(logits + softmax) 的实现形式。
        当 action=None 时采样新动作（rollout 阶段）；
        当 action 给定重算 logprob（更新阶段）。"""
        logits = self.actor_head(self.actor_backbone(x))
        # 对应第 1 章：Categorical 分布由 logits 参数化
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        # 对应第 1 章：log π(a|s) = log softmax(logits)[a]
        log_prob = dist.log_prob(action)
        # 对应第 1 章：H[π] = -Σ_a π(a) log π(a)
        entropy = dist.entropy()
        value = self.get_value(x)
        return action, log_prob, entropy, value
```

<a name="s7-4-2"></a>

### 7.4.2 Rollout 存储

```python
class RolloutBuffer:
    """Rollout 缓冲区，形状约定为 (T, N, ...)。
    对应第 6 章伪代码中的数据采集阶段。"""

    def __init__(self, num_steps, num_envs, obs_shape, action_dim, device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs = np.zeros((num_steps, num_envs) + obs_shape, dtype=np.float32)
        self.actions = np.zeros((num_steps, num_envs), dtype=np.int64)
        self.logprobs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        # 关键：dones 仅记录 terminated，不包含 truncated
        self.dones = np.zeros((num_steps, num_envs), dtype=np.float32)
        # truncated 单独存储，用于 GAE 的正确 bootstrap
        self.truncateds = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)
        # 每个环境在最后一步之后的 next_value，用于 GAE 的起始 bootstrap
        self.next_values = np.zeros(num_envs, dtype=np.float32)
        self.device = device

    def store(self, step, obs, action, logprob, reward, done, truncated, value):
        self.obs[step] = obs
        self.actions[step] = action
        self.logprobs[step] = logprob
        self.rewards[step] = reward
        self.dones[step] = done          # 仅 terminated
        self.truncateds[step] = truncated
        self.values[step] = value

    def set_next_values(self, next_values):
        """设置 rollout 结束后各环境的 V(s_{T})，用于 GAE 递推起点。"""
        self.next_values = next_values
```

<a name="s7-4-3"></a>

### 7.4.3 GAE 与 Return Target 计算

```python
def compute_gae(buffer, gamma=0.99, lam=0.95):
    """广义优势估计（GAE），严格对应第 4 章递推式：
       Â_t = δ_t + γλ Â_{t+1}
    其中 δ_t = r_t + γ·(1-terminated_t)·V(s_{t+1}) - V(s_t)

    ⚠️ 此处与理论的差异说明：
    - 理论上 GAE 是对无穷级数的截断近似，代码中只有 T 步有限窗口。
    - truncated 时仍 bootstrap（只用 terminated 屏蔽），见 7.1.2 节。
    """
    T, N = buffer.num_steps, buffer.num_envs
    advantages = np.zeros((T, N), dtype=np.float32)
    lastgaelam = 0.0

    for t in reversed(range(T)):
        if t == T - 1:
            # 最后一步使用存储的 next_value 作为 V(s_{T})
            nextnonterminal = 1.0 - buffer.dones[t]
            nextvalues = buffer.next_values
        else:
            nextnonterminal = 1.0 - buffer.dones[t + 1]
            nextvalues = buffer.values[t + 1]

        # 对应第 2 章 TD 误差 δ_t
        delta = (buffer.rewards[t]
                 + gamma * nextnonterminal * nextvalues
                 - buffer.values[t])
        # 对应第 4 章递推式 Â_t = δ_t + γλÂ_{t+1}
        advantages[t] = lastgaelam = (
            delta + gamma * lam * nextnonterminal * lastgaelam
        )

    # 对应第 4 章：V_t^target = Â_t + V(s_t)
    returns = advantages + buffer.values
    return advantages, returns
```

<a name="s7-4-4"></a>

### 7.4.4 Mini-batch 更新循环

```python
def update(agent, optimizer, buffer, advantages, returns,
           num_epochs=10, batch_size=64, clip_eps=0.2,
           vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5,
           target_kl=None):
    """PPO 更新，对应第 6 章完整损失函数：
       L^PPO = L^CLIP - c_1·L^VF + c_2·S

    参数：
        num_epochs (K): 对同一批数据的重复利用次数
        batch_size: mini-batch 大小
        clip_eps (ε): PPO-Clip 的裁剪半径
        vf_coef (c_1): 价值损失系数
        ent_coef (c_2): 熵正则系数
        max_grad_norm: 梯度裁剪的全局 L2 范数上限
        target_kl: 可选的 early stopping KL 阈值
    """
    # Flatten buffer 为 (T*N, ...)
    b_obs = torch.tensor(buffer.obs.reshape(-1, *buffer.obs.shape[2:]),
                         dtype=torch.float32, device=agent.device)
    b_actions = torch.tensor(buffer.actions.reshape(-1),
                             dtype=torch.int64, device=agent.device)
    b_logprobs = torch.tensor(buffer.logprobs.reshape(-1),
                              dtype=torch.float32, device=agent.device)
    b_advantages = torch.tensor(advantages.reshape(-1),
                                dtype=torch.float32, device=agent.device)
    b_returns = torch.tensor(returns.reshape(-1),
                             dtype=torch.float32, device=agent.device)
    b_values = torch.tensor(buffer.values.reshape(-1),
                            dtype=torch.float32, device=agent.device)

    total_samples = b_obs.shape[0]
    indices = np.arange(total_samples)

    approx_kl = None
    for epoch in range(num_epochs):
        # 每个 epoch 重新打乱，避免同一顺序重复更新
        np.random.shuffle(indices)
        for start in range(0, total_samples, batch_size):
            end = start + batch_size
            mb_idx = indices[start:end]

            # ---- 重新计算当前策略下的 logprob 和 value ----
            _, new_logprob, entropy, new_value = agent.get_action_and_value(
                b_obs[mb_idx], b_actions[mb_idx]
            )

            # ---- 重要性采样比率，对应第 5 章 r_t(θ) ----
            # ⚠️ 数值稳定性：在对数域计算 ratio，避免 exp 溢出
            logratio = new_logprob - b_logprobs[mb_idx]
            ratio = logratio.exp()

            # ---- 优势归一化，对应第 4 章讨论 ----
            # 在 mini-batch 级别做归一化（而非整个 rollout），
            # 实践中更稳定，减少 batch 间尺度差异的影响
            mb_advantages = b_advantages[mb_idx]
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                mb_advantages.std() + 1e-8
            )

            # ---- L^CLIP，对应第 6 章裁剪目标 ----
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(
                ratio, 1.0 - clip_eps, 1.0 + clip_eps
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            # ---- L^VF，对应第 6 章价值函数损失 ----
            # 此处未使用价值裁剪（clip_value_loss），因其争议较大且收益不确定
            v_loss = ((new_value - b_returns[mb_idx]) ** 2).mean()

            # ---- 熵正则 S，对应第 6 章鼓励探索项 ----
            entropy_loss = entropy.mean()

            # ---- 合并损失 L^PPO = L^CLIP - c_1·L^VF + c_2·S ----
            # 注意符号：pg_loss 已取负号（最大化→最小化），
            # entropy_loss 前加 ent_coef 是因为我们希望最大化熵
            loss = pg_loss + vf_coef * v_loss - ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            # 对应第 1 章梯度裁剪：裁剪全局 L2 范数，不是逐元素 clip
            # ⚠️ 这与 PPO-Clip 的 clip 完全是两回事！
            # PPO-Clip 裁剪的是比率 r_t；此处裁剪的是梯度向量的模长
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

        # ---- 近似 KL 监控与 early stopping ----
        # 对应 Schulman 的二阶近似：KL ≈ E[r - 1 - log r]
        # 恒非负，比一阶近似 E[old_logp - new_logp] 更稳定
        with torch.no_grad():
            _, new_lp, _, _ = agent.get_action_and_value(b_obs, b_actions)
            lr = (new_lp - b_logprobs).exp()
            approx_kl = (lr - 1.0 - (new_lp - b_logprobs)).mean().item()

        if target_kl is not None and approx_kl > target_kl:
            print(f"Early stopping at epoch {epoch}: approx_kl={approx_kl:.4f}")
            break

    return {
        "policy_loss": pg_loss.item(),
        "value_loss": v_loss.item(),
        "entropy": entropy_loss.item(),
        "approx_kl": approx_kl,
    }
```

<a name="s7-4-5"></a>

### 7.4.5 学习率线性衰减

对应第 6 章典型超参中提到的线性衰减策略：

```python
def get_lr(base_lr, current_step, total_steps):
    """线性衰减学习率：lr = base_lr × (1 - step/total_steps)。
    对应第 6 章推荐的 lr=3e-4 线性衰减。"""
    frac = 1.0 - current_step / total_steps
    return max(frac * base_lr, 0.0)  # 防止负学习率

# 在每个更新步之前调用：
# lr = get_lr(base_lr=3e-4, current_step=num_updates, total_steps=total_updates)
# for param_group in optimizer.param_groups:
#     param_group["lr"] = lr
```

<a name="s7-4-6"></a>

### 7.4.6 梯度裁剪澄清

`torch.nn.utils.clip_grad_norm_(params, max_grad_norm=0.5)` 裁剪的是所有参数梯度的**全局 L2 范数**：

$$
\|g\|_2 = \sqrt{\sum_i g_i^2}
$$

若 $\|g\|_2 > \text{max\_grad\_norm}$，则将所有梯度分量等比例缩放至 $\text{max\_grad\_norm}$。这与 PPO-Clip 中 $\text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)$ 的逐元素裁剪机制完全不同。前者是优化层面的稳定性手段（对应第 1 章），后者是算法层面的策略更新约束（对应第 6 章）。**这两个 "clip" 是 PPO 实现中最容易混淆的概念，务必区分。**

<a name="s7-5"></a>

## 7.5 数值稳定性与实现细节的数学解释

<a name="s7-5-1"></a>

### 7.5.1 对数域计算比率

代码中始终使用 `logratio = new_logprob - old_logprob` 然后 `ratio = logratio.exp()`，而非直接 `ratio = new_prob / old_prob`。原因：概率值可能极小（如 $10^{-30}$），直接相除会导致浮点下溢或上溢。对数域减法等价于原始域除法，且 `log_prob` 本身由分布对象精确计算，数值范围可控。仅在最终需要 `ratio` 时才做一次 `exp`，此时 `logratio` 通常在 $[-10, 10]$ 范围内，不会溢出。

<a name="s7-5-2"></a>

### 7.5.2 近似 KL 的选择

两种常见近似：

1. **一阶近似**：$\hat{D}_{KL}^{(1)} = \mathbb{E}[\log \pi_{\text{old}} - \log \pi_{\text{new}}]$。简单但可能为负（违反 KL 非负性），不适合做 early stopping 判据。
2. **Schulman 二阶近似**：$\hat{D}_{KL}^{(2)} = \mathbb{E}[r - 1 - \log r]$，其中 $r = \pi_{\text{new}} / \pi_{\text{old}}$。由不等式 $x - 1 \geq \log x$ 保证恒非负，且在 $r \approx 1$ 时二阶逼近真实 KL。

推荐使用第二种作为 early stopping 判据，因为它不会出现虚假的负值触发。

<a name="s7-5-3"></a>

### 7.5.3 优势归一化的位置

- **整个 rollout 归一化**：统计量基于全部 $T \times N$ 个样本，方差估计更准，但不同 mini-batch 的优势尺度相同，丧失了 batch 内的自适应调节能力。
- **Mini-batch 内归一化**：每个 mini-batch 独立标准化。实践中更常用，因为：(1) 减少了不同 mini-batch 之间优势尺度的差异对梯度的影响；(2) 等价于对每个 mini-batch 做了局部的基线校正，进一步降低方差。

<a name="s7-5-4"></a>

### 7.5.4 价值裁剪

价值裁剪将 $V_\theta(s)$ 限制在 $[V_{\text{old}}(s) - \epsilon_v, V_{\text{old}}(s) + \epsilon_v]$ 内，意图防止价值函数单次更新过大。但如第 6 章所述，其收益存在争议：价值函数的优化目标是 MSE，不像策略目标那样有单调改进保证，裁剪反而可能阻碍 critic 快速收敛。多数现代实现默认关闭价值裁剪，仅在发现 `value_loss` 剧烈震荡时才考虑启用。

<a name="s7-6"></a>

## 7.6 训练诊断指标与调参方法论

<a name="s7-6-1"></a>

### 7.6.1 必须监控的量及其健康范围

| 指标 | 数学含义 | 健康范围 | 异常排查方向 |
|------|---------|---------|-------------|
| Episode Return | $\mathbb{E}[\sum \gamma^t r_t]$，主性能指标 | 持续上升 | 若不升：检查奖励归一化、网络容量、lr |
| approx_kl | $\mathbb{E}[r-1-\log r]$，策略变化幅度 | < 0.015~0.02 | 过大：降 lr、减 K、增大 ε |
| clip_fraction | 被 clip 的样本占比 | 0.05~0.2 | ≈0：ε 太松或更新太保守；≈1：策略剧变 |
| Entropy | $H[\pi]$，探索程度 | 缓慢下降 | 突然崩塌：log_std 塌缩、lr 过高 |
| explained_variance | $1 - \frac{\text{Var}(R-V)}{\text{Var}(R)}$，critic 拟合质量 | > 0.5 | < 0：critic 比常数预测还差，检查网络/lr/归一化 |
| value_loss | $\mathbb{E}[(V_\theta(s) - \hat{R})^2]$ | 持续下降 | 不降或震荡：critic 容量不足、lr 不当 |
| policy_loss | $-\mathbb{E}[\min(r\hat{A}, \text{clip}(r)\hat{A})]$ | 无固定范围 | 突增：结合 approx_kl 判断 |
| grad_norm | $\|\nabla L\|_2$（裁剪前） | 平稳 | 突增：不稳定前兆，检查奖励尺度、观测归一化 |

<a name="s7-6-2"></a>

### 7.6.2 调参优先级

基于 Andrychowicz et al. (2021) 的大规模实证结论：

1. **先调学习率与网络结构**：这两个因素对性能影响最大。lr 过高导致 KL 爆炸，过低导致收敛极慢。
2. **再调 $\varepsilon$ 与 $K$**：$\varepsilon$ 控制单次更新的信任域大小，$K$ 控制数据复用程度。
3. **$\gamma$ 和 $\lambda$ 一般用默认值**（0.99 和 0.95），除非任务有特殊的时间尺度需求。

<a name="s7-6-3"></a>

### 7.6.3 训练崩溃排查清单

当训练出现异常时，按以下顺序排查：

1. **看 `explained_variance`**：若 < 0 或不升，critic 有问题 → 检查 critic 网络结构、lr、观测归一化。
2. **看 `entropy`**：若突然崩塌 → 检查 `log_std` 是否有 clamp、lr 是否过高。
3. **看 `approx_kl`**：若持续 > 0.02 → 降 lr、减少 K、增大 ε。
4. **检查观测/奖励归一化**：缺失归一化是 PPO 崩溃的最常见原因之一。

<a name="s7-6-4"></a>

### 7.6.4 观测归一化与奖励归一化

**观测归一化（Observation Normalization）**：使用 Welford 在线算法维护 running mean/std：

$$
\bar{x}_n = \bar{x}_{n-1} + \frac{x_n - \bar{x}_{n-1}}{n}, \quad M_n = M_{n-1} + (x_n - \bar{x}_{n-1})(x_n - \bar{x}_n)
$$

其中 $\sigma_n = \sqrt{M_n / n}$。PPO 对观测归一化极其敏感，尤其是多环境并行时，统计量必须用**全体环境的数据**更新，而非单个环境。

**奖励缩放（Reward Scaling）**：将奖励除以 running std，使回报的尺度大致在 $[-1, 1]$ 范围内。**注意**：奖励缩放会改变 $\gamma$ 的有效含义——缩放后的折扣因子等效于 $\gamma' = \gamma \cdot \sigma_{\text{old}} / \sigma_{\text{new}}$，在 running std 尚未收敛的训练初期，这种效应尤为明显。因此建议在 running statistics 稳定后再评估最终性能。

<a name="s7-7"></a>

## 7.7 常见坑清单

| # | 症状 | 根因 | 修复 |
|---|------|------|------|
| 1 | 价值函数系统性偏低，return 远低于预期 | truncated 时未 bootstrap，将"时间到"误当"失败" | `nextnonterminal` 只用 `terminated`，不用 `truncated` |
| 2 | GAE 在 episode 边界处出现异常大的优势值 | 未在 terminated 处重置 `lastgaelam` | 确保 `nextnonterminal = 1.0 - dones[t+1]` 正确断开递推链 |
| 3 | 训练初期正常，后期性能骤降 | 优势在整个 batch 上归一化，mini-batch 间尺度不一致 | 改为 mini-batch 内归一化 |
| 4 | entropy 突然降到接近 0，策略不再探索 | `log_std` 未约束，塌缩到极小值 | 添加 `log_std.clamp(min=-20, max=2)` |
| 5 | policy_loss 和 value_loss 互相干扰，训练震荡 | actor/critic 共享主干但损失未合理加权 | 分离主干，或调低 `vf_coef` |
| 6 | 同一 epoch 内 loss 几乎不变，更新无效 | mini-batch 采样未打乱，重复更新相同数据 | 每个 epoch 开始时 `np.random.shuffle(indices)` |
| 7 | approx_kl 持续 > 0.05，策略剧烈摆动 | 学习率过高 | 降低 lr 或使用线性衰减 |
| 8 | 每个 mini-batch 样本极少（如 < 32），梯度噪声大 | 向量环境数量 N 与 mini-batch size 配置不当 | 增加 N 或减小 batch_size，确保每 batch ≥ 64 样本 |
| 9 | 更新阶段的 logprob 与 rollout 阶段不一致 | rollout 时未在 `torch.no_grad()` 下同时取出 action 和 logprob | 确保 `action, logprob, _, _ = agent.get_action_and_value(obs)` 在同一次前向中完成 |
| 10 | 评测性能远低于训练性能 | 评测时仍使用采样动作而非确定性动作 | 评测时用 `dist.mode`（离散）或 `mean`（连续）代替 `sample()` |

<a name="s7-8"></a>

## 7.8 推荐学习资源与进阶路径

<a name="s7-8-1"></a>

### 7.8.1 论文清单（按阅读顺序）

1. **Sutton & Barto (2018)**《Reinforcement Learning: An Introduction》第 13 章 — 策略梯度方法的经典教材基础。
2. **Schulman et al. (2015)**《Trust Region Policy Optimization》— TRPO 原始论文，理解信任域与单调改进界。
3. **Schulman et al. (2015)**《High-Dimensional Continuous Control Using Generalized Advantage Estimation》— GAE 的提出与偏差-方差分析。
4. **Schulman et al. (2017)**《Proximal Policy Optimization Algorithms》— PPO 原始论文，重点读 Section 3 的目标函数推导。
5. **Engstrom et al. (2020)**《Implementation Matters in Deep RL》— 揭示实现细节对性能的巨大影响，必读。
6. **Andrychowicz et al. (2021)**《What Matters in On-Policy RL? A Large-Scale Empirical Study》— 大规模消融实验，调参优先级的实证依据。
7. **Hilton et al. (2022)**《Batch Size-Invariance for Policy Optimization》— mini-batch 大小对学习率的影响及修正方法。
8. **Stiennon et al. (2020)**《Learning to Summarize with Human Feedback》— RLHF 的起点，PPO 在文本生成中的首次成功应用。
9. **Ouyang et al. (2022)**《Training Language Models to Follow Instructions with Human Feedback》— InstructGPT，RLHF 工业化落地的里程碑。
10. **Shao et al. (2024)**《DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models》— GRPO（Group Relative Policy Optimization），PPO 在推理任务中的最新变体。

<a name="s7-8-2"></a>

### 7.8.2 代码库

| 代码库 | 定位 | 适用场景 |
|--------|------|---------|
| **CleanRL** | 单文件实现，代码与论文公式逐行对应 | **强烈推荐入门**，最适合逐行对照公式学习 |
| **Stable-Baselines3** | 工业级封装，API 简洁 | 快速跑实验、生产部署 |
| **OpenAI Spinning Up** | 配套教程质量最高 | 理论学习与代码实践并行 |
| **TRL** (Hugging Face) | RLHF/PPO 专用 | LLM 对齐、RLHF 研究 |

<a name="s7-8-3"></a>

### 7.8.3 务实的学习路径建议

1. **跑通基线**：用 CleanRL 的 `ppo.py` 在 CartPole-v1 上跑通，逐行对照第 6 章伪代码与本章实现，确认每个公式的代码位置。
2. **主动破坏实验**：故意去掉 truncated 处理、去掉优势归一化、将 `log_std` 的 clamp 移除、把学习率调高 10 倍……观察每种破坏对应的训练曲线变化。这是理解"为什么需要这个设计"最有效的方式。
3. **升级任务**：在 LunarLander-v3 上复现，体会网络容量、归一化、超参调整的必要性。
4. **阅读 RLHF 实现**：对照 TRL 的 PPO trainer，理解 token 级别的 PPO 与经典 PPO 的差异（KL 奖励塑形、reference model 等）。
5. **尝试改进**：在理解标准 PPO 后，尝试实现 PPO-Penalty、GRPO 等变体，加深对算法设计空间的理解。

<a name="s7-9"></a>

## 7.9 本章小结与下一步

**要点回顾：**

1. **terminated 与 truncated 的正确处理**是 PPO 实现中最关键的细节之一，直接影响 GAE 和价值函数的正确性。
2. **向量环境**是 PPO 的工程基础设施，决定了数据采集效率和 batch 质量。
3. **网络初始化**（特别是输出层的小权重正交初始化）对策略梯度的初始探索行为有决定性影响。
4. **数值稳定性**贯穿始终：对数域计算比率、`log_std` 约束、mini-batch 级优势归一化、梯度范数裁剪——每一处都有明确的数学动机。
5. **诊断指标**是调参的指南针：`explained_variance`、`approx_kl`、`entropy`、`clip_fraction` 四个量足以定位绝大多数训练问题。

**学完这份文档之后该做什么：**

不要停留在"读懂了"的阶段。打开 CleanRL 的 `ppo.py`，在一个你熟悉的环境上从头跑一遍，然后按照 7.8.3 节的路径主动破坏、修复、升级。当你能不看任何参考独立写出一个能稳定训练 LunarLander 的 PPO，并且能解释每一行代码背后的数学理由时，才算真正掌握了这份文档的内容。之后可以进入 RLHF 领域，将 PPO 应用于 LLM 对齐——那里的挑战不再是算法本身，而是如何在巨大的模型规模和有限的反馈信号下保持 PPO 的稳定性。

---

[上一章](chapter-06.md) · [返回目录](../README.md) · [阅读路线与符号表](README.md)
