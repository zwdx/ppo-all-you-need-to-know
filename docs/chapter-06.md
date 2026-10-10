[上一章](chapter-05.md) · [返回目录](../README.md) · [阅读路线与符号表](README.md) · [下一章](chapter-07.md)

---

<a id="chapter-6"></a>

# 第 6 章 PPO 算法本体

**本章目录**

- [6.1 PPO 的问题定位：从信任域到一阶近似](#user-content-s6-1)
- [6.2 PPO-Clip 的完整构造](#user-content-s6-2)
- [6.3 PPO-Penalty：自适应 KL 惩罚](#user-content-s6-3)
- [6.4 完整的 PPO 损失函数](#user-content-s6-4)
- [6.5 PPO 完整算法伪代码](#user-content-s6-5)
- [6.6 PPO 的理论局限与已知问题](#user-content-s6-6)
- [6.7 PPO 在 RLHF 中的应用](#user-content-s6-7)
- [6.8 与其他方法的定位关系](#user-content-s6-8)
- [6.9 本章小结与自测](#user-content-s6-9)

<a id="s6-1"></a>

## 6.1 PPO 的问题定位：从信任域到一阶近似

在前述章节中，我们已经建立了完整的理论链条：第 3 章揭示了朴素策略梯度的三大缺陷——样本效率低（每条轨迹只能用一次）、步长敏感且崩溃不可恢复、梯度方差高；第 5 章则给出了 TRPO（Trust Region Policy Optimization）的解决方案：通过约束新旧策略之间的 KL 散度 $`\mathbb{E}_{s \sim d^{\pi_{\theta_{\text{old}}}}}[D_{\text{KL}}(\pi_{\theta_{\text{old}}}(\cdot|s) \| \pi_\theta(\cdot|s))] \leq \delta`$，保证每次更新的性能单调改进下界 $`J(\pi') \geq L_{\pi}(L') - C \cdot \max_s D_{\text{KL}}`$。然而 TRPO 的实现代价极高：需要计算 Fisher 信息矩阵与向量的乘积、用共轭梯度法求解自然梯度方向、再执行线搜索确保约束满足。这些二阶优化步骤使得 TRPO 难以与随机梯度下降（SGD）兼容，也无法直接复用深度学习框架中成熟的优化器与分布式训练基础设施。

近端策略优化（Proximal Policy Optimization, PPO）的核心动机正是：**用一阶优化方法近似实现信任域约束**。具体而言，PPO 放弃了 TRPO 中对 $`\max_s D_{\text{KL}}`$ 的严格约束，转而通过对重要性采样比率 $`r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{\text{old}}}(a_t|s_t)`$ 的逐点截断（clipping）或自适应 KL 惩罚，在同一个 mini-batch SGD 框架内限制策略更新幅度。这使得 PPO 既能像 TRPO 一样防止过大的策略更新导致性能崩溃，又能像 A2C 一样用同一批采样数据做多次梯度更新以提升样本效率，同时完全兼容标准的一阶优化器。

本章将完整展开 PPO 的两种变体——PPO-Clip 与 PPO-Penalty——的数学构造、损失函数设计、算法流程、理论局限以及在 RLHF 中的应用。所有推导均建立在前序章节已给出的结论之上，不再重复。

<a id="s6-2"></a>

## 6.2 PPO-Clip 的完整构造

<a id="s6-2-1"></a>

### 6.2.1 目标函数的定义

PPO-Clip 的核心是一个经过裁剪的替代目标函数（clipped surrogate objective）。设当前策略参数为 $`\theta`$，采集数据时的旧策略参数为 $`\theta_{\text{old}}`$，定义重要性采样比率：

```math
r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}
```

其中 $`(s_t, a_t)`$ 来自旧策略 $`\pi_{\theta_{\text{old}}}`$ 生成的 rollout 数据。$`\hat{A}_t`$ 是由第 4 章 GAE 方法计算得到的优势估计。PPO-Clip 的目标函数为：

```math
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \; \text{clip}\big(r_t(\theta), 1-\varepsilon, 1+\varepsilon\big) \hat{A}_t \right) \right]
```

其中 $`\text{clip}(x, l, u) = \max(l, \min(x, u))`$ 将 $`x`$ 截断到区间 $`[l, u]`$ 内，$`\varepsilon \gt  0`$ 是裁剪半径（典型取值 0.1～0.3，常用 0.2）。$`\mathbb{E}_t`$ 表示对时间步 $`t`$ 在 rollout 数据上的经验平均。

这个目标函数包含三个关键组件：
- **未裁剪项** $`r_t(\theta) \hat{A}_t`$：即第 5 章中的标准替代目标，是真实性能差异 $`J(\pi_\theta) - J(\pi_{\theta_{\text{old}}})`$ 的一阶近似。
- **裁剪项** $`\text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_t`$：将比率限制在 $`[1-\varepsilon, 1+\varepsilon]`$ 内后再乘以优势。
- **$`\min`$ 操作**：取两者中较小的一个，构造出一个悲观下界（pessimistic lower bound）。

我们的目标是**最大化** $`L^{\text{CLIP}}(\theta)`$。在实际实现中，通常对 $`-L^{\text{CLIP}}(\theta)`$ 做梯度下降最小化。

<a id="s6-2-2"></a>

### 6.2.2 六种情形的完整分析

为了理解 clip 与 min 的联合作用，我们需要分 $`\hat{A}_t \gt  0`$ 和 $`\hat{A}_t \lt  0`$ 两大情形，每种情形下再按 $`r_t(\theta)`$ 落在三个区间讨论。记 $`r = r_t(\theta)`$，$`A = \hat{A}_t`$，$`c = \text{clip}(r, 1-\varepsilon, 1+\varepsilon)`$。

**情形一：$`A \gt  0`$（该动作比平均水平好，我们希望增大其概率）**

| $`r`$ 所在区间 | $`c`$ 的值 | $`rA`$ vs $`cA`$ | $`\min`$ 取值 | 梯度 | 行为解释 |
|:---:|:---:|:---:|:---:|:---:|:---|
| $`r \lt  1-\varepsilon`$ | $`1-\varepsilon`$ | $`rA \lt  cA`$（因 $`r \lt  1-\varepsilon`$） | $`rA`$（未裁剪项） | $`\nabla_\theta r \cdot A \neq 0`$ | 当前概率远低于旧策略，未裁剪项更小被选中，保留完整梯度鼓励增大概率 |
| $`1-\varepsilon \leq r \leq 1+\varepsilon`$ | $`r`$ | $`rA = cA`$ | $`rA = cA`$ | $`\nabla_\theta r \cdot A \neq 0`$ | 比率在安全区间内，两项相等，等价于未裁剪的策略梯度 |
| $`r \gt  1+\varepsilon`$ | $`1+\varepsilon`$ | $`rA \gt  cA`$（因 $`r \gt  1+\varepsilon`$） | $`cA`$（裁剪项） | $`0`$（$`c`$ 对 $`\theta`$ 为常数） | 概率已大幅增加超出安全范围，裁剪项更小被选中，梯度为零，阻止继续增大 |

**情形二：$`A \lt  0`$（该动作比平均水平差，我们希望减小其概率）**

| $`r`$ 所在区间 | $`c`$ 的值 | $`rA`$ vs $`cA`$ | $`\min`$ 取值 | 梯度 | 行为解释 |
|:---:|:---:|:---:|:---:|:---:|:---|
| $`r \lt  1-\varepsilon`$ | $`1-\varepsilon`$ | $`rA \gt  cA`$（因 $`A\lt 0`$，$`r\lt 1-\varepsilon`$ 使 $`rA`$ 负得少） | $`cA`$（裁剪项） | $`0`$ | 概率已大幅减小超出安全范围，裁剪项更小被选中，梯度为零，阻止继续减小 |
| $`1-\varepsilon \leq r \leq 1+\varepsilon`$ | $`r`$ | $`rA = cA`$ | $`rA = cA`$ | $`\nabla_\theta r \cdot A \neq 0`$ | 比率在安全区间内，等价于未裁剪的策略梯度 |
| $`r \gt  1+\varepsilon`$ | $`1+\varepsilon`$ | $`rA \lt  cA`$（因 $`A\lt 0`$，$`r\gt 1+\varepsilon`$ 使 $`rA`$ 负得多） | $`rA`$（未裁剪项） | $`\nabla_\theta r \cdot A \neq 0`$ | 当前概率远高于旧策略但动作实际很差，未裁剪项更小被选中，保留完整梯度鼓励减小概率 |

以上六行可以浓缩为一条核心规则：**$`\min`$ 操作总是选择对当前参数更新"更保守"的那一项**。当策略更新方向与优势信号一致（$`A\gt 0`$ 时增大 $`r`$，$`A\lt 0`$ 时减小 $`r`$）且幅度超过 $`\varepsilon`$ 时，裁剪项生效、梯度归零，阻止过度更新；当策略更新方向与优势信号相反（例如 $`A\gt 0`$ 但 $`r`$ 反而变小了），说明当前策略在该状态-动作对上犯了错，此时未裁剪项更小被选中，梯度不受阻碍，允许纠错。

<a id="s6-2-3"></a>

### 6.2.3 悲观下界的设计哲学与不对称性

PPO-Clip 最容易被误解的点在于其**不对称性**：它只阻止朝有利方向的过度更新，不阻止朝不利方向的回退。这不是 bug，而是刻意的设计。

从优化角度看，$`L^{\text{CLIP}}(\theta)`$ 是真实替代目标 $`L^{\text{SURR}}(\theta) = \mathbb{E}_t[r_t(\theta)\hat{A}_t]`$ 的一个**逐点下界**：对于每个样本 $`t`$，都有

```math
\min\big(r_t(\theta)\hat{A}_t, \; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\big) \leq r_t(\theta)\hat{A}_t
```

因此 $`L^{\text{CLIP}}(\theta) \leq L^{\text{SURR}}(\theta)`$。最大化一个下界是安全的：如果下界提升了，真实目标至少不会更差（在局部近似成立的范围内）。这就是"悲观"的含义——我们宁可低估收益，也不愿高估后过度乐观地迈出太大一步。

对比 TRPO 的理论保障：TRPO 通过约束 $`\max_s D_{\text{KL}}`$ 得到了 $`J(\pi') \geq L_\pi(\pi') - C \cdot \max_s D_{\text{KL}}`$ 这一全局下界，其中 $`C`$ 依赖于优势函数的最大绝对值和折扣因子。PPO-Clip 没有这样的全局保证，因为 clip 是对单个样本比率的逐点操作，不等价于对状态级 KL 散度的约束。但在实践中，这种启发式的悲观下界足够有效，且计算成本远低于 TRPO 的二阶优化。

关于不对称性的直觉：当 $`A \gt  0`$ 且 $`r \gt  1+\varepsilon`$ 时，策略已经大幅增加了该好动作的概率，继续增加的边际收益递减而风险递增（可能导致其他好动作的概率被挤压），所以截断是合理的。但当 $`A \gt  0`$ 且 $`r \lt  1-\varepsilon`$ 时，策略反而降低了好动作的概率，这是一个明确的错误信号，不应该被截断——让梯度自由流动以尽快纠正。同理适用于 $`A \lt  0`$ 的对称情形。

<a id="s6-2-4"></a>

### 6.2.4 梯度一致性验证

一个自然的疑问是：在 $`\theta = \theta_{\text{old}}`$ 处，$`L^{\text{CLIP}}`$ 的梯度是否等于标准策略梯度？答案是肯定的。

当 $`\theta = \theta_{\text{old}}`$ 时，$`r_t(\theta) = 1`$，对所有 $`t`$ 成立。由于 $`1 \in [1-\varepsilon, 1+\varepsilon]`$（只要 $`\varepsilon \gt  0`$），clip 不生效，即 $`\text{clip}(1, 1-\varepsilon, 1+\varepsilon) = 1`$。此时两项相等：

```math
\min(1 \cdot \hat{A}_t, \; 1 \cdot \hat{A}_t) = \hat{A}_t
```

因此在 $`\theta = \theta_{\text{old}}`$ 的一个邻域内（$`r_t`$ 尚未离开 $`[1-\varepsilon, 1+\varepsilon]`$），$`L^{\text{CLIP}}(\theta)`$ 与 $`L^{\text{SURR}}(\theta)`$ 完全一致，其梯度为：

```math
\nabla_\theta L^{\text{CLIP}}(\theta)\big|_{\theta=\theta_{\text{old}}} = \mathbb{E}_t \left[ \nabla_\theta r_t(\theta)\big|_{\theta=\theta_{\text{old}}} \cdot \hat{A}_t \right] = \mathbb{E}_t \left[ \nabla_\theta \log \pi_\theta(a_t|s_t)\big|_{\theta=\theta_{\text{old}}} \cdot \hat{A}_t \right]
```

这正是第 3 章策略梯度定理给出的形式（用 $`\hat{A}_t`$ 替代 $`Q^\pi`$ 作为无偏估计）。这保证了 PPO-Clip 在更新起点处的搜索方向与策略梯度一致，clip 仅在偏离起点足够远时才介入。

<a id="s6-3"></a>

## 6.3 PPO-Penalty：自适应 KL 惩罚

<a id="s6-3-1"></a>

### 6.3.1 目标函数

PPO 的另一种变体保留了完整的替代目标形式，但将信任域约束转化为惩罚项加入目标函数：

```math
L^{\text{KLPEN}}(\theta) = \mathbb{E}_t \left[ r_t(\theta) \hat{A}_t \right] - \beta \cdot \mathbb{E}_t \left[ D_{\text{KL}}\big(\pi_{\theta_{\text{old}}}(\cdot|s_t) \;\|\; \pi_\theta(\cdot|s_t)\big) \right]
```

其中 $`\beta \gt  0`$ 是 KL 惩罚系数，$`D_{\text{KL}}`$ 的定义见第 1 章。与 TRPO 的区别在于：TRPO 将 KL 作为硬约束（$`\leq \delta`$），而 PPO-Penalty 将其作为软惩罚项，并通过自适应机制调节 $`\beta`$。

<a id="s6-3-2"></a>

### 6.3.2 自适应 β 更新规则

固定 $`\beta`$ 的问题是：$`\beta`$ 太大则策略几乎不更新，$`\beta`$ 太小则约束失效。Schulman 等人在原论文中提出了以下自适应规则：

在每个 epoch（或每若干步）结束后，计算当前批次数据上的平均 KL 散度 $`d = \mathbb{E}_t[D_{\text{KL}}]`$，然后按如下规则更新 $`\beta`$：

```math
\beta \leftarrow \begin{cases}
\beta / 2 & \text{if } d < d_{\text{targ}} / 1.5 \\
\beta \times 2 & \text{if } d > d_{\text{targ}} \times 1.5 \\
\beta & \text{otherwise}
\end{cases}
```

其中 $`d_{\text{targ}}`$ 是目标 KL 散度，通常设为 0.01～0.02。$`\beta`$ 的初始值通常设为 1。这个规则的直觉是：如果实际 KL 远低于目标，说明约束过紧，放松惩罚以加速学习；如果实际 KL 远高于目标，说明约束过松，加强惩罚以防止过大更新。乘法因子 2 和阈值因子 1.5 提供了足够的滞后带（hysteresis），避免 $`\beta`$ 在临界值附近频繁振荡。

<a id="s6-3-3"></a>

### 6.3.3 PPO-Clip 与 PPO-Penalty 的对比

| 维度 | PPO-Clip | PPO-Penalty |
|:---|:---|:---|
| 目标函数形式 | 裁剪后的悲观下界 | 完整替代目标 + KL 惩罚 |
| 超参数 | $`\varepsilon`$（裁剪半径） | $`\beta`$（惩罚系数）+ $`d_{\text{targ}}`$ |
| 调参难度 | $`\varepsilon`$ 对性能不太敏感，0.2 几乎是万能默认值 | $`\beta`$ 的自适应虽有帮助但仍需关注 $`d_{\text{targ}}`$ 的选择 |
| 理论血统 | 启发式，无严格单调改进保证 | 更接近 TRPO 的拉格朗日松弛形式 |
| 实现复杂度 | 极简，几行代码 | 需额外计算 KL 并维护 $`\beta`$ 的自适应逻辑 |
| 社区采用度 | 事实标准，几乎所有库的默认实现 | 原论文实验显示效果相当，但社区极少使用 |

原论文（Schulman et al., 2017）的实验表明两种变体在 MuJoCo 控制任务上表现相当。但后续社区实践几乎一致选择了 Clip 变体，原因是其实现简单、超参鲁棒、且与各种工程技巧（优势归一化、观测归一化、学习率调度等）的组合效果更好。本章后续内容如无特别说明，"PPO" 均指 PPO-Clip。

<a id="s6-4"></a>

## 6.4 完整的 PPO 损失函数

<a id="s6-4-1"></a>

### 6.4.1 三项损失的组合

PPO 的实际优化目标不仅包含策略项，还包含价值函数拟合项和熵正则项。完整的单步损失为：

```math
L_t^{\text{PPO}}(\theta, \phi) = L_t^{\text{CLIP}}(\theta) - c_1 \, L_t^{\text{VF}}(\phi) + c_2 \, S[\pi_\theta](s_t)
```

其中 $`\phi`$ 是价值网络（critic）的参数，$`c_1, c_2 \gt  0`$ 是权重系数。下面逐项解释。

**策略项 $`L_t^{\text{CLIP}}(\theta)`$**：即 6.2 节定义的裁剪替代目标。注意这里写的是**最大化**目标，而在实际训练中我们通常构造一个**最小化**的损失函数。因此策略项在 loss 中取负号：$`-L_t^{\text{CLIP}}(\theta)`$。

**价值函数项 $`L_t^{\text{VF}}(\phi)`$**：critic 的均方误差回归损失：

```math
L_t^{\text{VF}}(\phi) = \big(V_\phi(s_t) - V_t^{\text{target}}\big)^2
```

其中 $`V_t^{\text{target}} = \hat{A}_t + V_{\phi_{\text{old}}}(s_t)`$ 是利用第 4 章 λ-return 构造的价值目标。这里用 $`V_{\phi_{\text{old}}}`$ 而非 $`V_\phi`$ 来计算目标，是为了在 mini-batch 更新过程中保持目标的稳定性（类似于 TD 学习中的 target network 思想）。$`L^{\text{VF}}`$ 本身是要最小化的，所以在总 loss 中以正号出现（前面已有负号的策略项与之方向相反）。

**熵正则项 $`S[\pi_\theta](s_t)`$**：策略在状态 $`s_t`$ 下的熵，定义为 $`S[\pi_\theta](s_t) = -\sum_a \pi_\theta(a|s_t) \log \pi_\theta(a|s_t)`$（离散动作空间）或对应的微分熵（连续动作空间）。熵衡量策略的随机性，最大化熵鼓励探索、防止策略过早收敛到确定性策略。因为我们要最大化熵，所以在最小化 loss 中它以负号出现。但注意上面公式中写的是 $`+ c_2 S`$，这是因为 $`S`$ 本身是非负的，而我们在最大化 $`L^{\text{PPO}}`$ 时希望 $`S`$ 越大越好；当转换为最小化 $`-L^{\text{PPO}}`$ 时，$`S`$ 项变为 $`-c_2 S`$，即最小化负熵等价于最大化熵。

**符号约定的澄清**：这是文献和代码中最容易混淆的地方。总结如下：
- 若定义 $`L^{\text{PPO}}`$ 为**最大化目标**：$`L^{\text{PPO}} = L^{\text{CLIP}} - c_1 L^{\text{VF}} + c_2 S`$
- 若定义 $`\mathcal{L}^{\text{PPO}}`$ 为**最小化损失**：$`\mathcal{L}^{\text{PPO}} = -L^{\text{CLIP}} + c_1 L^{\text{VF}} - c_2 S`$

两种写法完全等价，但阅读不同论文和代码时必须确认作者使用的是哪种约定。本章后续统一使用最大化目标的记号。

典型系数取值：$`c_1 = 0.5`$，$`c_2 = 0.01`$。$`c_1 = 0.5`$ 意味着 critic 损失的权重是策略损失的一半（考虑到 MSE 的量级通常大于策略梯度的量级，这个比例在实践中较为平衡）。$`c_2 = 0.01`$ 是一个较小的探索奖励，足以防止熵塌缩但不至于主导优化方向。

<a id="s6-4-2"></a>

### 6.4.2 价值函数裁剪的争议

PPO 原论文提到了一个价值函数裁剪（value function clipping）变体：

```math
L_t^{\text{VF,clip}}(\phi) = \max\!\left( \big(V_\phi(s_t) - V_t^{\text{target}}\big)^2, \; \big(\text{clip}(V_\phi(s_t), V_{\phi_{\text{old}}}(s_t)-\varepsilon, V_{\phi_{\text{old}}}(s_t)+\varepsilon) - V_t^{\text{target}}\big)^2 \right)
```

其动机与策略裁剪类似：限制价值网络的更新幅度，防止 critic 在某些状态下产生过大的预测跳变。注意这里用的是 $`\max`$ 而非 $`\min`$，因为价值函数损失是最小化目标，$`\max`$ 取两者中较大的损失值，相当于对价值更新施加了一个保守的上界。

然而，后续大规模实证研究（Andrychowicz et al., 2021, "What Matters in On-Policy Reinforcement Learning? A Large-Scale Empirical Study"）发现，价值函数裁剪在多数任务上**没有帮助甚至有害**。可能的解释是：critic 的目标是准确拟合真实的价值函数，裁剪人为限制了拟合能力，导致偏差累积；而策略裁剪之所以有效，是因为策略优化的 landscape 比价值拟合更脆弱。目前主流实现（如 Stable-Baselines3、CleanRL）默认不使用价值函数裁剪。本节客观呈现这一争议，读者在实践中应自行验证。

<a id="s6-4-3"></a>

### 6.4.3 Actor 与 Critic 的网络共享

PPO 原论文在部分任务（如 Atari）中使用了 actor 与 critic 共享主干网络（shared trunk）的架构：底层特征提取层共享，顶部分支为策略头和价值头。共享的好处是提升样本效率（特征表示只需学一次），坏处是两类损失的梯度会互相干扰——critic 的 MSE 梯度可能压倒策略梯度，或反之。

当使用共享架构时，$`c_1`$ 的选择尤为关键：过大会导致 critic 主导训练、策略停滞；过小则 critic 学不准、优势估计噪声大。独立网络架构避免了这个问题，但参数量翻倍。在连续控制任务（如 MuJoCo）中，独立网络更为常见；在像素级输入任务（如 Atari）中，共享架构更为常见。这一取舍属于工程细节，将在第 7 章详细讨论。

<a id="s6-5"></a>

## 6.5 PPO 完整算法伪代码

以下是 PPO-Clip 的数学形式伪代码，聚焦算法逻辑而非工程实现：

```
输入：环境 E，并行环境数 N，rollout 长度 T，epoch 数 K，mini-batch 大小 M
      裁剪半径 ε，折扣因子 γ，GAE 参数 λ，系数 c₁, c₂，学习率 α
初始化：策略参数 θ，价值网络参数 φ

for 每轮迭代 do
    // ===== 数据采集阶段 =====
    用 π_{θ_old} ← π_θ 与 N 个并行环境交互 T 步
    收集轨迹数据 {(s_t, a_t, r_t, s_{t+1})} for t = 0, ..., NT-1
    
    // ===== 优势与目标计算 =====
    for t = NT-2 downto 0 do
        δ_t ← r_t + γ V_φ(s_{t+1}) - V_φ(s_t)          // TD 误差
        Â_t ← δ_t + (γλ) Â_{t+1}                        // GAE 递推（第 4 章）
        V_t^target ← Â_t + V_φ(s_t)                      // λ-return 价值目标
    end for
    对 {Â_t} 做优势归一化：Â_t ← (Â_t - mean(Â)) / std(Â)
    
    // ===== 多 epoch 策略更新（PPO 核心）=====
    for k = 1 to K do
        将 NT 个样本随机打乱，分成 ⌈NT/M⌉ 个 mini-batch
        for 每个 mini-batch B do
            r_t(θ) ← π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)   for (s_t,a_t) ∈ B
            
            L_clip ← E_{t∈B}[ min(r_t Â_t, clip(r_t, 1-ε, 1+ε) Â_t) ]
            L_VF   ← E_{t∈B}[ (V_φ(s_t) - V_t^target)² ]
            S      ← E_{t∈B}[ H(π_θ(·|s_t)) ]             // 策略熵
            
            L ← L_clip - c₁ L_VF + c₂ S                   // 最大化目标
            
            θ ← θ + α ∇_θ L                                // 策略更新
            φ ← φ + α ∇_φ (-c₁ L_VF)                      // critic 更新
        end for
    end for
    
    θ_old ← θ    // 更新旧策略引用
    丢弃本轮所有采集数据
end for
```

**关键超参的典型取值**：
- $`N = 8 \sim 64`$（并行环境数，取决于任务复杂度和硬件）
- $`T = 128 \sim 2048`$（每个环境的 rollout 长度）
- $`K = 3 \sim 10`$（内层 epoch 数）
- Mini-batch size $`M = 64 \sim 4096`$
- $`\varepsilon = 0.1 \sim 0.3`$（常用 0.2）
- $`\gamma = 0.99`$，$`\lambda = 0.95`$
- $`c_1 = 0.5`$，$`c_2 = 0.01`$
- 学习率 $`\alpha = 3 \times 10^{-4}`$，通常线性衰减到 0

**PPO 相对 A2C 的本质改进**：A2C 只用采集到的数据做一次梯度更新就丢弃，样本效率低。PPO 通过内层 $`K`$ 个 epoch 复用同一批数据做多次更新，这正是 clip 机制存在的理由——没有 clip，多次复用 off-policy 数据会导致重要性采样比率 $`r_t`$ 偏离 1 越来越远，梯度估计的方差爆炸（第 5 章已分析）。Clip 通过将 $`r_t`$ 的有效范围限制在 $`[1-\varepsilon, 1+\varepsilon]`$ 内，使得即使经过多个 epoch 的更新，目标函数仍然保持合理的行为。

<a id="s6-6"></a>

## 6.6 PPO 的理论局限与已知问题

尽管 PPO 在实践中极为成功，但诚实地呈现其理论局限是必要的。以下是深度学习社区已形成共识的几个关键问题。

<a id="s6-6-1"></a>

### 6.6.1 Clip 不保证单调改进

TRPO 的性能下界 $`J(\pi') \geq L_\pi(\pi') - C \cdot \max_s D_{\text{KL}}(\pi_{\text{old}} \| \pi')`$ 依赖于对**所有状态**上 KL 散度的最大值约束。PPO-Clip 将这一全局约束替换为对单个样本重要性比率的逐点截断，这在数学上不等价于任何状态级 KL 约束。因此，PPO-Clip **没有** TRPO 那样的单调改进理论保证。在实践中，PPO 的性能曲线偶尔也会出现下降，只是频率和幅度远小于未裁剪的策略梯度。

<a id="s6-6-2"></a>

### 6.6.2 梯度截断引入偏差

当 $`r_t(\theta)`$ 被 clip 时（即 $`r_t \lt  1-\varepsilon`$ 或 $`r_t \gt  1+\varepsilon`$ 且对应情形下裁剪项被 $`\min`$ 选中），该项对 $`\theta`$ 的梯度为零。这意味着这部分样本的信息被完全丢弃了。从统计估计的角度看，这等价于对梯度估计施加了一个与数据相关的截断，引入了系统性偏差。在理论上，这个偏差的大小与 $`\varepsilon`$ 和策略更新步数 $`K`$ 有关：$`\varepsilon`$ 越小或 $`K`$ 越大，被截断的样本比例越高，偏差越大。但在实践中，适度的偏差换取了方差的显著降低，总体效果是正面的。

<a id="s6-6-3"></a>

### 6.6.3 工程细节的重要性

Andrychowicz et al. (2021) 的大规模实证研究得出了一个重要结论：**PPO 的成功很大程度来自工程细节而非 clip 机制本身**。该研究发现：
- 网络结构（特别是 tanh 激活 vs ReLU）、权重初始化方案、学习率调度策略对最终性能的影响与 clip 机制同等重要甚至更大。
- 优势归一化（每批数据减均值除标准差）和观测归一化（running mean/std normalization）对训练稳定性至关重要。
- 在某些设置下，去掉 clip 的普通策略梯度配合良好的工程实践也能达到接近 PPO 的性能。

这一发现并不否定 PPO 的实用价值——clip 仍然是最简单有效的稳定化手段之一——但它提醒我们不应将 PPO 的成功过度归因于单一机制。

<a id="s6-6-4"></a>

### 6.6.4 状态分布不匹配

这是第 5 章遗留的问题。PPO 使用的替代目标 $`\mathbb{E}_{s \sim d^{\pi_{\theta_{\text{old}}}}}[r_t(\theta) \hat{A}_t]`$ 中的期望是在旧策略的状态分布 $`d^{\pi_{\theta_{\text{old}}}}`$ 下计算的，但我们真正关心的是新策略分布 $`d^{\pi_\theta}`$ 下的性能。单步比率 $`r_t`$ 只修正了动作分布的差异，没有修正状态分布的差异。在内层 $`K`$ 个 epoch 的更新过程中，$`\theta`$ 逐渐远离 $`\theta_{\text{old}}`$，$`d^{\pi_\theta}`$ 与 $`d^{\pi_{\theta_{\text{old}}}}`$ 的差异累积增大，替代目标的偏差也随之增大。这就是为什么 $`K`$ 不能取太大的根本原因。

<a id="s6-6-5"></a>

### 6.6.5 后续改进方向简述

针对上述局限，学术界提出了多种改进：
- **PPO-RNN**：将 PPO 扩展到循环网络以处理部分可观测环境。
- **VMPO（V-MPO）**：用指数加权替代 clip，提供更平滑的梯度。
- **TRPL（Trust Region Policy Learning）**：尝试在 PPO 框架内恢复更接近 TRPO 的理论保证。
- **GRPO（Group Relative Policy Optimization）**：2024 年后在 RLHF 场景中兴起，去掉了 critic 网络，改用组内相对优势估计，显存开销减半。
- **DAPO（Decoupled Clip and Dynamic sAmpling Policy Optimization）**：解耦上下裁剪阈值（clip-higher）以抑制熵崩塌，并用动态采样过滤全对/全错的无效 prompt，是面向长链推理的开源大规模 RL 系统。

这些变体的详细推导不在本章范围内，但它们的存在本身就说明了 PPO 并非终点，而是一个仍在活跃演进的研究方向。

<a id="s6-7"></a>

## 6.7 PPO 在 RLHF 中的应用

<a id="s6-7-1"></a>

### 6.7.1 RLHF 场景的特殊性

基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）是 PPO 最重要的应用场景之一。与普通 RL 的关键差异在于：

- **奖励来源**：环境奖励 $`r(s,a)`$ 被替换为一个训练好的奖励模型 $`R_\phi(s, a)`$，该模型从人类偏好数据中学习，输出标量分数作为奖励信号。
- **策略与动作**：策略 $`\pi_\theta`$ 是语言模型（LLM），状态 $`s`$ 是提示词（prompt）加上已生成的 token 前缀，动作 $`a`$ 是下一个 token。动作空间是词表大小（通常数万到数十万）。
- **参考模型**：除了策略模型 $`\pi_\theta`$，还有一个固定的参考模型 $`\pi_{\text{ref}}`$（通常是监督微调 SFT 后的模型），用于 KL 惩罚。

<a id="s6-7-2"></a>

### 6.7.2 逐 Token 的比率与优势

PPO 在 LLM 上是**逐 token** 应用的，而非逐句或逐段。对于生成序列 $`y = (y_1, y_2, \ldots, y_T)`$，给定输入 $`x`$，每一步的比率为：

```math
r_t(\theta) = \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{\theta_{\text{old}}}(y_t | x, y_{<t})}
```

其中 $`\pi_\theta(y_t | x, y_{\lt t})`$ 是语言模型在给定前缀条件下生成 token $`y_t`$ 的条件概率。优势 $`\hat{A}_t`$ 同样逐 token 计算，通常使用 GAE 或直接使用奖励模型的输出作为终端奖励进行反向传播。

<a id="s6-7-3"></a>

### 6.7.3 KL 惩罚的双重角色

RLHF 中的 PPO 包含**两个不同层面**的 KL 相关机制，这是最常被混淆的点：

**第一个 KL：奖励塑形中的 KL 惩罚**。在计算奖励时，从奖励模型的输出中减去一个 KL 惩罚项：

```math
\tilde{r}_t = R_\phi(x, y_t, y_{<t}) - \beta_{\text{reward}} \cdot D_{\text{KL}}\big(\pi_\theta(\cdot|x,y_{<t}) \;\|\; \pi_{\text{ref}}(\cdot|x,y_{<t})\big)
```

这里的 $`\pi_{\text{ref}}`$ 是 SFT 模型，$`\beta_{\text{reward}}`$ 是奖励层面的 KL 系数。这个 KL 惩罚的作用是**防止 reward hacking**：如果没有它，策略会学会生成奖励模型打分高但实际质量差的文本（例如重复特定短语、输出退化格式），因为奖励模型只是在有限的人类偏好数据上训练的近似器，必然存在漏洞。KL 惩罚将策略锚定在 SFT 模型附近，确保生成的文本不会偏离太远。

**第二个 KL：PPO-Penalty 中的 KL 项**（如果使用 PPO-Penalty 变体）。这与 6.3 节描述的完全相同，是损失函数中约束新旧策略 $`\pi_\theta`$ 与 $`\pi_{\theta_{\text{old}}}`$ 之间距离的项。

两者的关键区别：
- 奖励塑形的 KL 是 $`\pi_\theta`$ 与 $`\pi_{\text{ref}}`$（固定的 SFT 模型）之间的距离，进入 advantage 计算，影响的是"什么行为值得鼓励"。
- PPO-Penalty 的 KL 是 $`\pi_\theta`$ 与 $`\pi_{\theta_{\text{old}}}`$（上一轮采集数据的策略）之间的距离，进入损失函数，影响的是"每次更新能走多远"。
- 即使使用 PPO-Clip（不含 PPO-Penalty 的 KL 项），奖励塑形的 KL 惩罚仍然必须存在。

<a id="s6-7-4"></a>

### 6.7.4 Reward Hacking 与 Mode Collapse

Reward hacking 的表现形式包括：生成重复性文本、输出特定格式的"作弊"回答、长度异常（过长或过短）、语义空洞但语法正确的废话。Mode collapse 则表现为策略只在少数几种"安全"模式上生成，丧失了语言的多样性。

这两个问题的根源都是奖励模型的不完美。KL 惩罚是主要的防御手段，但 $`\beta_{\text{reward}}`$ 的选择极为微妙：太大则策略被过度约束在 SFT 模型附近，无法从人类偏好中学习改进；太小则 reward hacking 失控。实践中通常需要大量的消融实验来确定合适的值。

在 RLHF 场景下，clip 的 $`\varepsilon`$ 通常取得比普通 RL 更小（0.1～0.2 而非 0.2～0.3），原因是语言模型的输出空间极其高维，策略的微小变化就可能导致生成质量的剧烈波动，需要更保守的更新步长。

<a id="s6-7-5"></a>

### 6.7.5 GRPO 对 PPO 的替代趋势

在大规模 LLM 训练中，GRPO（Group Relative Policy Optimization）正在逐渐替代 PPO。GRPO 的核心思想是：对同一个 prompt 采样一组（group）回复，用组内回复的奖励相对排名作为优势估计，从而完全省去了 critic 网络。由于 LLM 规模的 critic 与策略模型参数量相当，省掉 critic 意味着显存和算力开销近乎减半。GRPO 在 DeepSeek-R1 等模型训练中展现了与 PPO 相当甚至更好的效果，但其理论基础和长期稳定性仍需更多研究验证。

<a id="s6-8"></a>

## 6.8 与其他方法的定位关系

为了帮助读者建立全局视角，简要梳理 PPO 在策略优化方法谱系中的位置：

- **REINFORCE / Vanilla PG**（第 3 章）：最基础的策略梯度，on-policy，每条数据只用一次，高方差。
- **A2C / A3C**（第 3 章）：引入 critic 降低方差，但仍是一次性使用数据，样本效率低。
- **TRPO**（第 5 章）：通过 KL 约束保证单调改进，理论优美但实现复杂，需要二阶优化。
- **PPO**（本章）：用 clip 或自适应 KL 惩罚近似信任域，一阶优化，实现简单，效果好，是当前 on-policy RL 的事实标准。
- **SAC / TD3**（off-policy 方法，不在本文档范围）：适用于连续控制，样本效率更高但对超参更敏感。
- **GRPO / DAPO**（6.6.5 节提及）：PPO 在 RLHF 场景下的最新演进，去掉 critic 或改进优势估计。

PPO 的定位可以概括为：**在理论保证与工程简洁性之间取得了最佳平衡点的 on-policy 策略优化算法**。它不是理论上最优的，也不是最简单的，但在绝大多数实际任务中，它是第一个应该尝试的方法。

<a id="s6-9"></a>

## 6.9 本章小结与自测

<a id="h216"></a>

### 要点回顾

1. **PPO 的核心动机**是用一阶优化近似实现信任域约束，解决策略梯度样本效率低和步长敏感的问题。
2. **PPO-Clip 的目标函数**通过 $`\min`$ 操作在未裁剪项和裁剪项之间取较小值，构造出替代目标的悲观下界。六种情形分析表明：clip 只阻止朝有利方向的过度更新，不阻止朝不利方向的纠错。
3. **PPO-Penalty** 保留了完整替代目标并用自适应 $`\beta`$ 调节 KL 惩罚强度，理论上更接近 TRPO 但实践中不如 Clip 流行。
4. **完整损失函数**包含策略项（最大化）、价值函数项（最小化）和熵正则项（最大化），符号约定需特别注意。价值函数裁剪在实践中有争议，多数实现不使用。
5. **PPO 的理论局限**包括：clip 不保证单调改进、梯度截断引入偏差、成功很大程度依赖工程细节、状态分布不匹配随 epoch 数累积。
6. **RLHF 中的 PPO** 有两个不同的 KL 机制：奖励塑形中的 KL 惩罚（防 reward hacking）和 PPO-Penalty 中的 KL 项（约束更新步长），不可混淆。GRPO 正逐步替代 PPO 用于大规模 LLM 训练。

<a id="h217"></a>

### 自测题

**题目 1**：当 $`\hat{A}_t \gt  0`$ 且 $`r_t(\theta) \gt  1+\varepsilon`$ 时，$`L^{\text{CLIP}}`$ 中哪一项被 $`\min`$ 选中？该项对 $`\theta`$ 的梯度是多少？物理含义是什么？

<details>
<summary>查看参考答案</summary>

参考答案：裁剪项 $`\text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\hat{A}_t = (1+\varepsilon)\hat{A}_t`$ 被选中（因为它小于 $`r_t\hat{A}_t`$）。梯度为零，因为 $`1+\varepsilon`$ 是常数。物理含义是策略已经大幅增加了好动作的概率，clip 阻止继续增加以防止过度更新。

</details>

**题目 2**：当 $`\hat{A}_t \lt  0`$ 且 $`r_t(\theta) \lt  1-\varepsilon`$ 时，$`\min`$ 选中的是哪一项？为什么这种情况下不阻止更新？

<details>
<summary>查看参考答案</summary>

参考答案：裁剪项 $`(1-\varepsilon)\hat{A}_t`$ 被选中（因为 $`A\lt 0`$ 时 $`(1-\varepsilon)A \lt  rA`$）。梯度为零。等等——重新审视：当 $`A\lt 0`$ 且 $`r\lt 1-\varepsilon`$ 时，$`rA \gt  (1-\varepsilon)A`$（两个负数，$`r`$ 更小使 $`rA`$ 负得少即更大），所以 $`\min`$ 取 $`(1-\varepsilon)A`$，梯度确实为零。这实际上是阻止了"概率已经降得太低还要继续降"的过度更新。题目问"为什么不阻止"的前提有误——PPO 在这种情况下确实阻止了更新。正确理解是：PPO 阻止的是朝任一方向的过度更新，而非仅阻止有利方向。

</details>

**题目 3**：证明在 $`\theta = \theta_{\text{old}}`$ 处，$`L^{\text{CLIP}}`$ 的梯度等于标准策略梯度 $`\mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot \hat{A}]`$。

<details>
<summary>查看参考答案</summary>

参考答案：$`\theta = \theta_{\text{old}}`$ 时 $`r_t = 1 \in [1-\varepsilon, 1+\varepsilon]`$，clip 不生效，两项相等均为 $`\hat{A}_t`$。$`\min`$ 取任意一项均可，$`L^{\text{CLIP}} = \mathbb{E}[\hat{A}_t]`$。对其求梯度得 $`\mathbb{E}[\nabla_\theta r_t \cdot \hat{A}_t] = \mathbb{E}[\nabla_\theta \log \pi_\theta \cdot \hat{A}_t]`$，即标准策略梯度。

</details>

**题目 4**：RLHF 中奖励塑形的 KL 惩罚与 PPO-Penalty 中的 KL 项有何本质区别？如果只用 PPO-Clip 做 RLHF，是否可以省略所有 KL 相关项？

<details>
<summary>查看参考答案</summary>

参考答案：奖励塑形的 KL 是 $`\pi_\theta`$ 与固定参考模型 $`\pi_{\text{ref}}`$ 的距离，进入奖励计算防止 reward hacking；PPO-Penalty 的 KL 是 $`\pi_\theta`$ 与 $`\pi_{\theta_{\text{old}}}`$ 的距离，进入损失函数约束更新步长。即使用 PPO-Clip，奖励塑形的 KL 惩罚也不可省略，否则策略会钻奖励模型漏洞。PPO-Penalty 的 KL 项可以用 clip 替代。

</details>

**题目 5**：为什么 PPO 内层可以做多个 epoch 的数据复用，而普通策略梯度不行？clip 在其中扮演什么角色？

<details>
<summary>查看参考答案</summary>

参考答案：普通策略梯度是 on-policy 的，数据必须来自当前策略，复用旧数据会导致重要性采样比率 $`r_t`$ 偏离 1，梯度估计方差爆炸。PPO 通过 clip 将 $`r_t`$ 的有效影响限制在 $`[1-\varepsilon, 1+\varepsilon]`$ 内，即使 $`r_t`$ 实际值远离 1，目标函数的行为和梯度都被控制在合理范围内，使得多 epoch 复用数据成为可能。

</details>

**题目 6**：Andrychowicz et al. (2021) 的发现对"PPO 之所以有效是因为 clip 机制"这一观点构成了怎样的挑战？我们应该如何重新理解 PPO 的成功？

<details>
<summary>查看参考答案</summary>

参考答案：该研究发现工程细节（网络结构、初始化、归一化、学习率调度）对性能的影响与 clip 同等重要，且在某些设置下去掉 clip 也能工作。这说明 PPO 的成功是 clip 机制与一系列工程实践共同作用的结果，不应过度归因于单一组件。clip 提供了必要的稳定化，但良好的工程实践同样是不可或缺的。理解 PPO 应将其视为一个系统性的方法论，而非仅仅是一个目标函数。

</details>

---

---

[上一章](chapter-05.md) · [返回目录](../README.md) · [阅读路线与符号表](README.md) · [下一章](chapter-07.md)
