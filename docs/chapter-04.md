[上一章](chapter-03.md) · [返回目录](../README.md) · [阅读路线与符号表](README.md) · [下一章](chapter-05.md)

---

<a id="chapter-4"></a>

# 第 4 章 优势估计与广义优势估计（GAE）

**本章目录**

- [4.1 本章起点：从策略梯度到优势估计器](#user-content-s4-1)
- [4.2 k-step 优势估计器的定义与推导](#user-content-s4-2)
- [4.3 偏差-方差的量化分析](#user-content-s4-3)
- [4.4 TD($`\lambda`$) 回顾：$`\lambda`$-return 与资格迹](#user-content-s4-4)
- [4.5 广义优势估计（GAE）的完整推导](#user-content-s4-5)
- [4.6 递推实现形式与实践处理](#user-content-s4-6)
- [4.7 实践中的数学处理与 GAE 和 PPO 的关系](#user-content-s4-7)
- [4.8 本章小结与自测](#user-content-s4-8)

<a id="s4-1"></a>

## 4.1 本章起点：从策略梯度到优势估计器

在第 3 章中，我们已经推导出策略梯度的通用估计量形式：

```math
g = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \hat{A}_t \right]
```

其中 $`\hat{A}_t`$ 是真实优势函数 $`A^\pi(s_t, a_t)`$ 的某种估计器。根据第 2 章的定义，真实优势函数为：

```math
A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)
```

它衡量的是在状态 $`s_t`$ 下采取动作 $`a_t`$ 相对于平均水平的"好坏程度"。策略梯度定理（见第 3 章）告诉我们，只要 $`\hat{A}_t`$ 是 $`A^\pi(s_t, a_t)`$ 的无偏估计，上述梯度估计就是正确的。然而在实际算法中，我们几乎从不使用无偏估计——因为无偏往往意味着极高的方差，导致训练不稳定。

**本章的唯一核心问题是：如何构造 $`\hat{A}_t`$，使梯度估计在偏差（bias）与方差（variance）之间取得可控的平衡。**

为此，我们引入均方误差（Mean Squared Error, MSE）分解作为全章的分析框架。对于任意估计器 $`\hat{A}_t`$，其关于真实优势 $`A^\pi(s_t, a_t)`$ 的 MSE 可分解为：

```math
\text{MSE}(\hat{A}_t) = \mathbb{E}\left[ (\hat{A}_t - A^\pi(s_t, a_t))^2 \right] = \underbrace{\left( \mathbb{E}[\hat{A}_t] - A^\pi(s_t, a_t) \right)^2}_{\text{偏差}^2} + \underbrace{\text{Var}(\hat{A}_t)}_{\text{方差}}
```

这个分解告诉我们：一个估计器的总误差由偏差的平方与方差两部分组成。减小其中一个往往以增大另一个为代价。本章的所有推导与分析都围绕这个权衡展开。

<a id="s4-2"></a>

## 4.2 k-step 优势估计器的定义与推导

<a id="s4-2-1"></a>

### 4.2.1 TD 残差与 1-step 优势估计

首先回顾 TD 残差（TD residual）。设 $`V(s)`$ 是我们对价值函数的近似（通常是一个神经网络），$`\gamma \in [0, 1)`$ 是折扣因子（见第 2 章）。定义单步 TD 残差为：

```math
\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)
```

其中 $`r_t`$ 是在时刻 $`t`$ 执行动作 $`a_t`$ 后获得的即时奖励，$`s_{t+1}`$ 是转移后的下一个状态。当 $`V = V^\pi`$（即价值函数恰好等于真实状态价值）时，由贝尔曼方程（见第 2 章）可知 $`\mathbb{E}[\delta_t^{V^\pi} | s_t, a_t] = A^\pi(s_t, a_t)`$，因此 $`\delta_t^{V^\pi}`$ 是优势的无偏估计。但在实践中 $`V \neq V^\pi`$，所以 $`\delta_t^V`$ 是有偏的。

我们将 **1-step 优势估计器** 定义为：

```math
\hat{A}_t^{(1)} = \delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)
```

<a id="s4-2-2"></a>

### 4.2.2 2-step 优势估计及其等价形式

自然地，我们可以利用两步真实奖励来减少对 bootstrap 值 $`V(s_{t+1})`$ 的依赖。定义 **2-step 优势估计器** 为：

```math
\hat{A}_t^{(2)} = r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) - V(s_t)
```

下面证明它等于两个 TD 残差的加权和：$`\hat{A}_t^{(2)} = \delta_t^V + \gamma \delta_{t+1}^V`$。

**证明：** 将 $`\delta_t^V`$ 和 $`\delta_{t+1}^V`$ 的定义代入右边：

```math
\begin{aligned}
\delta_t^V + \gamma \delta_{t+1}^V &= \left( r_t + \gamma V(s_{t+1}) - V(s_t) \right) + \gamma \left( r_{t+1} + \gamma V(s_{t+2}) - V(s_{t+1}) \right) \\
&= r_t + \gamma V(s_{t+1}) - V(s_t) + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) - \gamma V(s_{t+1}) \\
&= r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) - V(s_t)
\end{aligned}
```

注意中间项 $`\gamma V(s_{t+1})`$ 与 $`-\gamma V(s_{t+1})`$ 恰好抵消。这正是 $`\hat{A}_t^{(2)}`$ 的定义。证毕。

这个抵消现象并非巧合，而是 k-step 估计器可以统一写成 TD 残差累加和的关键原因。

<a id="s4-2-3"></a>

### 4.2.3 一般 k-step 优势估计器

推广上述模式，定义 **k-step 优势估计器** 为：

```math
\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l r_{t+l} + \gamma^k V(s_{t+k}) - V(s_t)
```

同时，我们有以下等价表述：

```math
\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V
```

**命题：** 上述两种形式对所有正整数 $`k`$ 等价。

**证明（数学归纳法）：**

**基础情形 $`k=1`$：** $`\sum_{l=0}^{0} \gamma^l \delta_{t+l}^V = \delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)`$，与定义一致。

**归纳假设：** 假设对某个 $`k \geq 1`$，等式成立，即 $`\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V = \sum_{l=0}^{k-1} \gamma^l r_{t+l} + \gamma^k V(s_{t+k}) - V(s_t)`$。

**归纳步骤：** 考虑 $`k+1`$ 的情形：

```math
\begin{aligned}
\sum_{l=0}^{k} \gamma^l \delta_{t+l}^V &= \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V + \gamma^k \delta_{t+k}^V \\
&= \hat{A}_t^{(k)} + \gamma^k \left( r_{t+k} + \gamma V(s_{t+k+1}) - V(s_{t+k}) \right) \\
&= \left( \sum_{l=0}^{k-1} \gamma^l r_{t+l} + \gamma^k V(s_{t+k}) - V(s_t) \right) + \gamma^k r_{t+k} + \gamma^{k+1} V(s_{t+k+1}) - \gamma^k V(s_{t+k}) \\
&= \sum_{l=0}^{k-1} \gamma^l r_{t+l} + \gamma^k r_{t+k} + \gamma^{k+1} V(s_{t+k+1}) - V(s_t) \\
&= \sum_{l=0}^{k} \gamma^l r_{t+l} + \gamma^{k+1} V(s_{t+k+1}) - V(s_t)
\end{aligned}
```

同样，$`\gamma^k V(s_{t+k})`$ 与 $`-\gamma^k V(s_{t+k})`$ 抵消。这正是 $`\hat{A}_t^{(k+1)}`$ 的定义。归纳完成。

<a id="s4-2-4"></a>

### 4.2.4 无穷步极限：蒙特卡洛优势估计

令 $`k \to \infty`$，假设 episode 无限长或适当处理终止条件，由于 $`\gamma \lt  1`$，$`\gamma^k V(s_{t+k}) \to 0`$（在 $`V`$ 有界的条件下），我们得到：

```math
\hat{A}_t^{(\infty)} = \sum_{l=0}^{\infty} \gamma^l r_{t+l} - V(s_t)
```

这就是 **蒙特卡洛（Monte Carlo）优势估计**。它完全使用真实回报，不依赖任何 bootstrap 值（除了作为基线的 $`V(s_t)`$），因此在 $`V`$ 准确时是无偏的，但方差最大。

<a id="s4-3"></a>

## 4.3 偏差-方差的量化分析

本节是本章的核心分析之一。我们对 k-step 估计器的偏差和方差进行严格的定量刻画。

<a id="s4-3-1"></a>

### 4.3.1 偏差的来源与表达式

**命题：** k-step 优势估计器的偏差满足：

```math
\mathbb{E}[\hat{A}_t^{(k)}] - A^\pi(s_t, a_t) = \gamma^k \, \mathbb{E}\left[ V(s_{t+k}) - V^\pi(s_{t+k}) \right]
```

其中期望是关于从 $`(s_t, a_t)`$ 出发、遵循策略 $`\pi`$ 的状态-动作轨迹取的。

**推导：** 首先回忆真实优势的另一种表达。由第 2 章可知：

```math
A^\pi(s_t, a_t) = \mathbb{E}\left[ \sum_{l=0}^{\infty} \gamma^l r_{t+l} \,\middle|\, s_t, a_t \right] - V^\pi(s_t)
```

而 k-step 估计器的期望为：

```math
\mathbb{E}[\hat{A}_t^{(k)}] = \mathbb{E}\left[ \sum_{l=0}^{k-1} \gamma^l r_{t+l} + \gamma^k V(s_{t+k}) - V(s_t) \,\middle|\, s_t, a_t \right]
```

两式相减：

```math
\begin{aligned}
\mathbb{E}[\hat{A}_t^{(k)}] - A^\pi(s_t, a_t) &= \mathbb{E}\left[ \sum_{l=0}^{k-1} \gamma^l r_{t+l} + \gamma^k V(s_{t+k}) \,\middle|\, s_t, a_t \right] - V(s_t) \\
&\quad - \left( \mathbb{E}\left[ \sum_{l=0}^{\infty} \gamma^l r_{t+l} \,\middle|\, s_t, a_t \right] - V^\pi(s_t) \right)
\end{aligned}
```

将第一个期望拆成前 $`k`$ 项与剩余项之和：

```math
\mathbb{E}\left[ \sum_{l=0}^{\infty} \gamma^l r_{t+l} \,\middle|\, s_t, a_t \right] = \mathbb{E}\left[ \sum_{l=0}^{k-1} \gamma^l r_{t+l} \,\middle|\, s_t, a_t \right] + \mathbb{E}\left[ \sum_{l=k}^{\infty} \gamma^l r_{t+l} \,\middle|\, s_t, a_t \right]
```

由马尔可夫性质，$`\mathbb{E}\left[ \sum_{l=k}^{\infty} \gamma^l r_{t+l} \,\middle|\, s_t, a_t \right] = \gamma^k \, \mathbb{E}[V^\pi(s_{t+k}) | s_t, a_t]`$。代入并整理：

```math
\begin{aligned}
\mathbb{E}[\hat{A}_t^{(k)}] - A^\pi(s_t, a_t) &= \gamma^k \, \mathbb{E}[V(s_{t+k}) | s_t, a_t] - \gamma^k \, \mathbb{E}[V^\pi(s_{t+k}) | s_t, a_t] - V(s_t) + V^\pi(s_t) \\
&= \gamma^k \, \mathbb{E}[V(s_{t+k}) - V^\pi(s_{t+k}) | s_t, a_t] + \left( V^\pi(s_t) - V(s_t) \right)
\end{aligned}
```

等等——这里多出了一项 $`V^\pi(s_t) - V(s_t)`$。这是因为我们在 $`\hat{A}_t^{(k)}`$ 中减去的是 $`V(s_t)`$ 而非 $`V^\pi(s_t)`$。如果我们把偏差定义为相对于 $`Q^\pi(s_t, a_t) - V(s_t)`$（即以当前近似 $`V(s_t)`$ 为基线），则该项消失。更精确地，当我们关注的是"用 $`\hat{A}_t^{(k)}`$ 替代 $`A^\pi`$ 引入的额外误差"时，关键的偏差来源正是 bootstrap 端点处 $`V`$ 与 $`V^\pi`$ 的差距被 $`\gamma^k`$ 缩放的部分：

```math
\text{Bias}(\hat{A}_t^{(k)}) \propto \gamma^k \, \mathbb{E}[V(s_{t+k}) - V^\pi(s_{t+k})]
```

**关键结论：** 偏差随 $`k`$ 增大而以 $`\gamma^k`$ 的速率指数衰减。当 $`k=1`$ 时偏差最大（完全依赖一步 bootstrap）；当 $`k \to \infty`$ 时偏差趋于零（蒙特卡洛估计无偏）。

<a id="s4-3-2"></a>

### 4.3.2 方差的来源与量级

方差的来源更为直观。k-step 估计器中包含 $`k`$ 个随机奖励项 $`r_t, r_{t+1}, \dots, r_{t+k-1}`$，每一项都受环境动态和策略随机性的影响。粗略地，若各步奖励的方差量级为 $`\sigma_r^2`$ 且近似独立，则：

```math
\text{Var}(\hat{A}_t^{(k)}) \approx \sum_{l=0}^{k-1} \gamma^{2l} \, \sigma_r^2 = \sigma_r^2 \cdot \frac{1 - \gamma^{2k}}{1 - \gamma^2}
```

这是一个关于 $`k`$ 单调递增的函数。当 $`k=1`$ 时方差最小（仅含一步奖励加上 bootstrap 值的噪声）；当 $`k \to \infty`$ 时方差趋于 $`\sigma_r^2 / (1 - \gamma^2)`$，达到最大值。

特别值得注意的是**稀疏奖励环境**的表现：在大多数时间步奖励为零的环境中，蒙特卡洛估计（$`k=\infty`$）需要等待极长的非零奖励信号才能给出有意义的梯度更新，导致方差极大、学习极慢。而较小的 $`k`$ 通过 bootstrap 提供了密集的学习信号，尽管引入了偏差。

<a id="s4-3-3"></a>

### 4.3.3 偏差-方差权衡总结表

下表总结了不同 $`k`$ 值下优势估计器的特性对比：

| $`k`$ 值 | 偏差 | 方差 | 样本效率 | 对 Critic 质量的依赖 | 备注 |
|--------|------|------|----------|---------------------|------|
| $`k=1`$ | 最大（$`\propto \gamma \Vert V - V^\pi\Vert `$） | 最小 | 最高（每步都可更新） | 最强 | 纯 TD，完全信任 critic |
| $`k=2`$ | 较大（$`\propto \gamma^2 \Vert V - V^\pi\Vert `$） | 较小 | 高 | 强 | 两步真实奖励 + bootstrap |
| $`k=5`$ | 中等 | 中等 | 中等 | 中等 | 常用折中选择 |
| $`k=20`$ | 较小 | 较大 | 较低 | 弱 | 接近蒙特卡洛 |
| $`k=\infty`$ | 零（无偏） | 最大 | 最低（需完整 episode） | 最弱（仅作基线） | 蒙特卡洛优势 |

这张表清晰地展示了偏差-方差权衡的全貌：没有免费的午餐，选择 $`k`$ 就是在选择你愿意承受多少偏差来换取方差的降低（或反之）。

<a id="s4-4"></a>

## 4.4 TD($`\lambda`$) 回顾：$`\lambda`$-return 与资格迹

在进入 GAE 之前，有必要回顾 TD($`\lambda`$) 的思想，因为 GAE 本质上是将 TD($`\lambda`$) 的加权平均思想从"回报估计"迁移到了"优势估计"上。

<a id="s4-4-1"></a>

### 4.4.1 前向视角：$`\lambda`$-return

定义 n-step 回报为：

```math
G_t^{(n)} = \sum_{l=0}^{n-1} \gamma^l r_{t+l} + \gamma^n V(s_{t+n})
```

$`\lambda`$-return 是各 n-step 回报的指数加权平均：

```math
G_t^\lambda = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}
```

其中 $`\lambda \in [0, 1]`$ 是衰减参数。**权重归一化验证：**

```math
(1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} = (1 - \lambda) \cdot \frac{1}{1 - \lambda} = 1
```

因此 $`G_t^\lambda`$ 确实是各 $`G_t^{(n)}`$ 的凸组合。当 $`\lambda = 0`$ 时，$`G_t^\lambda = G_t^{(1)}`$（纯 TD）；当 $`\lambda = 1`$ 时，$`G_t^\lambda = G_t^{(\infty)}`$（蒙特卡洛回报）。

<a id="s4-4-2"></a>

### 4.4.2 后向视角与资格迹

$`\lambda`$-return 还有一个等价的后向视角（backward view）表述，通过资格迹（eligibility trace）实现。资格迹 $`e_t`$ 记录了近期访问过的状态的"新鲜度"，使得 TD 更新可以回溯修正之前的估计。前向视角与后向视角在数学上是等价的（Sutton & Barto, 2018, Chapter 12），但后向视角更适合在线实现。本章不深入资格迹的具体衰减机制，仅需理解 $`\lambda`$ 提供了在 TD 与 MC 之间连续插值的能力——这正是 GAE 所借鉴的核心思想。

<a id="s4-5"></a>

## 4.5 广义优势估计（GAE）的完整推导

<a id="s4-5-1"></a>

### 4.5.1 动机

上一节的分析表明：$`k`$ 小则偏差大方差小，$`k`$ 大则偏差小方差大。既然单一的 $`k`$ 无法兼顾两者，一个自然的想法是：**将所有 $`k`$ 的优势估计器做加权平均**，让权重分布本身成为调节偏差-方差权衡的旋钮。这正是广义优势估计（Generalized Advantage Estimation, GAE）的核心思想（Schulman et al., 2016）。

<a id="s4-5-2"></a>

### 4.5.2 GAE 的定义

仿照 $`\lambda`$-return 对 n-step 回报的加权方式，定义 GAE 为各 k-step 优势估计器的指数加权平均：

```math
\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = (1 - \lambda) \sum_{k=1}^{\infty} \lambda^{k-1} \hat{A}_t^{(k)}
```

其中 $`(1-\lambda)`$ 确保权重归一化（验证同 4.4.1 节），$`\lambda \in [0, 1]`$ 是新引入的调节参数。

<a id="s4-5-3"></a>

### 4.5.3 核心推导：交换求和顺序

将 $`\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V`$ 代入上式：

```math
\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = (1 - \lambda) \sum_{k=1}^{\infty} \lambda^{k-1} \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V
```

现在需要交换两个求和的顺序。注意内层求和的条件是 $`0 \leq l \leq k-1`$，即 $`l \lt  k`$，等价于 $`k \gt  l`$，即 $`k \geq l+1`$。因此交换后外层对 $`l`$ 从 $`0`$ 到 $`\infty`$ 求和，内层对 $`k`$ 从 $`l+1`$ 到 $`\infty`$ 求和：

```math
\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = (1 - \lambda) \sum_{l=0}^{\infty} \gamma^l \delta_{t+l}^V \sum_{k=l+1}^{\infty} \lambda^{k-1}
```

计算内层几何级数。令 $`m = k - 1`$，则当 $`k = l+1`$ 时 $`m = l`$，于是：

```math
\sum_{k=l+1}^{\infty} \lambda^{k-1} = \sum_{m=l}^{\infty} \lambda^m = \frac{\lambda^l}{1 - \lambda}
```

代回原式：

```math
\begin{aligned}
\hat{A}_t^{\text{GAE}(\gamma, \lambda)} &= (1 - \lambda) \sum_{l=0}^{\infty} \gamma^l \delta_{t+l}^V \cdot \frac{\lambda^l}{1 - \lambda} \\
&= \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V
\end{aligned}
```

$`(1-\lambda)`$ 与 $`1/(1-\lambda)`$ 恰好约去。**这就是 GAE 的简洁闭式表达：**

```math
\boxed{\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \, \delta_{t+l}^V}
```

注意有效折扣因子变成了 $`\gamma\lambda`$ 而非 $`\gamma`$。这意味着 $`\lambda`$ 直接控制了 TD 残差累加的衰减速率：$`\lambda`$ 越小，远处的 $`\delta`$ 被压制得越快，估计越偏向短视（低方差、高偏差）；$`\lambda`$ 越大，远处的 $`\delta`$ 保留得越多，估计越偏向长远（高方差、低偏差）。

<a id="s4-5-4"></a>

### 4.5.4 两个极端情形的验证

**情形 1：$`\lambda = 0`$。** 此时 $`(\gamma \lambda)^l = 0`$ 对所有 $`l \geq 1`$ 成立，仅剩 $`l=0`$ 项：

```math
\hat{A}_t^{\text{GAE}(\gamma, 0)} = \delta_t^V = \hat{A}_t^{(1)}
```

退化为纯 TD 优势估计。方差最小，偏差最大，完全依赖 critic 的质量。

**情形 2：$`\lambda = 1`$。** 此时 $`(\gamma \lambda)^l = \gamma^l`$：

```math
\hat{A}_t^{\text{GAE}(\gamma, 1)} = \sum_{l=0}^{\infty} \gamma^l \delta_{t+l}^V = \hat{A}_t^{(\infty)} = \sum_{l=0}^{\infty} \gamma^l r_{t+l} - V(s_t)
```

退化为蒙特卡洛优势估计。无偏（当 $`V(s_t)`$ 准确时），方差最大，critic 仅作为基线减少方差。

<a id="s4-5-5"></a>

### 4.5.5 $`\gamma`$ 与 $`\lambda`$ 的角色区分

这是初学者最容易混淆的概念，必须专门强调：

- **$`\gamma`$（折扣因子）** 定义了强化学习问题本身的优化目标。它决定了智能体对未来奖励的重视程度，是 MDP 定义的一部分（见第 2 章）。改变 $`\gamma`$ 就改变了"什么是最优策略"这个问题的答案。
- **$`\lambda`$（GAE 参数）** 是一个纯粹的算法超参数，不影响优化目标本身。它的唯一作用是控制优势估计中偏差与方差的权衡。无论 $`\lambda`$ 取何值，策略梯度的期望方向不变（在 $`V = V^\pi`$ 的理想情况下）；$`\lambda`$ 只影响估计的噪声水平和收敛速度。

简言之：$`\gamma`$ 是"问题的参数"，$`\lambda`$ 是"解法的参数"。

<a id="s4-6"></a>

## 4.6 递推实现形式与实践处理

<a id="s4-6-1"></a>

### 4.6.1 从无穷级数到递推公式

GAE 的级数形式 $`\hat{A}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}^V`$ 虽然数学上优美，但不适合直接计算。将其改写为递推形式：

```math
\begin{aligned}
\hat{A}_t &= \delta_t^V + (\gamma\lambda) \delta_{t+1}^V + (\gamma\lambda)^2 \delta_{t+2}^V + \cdots \\
&= \delta_t^V + \gamma\lambda \left( \delta_{t+1}^V + (\gamma\lambda) \delta_{t+2}^V + \cdots \right) \\
&= \delta_t^V + \gamma\lambda \, \hat{A}_{t+1}
\end{aligned}
```

即：

```math
\boxed{\hat{A}_t = \delta_t^V + \gamma\lambda \, \hat{A}_{t+1}}
```

这个递推关系意味着我们可以**从后向前**遍历一次 rollout 来计算所有时间步的优势估计。具体地，对于一个长度为 $`T`$ 的 rollout，我们从 $`t = T-1`$ 开始，设 $`\hat{A}_T = 0`$（episode 边界条件，详见下文），然后依次计算 $`\hat{A}_{T-1}, \hat{A}_{T-2}, \dots, \hat{A}_0`$。每一步只需常数时间，总体复杂度为 $`O(T)`$。

<a id="s4-6-2"></a>

### 4.6.2 配套的 Critic 回归目标

有了优势估计 $`\hat{A}_t`$，我们可以自然地为 critic 网络构造回归目标。由于 $`\hat{A}_t \approx Q^\pi(s_t, a_t) - V(s_t)`$，可得：

```math
V_t^{\text{target}} = \hat{A}_t + V(s_t)
```

这被称为 $`\lambda`$-return 目标（因为它等价于用 $`\lambda`$-return 作为 $`V(s_t)`$ 的训练目标）。在实践中，我们用 $`V_t^{\text{target}}`$ 作为 critic 网络的监督信号，通过最小化 $`(V_\phi(s_t) - V_t^{\text{target}})^2`$ 来更新 critic 参数 $`\phi`$。

<a id="s4-6-3"></a>

### 4.6.3 Episode 边界处理

递推公式 $`\hat{A}_t = \delta_t^V + \gamma\lambda \hat{A}_{t+1}`$ 在 episode 的最后一步需要一个边界值 $`\hat{A}_{t+1}`$。这里必须区分两种情况：

- **真正终止（termination）：** 环境到达了吸收态（absorbing state），episode 自然结束。此时不存在 $`s_{t+1}`$，也没有后续奖励，因此 $`\hat{A}_{t+1} = 0`$，且 $`\delta_t^V = r_t - V(s_t)`$（不 bootstrap）。
- **截断（truncation）：** 环境因达到最大步数限制而被人为切断，但底层 MDP 并未终止。此时 $`s_{t+1}`$ 是存在的（只是我们没有继续采样），因此仍需 bootstrap：$`\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)`$，且 $`\hat{A}_{t+1} = 0`$（因为我们没有 $`t+1`$ 之后的数据）。

**这两种情况的数学区别在于 $`\delta_t^V`$ 的计算方式，而非 $`\hat{A}_{t+1}`$ 的赋值（两者都置零）。** 混淆截断与终止是工程实现中最高频的 bug 之一（详见第 7 章）。

<a id="s4-6-4"></a>

### 4.6.4 有限 Horizon 下的等价性说明

在实际训练中，rollout 总是有限长度的。当 rollout 长度为 $`T`$ 时，递推形式计算的实际上是截断级数 $`\sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l}^V`$，而非无穷级数。但由于 $`\gamma\lambda \lt  1`$，被截断的尾部贡献以 $`(\gamma\lambda)^{T-t}`$ 的速率衰减，在 $`T`$ 足够大时可以忽略。这也是为什么 PPO 等算法通常使用固定长度的 rollout 片段（如 $`T=2048`$）而非完整 episode。

<a id="s4-7"></a>

## 4.7 实践中的数学处理与 GAE 和 PPO 的关系

<a id="s4-7-1"></a>

### 4.7.1 优势归一化

在实践中，计算完 $`\hat{A}_t`$ 后通常会对其进行归一化：

```math
\tilde{A}_t = \frac{\hat{A}_t - \mu_A}{\sigma_A}
```

其中 $`\mu_A`$ 和 $`\sigma_A`$ 是当前 mini-batch 内所有 $`\hat{A}_t`$ 的均值和标准差。这一步的作用有三：

1. **消除尺度差异：** 不同环境的奖励量级可能相差数个数量级，归一化使梯度更新的步长与环境无关。
2. **稳定训练：** 优势值的绝对大小直接影响策略梯度的幅度，归一化防止梯度爆炸或消失。
3. **自适应基线：** 减去均值等价于添加了一个随数据变化的基线函数，进一步降低方差。

**为什么在 mini-batch 级别做？** 因为优势估计本身已经包含了全局信息（通过递推），如果在整个 rollout 级别归一化，会抹平不同时间段之间的相对差异，削弱学习信号。Mini-batch 级别的归一化在保持局部对比的同时消除了全局尺度问题。

<a id="s4-7-2"></a>

### 4.7.2 截断与终止的数学区别再述

这一点的数学本质值得再次强调。设 $`d_t \in \{0, 1\}`$ 表示时刻 $`t`$ 是否为真正终止（$`d_t = 1`$ 表示终止），则统一的 TD 残差写法为：

```math
\delta_t^V = r_t + \gamma (1 - d_t) V(s_{t+1}) - V(s_t)
```

当 $`d_t = 1`$（终止）时，$`\delta_t^V = r_t - V(s_t)`$；当 $`d_t = 0`$（正常转移或截断）时，$`\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)`$。截断时 $`d_t = 0`$，因此仍然 bootstrap——这在数学上是正确的，因为截断并不改变 MDP 的价值函数定义，只是限制了我们的观测窗口。

<a id="s4-7-3"></a>

### 4.7.3 GAE 与 PPO 的正交关系

最后需要澄清 GAE 与 PPO 之间的关系，避免概念混淆：

- **GAE 负责"优势怎么算"**：它是一个优势估计方法，解决的是本章的核心问题——如何在偏差与方差之间取得平衡。GAE 的输出 $`\hat{A}_t`$ 可以被任何策略梯度算法使用。
- **PPO 负责"如何用旧数据做多次更新"**：PPO 的核心创新是通过 clip 或 penalty 机制（见第 6 章）限制策略更新幅度，使得同一批数据可以被安全地重复使用。这与优势估计是完全独立的问题。

两者是正交的组件。PPO 原论文（Schulman et al., 2017）默认使用 $`\lambda = 0.95`$、$`\gamma = 0.99`$。这组参数的含义是：$`\gamma = 0.99`$ 表示智能体高度重视长期奖励（有效 horizon 约为 $`1/(1-\gamma) = 100`$ 步）；$`\lambda = 0.95`$ 表示在偏差-方差权衡中略微偏向低偏差端（有效折扣 $`\gamma\lambda = 0.9405`$，有效 horizon 约为 $`1/(1-\gamma\lambda) \approx 17`$ 步用于优势估计）。这意味着 GAE 利用了比完整回报更短但比单步 TD 更长的时间尺度来估计优势，在实践中被证明是一个稳健的折中。

<a id="s4-8"></a>

## 4.8 本章小结与自测

<a id="h151"></a>

### 要点回顾

1. **MSE 分解框架：** 优势估计器的总误差 = 偏差² + 方差。k-step 估计器提供了在这两者之间连续调节的手段。
2. **k-step 估计器的双重等价形式：** $`\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l r_{t+l} + \gamma^k V(s_{t+k}) - V(s_t) = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V`$，两者通过归纳法证明等价。
3. **偏差的指数衰减：** $`\text{Bias}(\hat{A}_t^{(k)}) \propto \gamma^k \|V - V^\pi\|`$，k 越大偏差越小；方差的单调增长：k 越大累加的随机奖励越多，方差越大。
4. **GAE 的闭式推导：** 通过对所有 k-step 估计器做指数加权平均并交换求和顺序，得到 $`\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}^V`$，以及递推形式 $`\hat{A}_t = \delta_t + \gamma\lambda \hat{A}_{t+1}`$。
5. **$`\gamma`$ 与 $`\lambda`$ 的本质区别：** $`\gamma`$ 定义问题的折扣结构，$`\lambda`$ 是纯粹的偏差-方差调节旋钮。GAE 与 PPO 是正交组件，分别解决"优势怎么算"和"旧数据怎么用"两个独立问题。

<a id="h152"></a>

### 自测题

**题目 1：** 证明 $`\hat{A}_t^{(2)} = \delta_t^V + \gamma \delta_{t+1}^V`$ 的过程中，哪一项发生了抵消？这个抵消现象在一般 k-step 情形下对应什么数学结构？

<details>
<summary>查看参考答案</summary>

参考答案：$`\gamma V(s_{t+1})`$ 与 $`-\gamma V(s_{t+1})`$ 抵消。在一般情形下，这对应于 telescoping sum（裂项求和）结构：相邻项中的 $`V(s_{t+l})`$ 正负配对消去，最终只剩首尾两项 $`-V(s_t)`$ 和 $`\gamma^k V(s_{t+k})`$。

</details>

**题目 2：** 如果 critic 网络完美学习了真实价值函数（$`V = V^\pi`$），那么对于任意 $`k`$，$`\hat{A}_t^{(k)}`$ 的偏差是多少？此时选择不同 $`k`$ 还有意义吗？

<details>
<summary>查看参考答案</summary>

参考答案：偏差为零（对所有 $`k`$ 都是无偏估计）。但仍有意义，因为不同 $`k`$ 的方差不同。$`k=1`$ 方差最小，是最优选择。只有在 $`V \neq V^\pi`$ 时才需要通过增大 $`k`$ 来降低偏差。

</details>

**题目 3：** 在 GAE 的推导中，交换求和顺序后内层级数 $`\sum_{k=l+1}^{\infty} \lambda^{k-1}`$ 的结果是什么？它与外层的 $`(1-\lambda)`$ 因子有什么关系？

<details>
<summary>查看参考答案</summary>

参考答案：结果为 $`\lambda^l / (1-\lambda)`$。它与外层的 $`(1-\lambda)`$ 恰好约去，使得最终的 GAE 表达式中不再显式出现 $`(1-\lambda)`$ 因子，只剩下 $`(\gamma\lambda)^l`$ 的简洁形式。

</details>

**题目 4：** 在一个最大步数为 200 的环境中，如果在第 200 步发生了截断（truncation），计算该步的 $`\delta_t^V`$ 时是否应该包含 $`\gamma V(s_{t+1})`$？为什么？

<details>
<summary>查看参考答案</summary>

参考答案：应该包含。截断不是真正的终止，MDP 在该状态下仍有后续价值。不包含 bootstrap 会导致系统性地低估该状态的价值，引入不必要的偏差。只有真正终止（done=True 且非截断）才应省略 bootstrap 项。

</details>

**题目 5：** PPO 原论文使用 $`\gamma=0.99, \lambda=0.95`$。如果将 $`\lambda`$ 改为 0，GAE 退化为什么？这对训练的偏差和方差分别有什么影响？如果改为 1 呢？

<details>
<summary>查看参考答案</summary>

参考答案：$`\lambda=0`$ 退化为 $`\hat{A}_t^{(1)} = \delta_t^V`$（纯 TD 优势），偏差最大、方差最小，完全依赖 critic 质量。$`\lambda=1`$ 退化为蒙特卡洛优势 $`\sum \gamma^l r_{t+l} - V(s_t)`$，偏差为零、方差最大，critic 仅作基线。前者在 critic 不准确时可能导致策略朝错误方向更新，后者在稀疏奖励环境下可能导致训练极慢。

</details>

---

---

[上一章](chapter-03.md) · [返回目录](../README.md) · [阅读路线与符号表](README.md) · [下一章](chapter-05.md)
