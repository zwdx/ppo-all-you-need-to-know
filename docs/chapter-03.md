[上一章](chapter-02.md) · [返回目录](../README.md) · [阅读路线与符号表](README.md) · [下一章](chapter-04.md)

---

<a name="chapter-3"></a>

# 第 3 章 策略梯度方法

**本章目录**

- [3.1 优化目标的两种等价形式](#s3-1)
- [3.2 策略梯度定理的完整推导](#s3-2)
- [3.3 因果性与 Reward-to-Go](#s3-3)
- [3.4 用 Q 函数表达的策略梯度定理](#s3-4)
- [3.5 基线理论与优势函数](#s3-5)
- [3.6 策略梯度的偏差-方差分解框架](#s3-6)
- [3.7 Actor-Critic 架构](#s3-7)
- [3.8 n-step 方法与 A2C/A3C](#s3-8)
- [3.9 REINFORCE 伪代码](#s3-9)
- [3.10 策略梯度方法的固有缺陷](#s3-10)
- [3.11 本章小结与自测](#s3-11)

<a name="s3-1"></a>

## 3.1 优化目标的两种等价形式

在第 2 章中，我们已经建立了马尔可夫决策过程（Markov Decision Process, MDP）的基本框架，并定义了状态值函数 $`V^\pi(s)`$、动作值函数 $`Q^\pi(s,a)`$ 以及优势函数（advantage function）$`A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)`$。本章的核心任务是回答一个问题：**如何直接对策略 $`\pi_\theta`$ 的参数 $`\theta`$ 求梯度，从而用梯度上升改进策略？**

首先明确优化目标。设 $`\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots)`$ 为一条轨迹（trajectory），其累积折扣回报为：

$$
R(\tau) = \sum_{t=0}^{\infty} \gamma^t r_t
$$

其中 $`\gamma \in [0,1]`$ 为折扣因子（见第 2 章）。策略 $`\pi_\theta`$ 下的期望回报定义为：

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\big[R(\tau)\big] = \int p_\theta(\tau)\, R(\tau)\, d\tau \tag{3.1}
$$

这里 $`p_\theta(\tau)`$ 是策略 $`\pi_\theta`$ 生成的轨迹分布。具体地，若初始状态分布为 $`\rho_0(s_0)`$，环境动力学（transition dynamics）为 $`P(s_{t+1}|s_t, a_t)`$，则：

$$
p_\theta(\tau) = \rho_0(s_0) \prod_{t=0}^{\infty} P(s_{t+1}|s_t, a_t)\, \pi_\theta(a_t|s_t) \tag{3.2}
$$

式 (3.1) 是策略梯度的**轨迹形式**，也是后续推导的出发点。

另一种等价写法利用**折扣状态访问分布**（discounted state visitation distribution）$`d^\pi(s)`$，其定义为：

$$
d^\pi(s) = (1-\gamma) \sum_{t=0}^{\infty} \gamma^t \Pr(s_t = s \mid \pi) \tag{3.3}
$$

系数 $`(1-\gamma)`$ 使 $`d^\pi(s)`$ 归一化为概率分布（当 $`\gamma < 1`$ 时）。可以证明（推导见第 2 章）：

$$
J(\theta) = \frac{1}{1-\gamma} \mathbb{E}_{s \sim d^{\pi_\theta}}\big[V^{\pi_\theta}(s)\big] \tag{3.4}
$$

式 (3.4) 在理论分析中非常方便，但在实际计算策略梯度时，我们通常使用式 (3.1) 的轨迹形式。原因在于：$`d^{\pi_\theta}`$ 本身依赖于 $`\theta`$，对式 (3.4) 直接求梯度会产生 $`\nabla_\theta d^{\pi_\theta}`$ 项，该项涉及整个状态空间的无穷级数，难以处理。而策略梯度定理的精妙之处恰恰在于：**对式 (3.1) 求梯度后，结果恰好可以用 $`d^{\pi_\theta}`$ 下的期望来表达**，从而绕开了对 $`d^{\pi_\theta}`$ 直接求导的困难。

<a name="s3-2"></a>

## 3.2 策略梯度定理的完整推导

本节是本章的核心。我们给出两条等价的推导路径，最终得到同一个结果。

<a name="s3-2-1"></a>

### 3.2.1 路径一：直接展开轨迹分布的梯度

从式 (3.1) 出发，假设积分与微分可交换（正则性条件，要求 $`\pi_\theta(a|s)`$ 关于 $`\theta`$ 连续可微且支撑集不随 $`\theta`$ 变化），有：

$$
\nabla_\theta J(\theta) = \nabla_\theta \int p_\theta(\tau)\, R(\tau)\, d\tau = \int \nabla_\theta p_\theta(\tau)\, R(\tau)\, d\tau \tag{3.5}
$$

关键技巧是**乘除 $`p_\theta(\tau)`$**（即似然比技巧 / likelihood ratio trick，其数学基础见第 1 章）：

$$
\nabla_\theta p_\theta(\tau) = p_\theta(\tau) \cdot \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} = p_\theta(\tau) \cdot \nabla_\theta \log p_\theta(\tau) \tag{3.6}
$$

代入式 (3.5)：

$$
\nabla_\theta J(\theta) = \int p_\theta(\tau) \cdot \nabla_\theta \log p_\theta(\tau) \cdot R(\tau)\, d\tau = \mathbb{E}_{\tau \sim \pi_\theta}\big[R(\tau) \cdot \nabla_\theta \log p_\theta(\tau)\big] \tag{3.7}
$$

现在展开 $`\log p_\theta(\tau)`$。由式 (3.2) 取对数：

$$
\log p_\theta(\tau) = \log \rho_0(s_0) + \sum_{t=0}^{\infty} \Big[\log P(s_{t+1}|s_t, a_t) + \log \pi_\theta(a_t|s_t)\Big] \tag{3.8}
$$

对 $`\theta`$ 求梯度时，注意 $`\rho_0(s_0)`$ 和 $`P(s_{t+1}|s_t, a_t)`$ **均不含 $`\theta`$**——它们由环境和初始分布决定，与策略参数无关。这正是 **model-free**（模型无关）方法的关键：我们不需要知道环境动力学的解析形式，也不需要对其求导。因此：

$$
\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{\infty} \nabla_\theta \log \pi_\theta(a_t|s_t) \tag{3.9}
$$

将式 (3.9) 代入式 (3.7)，得到 **REINFORCE** 算法的策略梯度估计量：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[R(\tau) \sum_{t=0}^{\infty} \nabla_\theta \log \pi_\theta(a_t|s_t)\right] \tag{3.10}
$$

这就是策略梯度定理的最基本形式。它的含义非常直观：如果某条轨迹的回报 $`R(\tau)`$ 较高，就增大该轨迹上所有动作的概率；反之则减小。

<a name="s3-2-2"></a>

### 3.2.2 路径二：似然比技巧一步得到

第 1 章已经证明了似然比恒等式：对于任意关于 $`\theta`$ 可微的概率密度 $`p_\theta(x)`$ 和函数 $`f(x)`$，只要正则性条件满足，就有：

$$
\nabla_\theta \mathbb{E}_{x \sim p_\theta}[f(x)] = \mathbb{E}_{x \sim p_\theta}\big[f(x) \cdot \nabla_\theta \log p_\theta(x)\big] \tag{3.11}
$$

直接将 $`x = \tau`$、$`f(\tau) = R(\tau)`$、$`p_\theta = p_\theta(\tau)`$ 代入式 (3.11)，立即得到式 (3.7)。再将式 (3.9) 代入，同样得到式 (3.10)。

两条路径完全等价。路径一的优势是展示了"为什么环境动力学项消失"这一关键细节；路径二的优势是简洁，且揭示了策略梯度本质上是似然比估计量的一个特例。

<a name="s3-2-3"></a>

### 3.2.3 策略梯度定理的标准形式

式 (3.10) 虽然正确，但还不是最常用的形式。我们将在 3.3 节引入因果性简化后，在 3.4 节将其改写为用 $`Q^\pi`$ 表达的标准形式。

<a name="s3-3"></a>

## 3.3 因果性与 Reward-to-Go

式 (3.10) 中，每个时刻 $`t`$ 的策略梯度 $`\nabla_\theta \log \pi_\theta(a_t|s_t)`$ 被整条轨迹的总回报 $`R(\tau)`$ 加权。然而直觉告诉我们：**时刻 $`t`$ 的动作不可能影响 $`t`$ 之前的奖励**。下面的定理将这一因果性（causality）严格化。

<a name="s3-3-1"></a>

### 3.3.1 因果性定理及其证明

**定理 3.1（因果性）**：对于 $`t' < t`$，有

$$
\mathbb{E}_{\tau \sim \pi_\theta}\big[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot r_{t'}\big] = 0 \tag{3.12}
$$

**证明**：利用条件期望的塔性质（tower property / law of iterated expectations）。记 $`\tau_{0:t} = (s_0, a_0, r_0, \dots, s_t, a_t)`$ 为截至时刻 $`t`$ 的历史。注意 $`r_{t'}`$ 在 $`t' < t`$ 时是 $`\tau_{0:t}`$ 的确定性函数，而 $`\nabla_\theta \log \pi_\theta(a_t|s_t)`$ 仅依赖于 $`(s_t, a_t)`$。我们有：

$$
\begin{aligned}
\mathbb{E}_{\tau}\big[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot r_{t'}\big] &= \mathbb{E}_{\tau_{0:t}}\left[r_{t'} \cdot \mathbb{E}_{a_t \sim \pi_\theta(\cdot|s_t)}\big[\nabla_\theta \log \pi_\theta(a_t|s_t) \,\big|\, \tau_{0:t-1}, s_t\big]\right] \\
&= \mathbb{E}_{\tau_{0:t}}\left[r_{t'} \cdot \nabla_\theta \sum_{a_t} \pi_\theta(a_t|s_t)\right] \\
&= \mathbb{E}_{\tau_{0:t}}\left[r_{t'} \cdot \nabla_\theta 1\right] \\
&= 0
\end{aligned} \tag{3.13}
$$

第二步利用了 $`s_t`$ 在给定 $`\tau_{0:t-1}`$ 后已由环境动力学确定（$`s_t`$ 是 $`\tau_{0:t-1}`$ 的函数），因此内层期望只对 $`a_t`$ 取；第三步利用了概率归一化 $`\sum_a \pi_\theta(a|s) = 1`$。**证毕。** ∎

这个证明的核心洞察是：$`\nabla_\theta \log \pi_\theta(a|s)`$ 关于 $`a \sim \pi_\theta(\cdot|s)`$ 的期望恒为零向量。这是 score function（得分函数）的基本性质。

<a name="s3-3-2"></a>

### 3.3.2 Reward-to-Go 形式

由因果性定理，式 (3.10) 中 $`R(\tau) = \sum_{k=0}^{\infty} \gamma^k r_k`$ 里所有 $`k < t`$ 的项对第 $`t`$ 步梯度的期望贡献为零，可以安全去掉。定义 **reward-to-go**（未来回报）：

$$
G_t = \sum_{k=t}^{\infty} \gamma^{k-t} r_k \tag{3.14}
$$

注意这里 $`G_t`$ 从 $`r_t`$ 开始累加，且以 $`\gamma^{k-t}`$ 而非 $`\gamma^k`$ 加权（即相对于时刻 $`t`$ 重新归一化了折扣）。于是策略梯度可以写为：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{\infty} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right] \tag{3.15}
$$

**关于折扣权重的说明**：有些文献使用 $`\tilde{G}_t = \sum_{k=t}^{\infty} \gamma^k r_k`$（保留全局折扣权重 $`\gamma^k`$），此时式 (3.15) 变为 $`\nabla_\theta J(\theta) = \mathbb{E}[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \tilde{G}_t]`$。这两种写法都是无偏的（因为被去掉的项期望为零），但方差特性略有不同。实践中更常用式 (3.14) 的定义，因为它与 $`Q^\pi(s_t, a_t)`$ 的尺度一致。

Reward-to-go 的意义不仅是降方差：它还将全轨迹回报替换为一个**只依赖当前及未来信息**的量，这使得在线（online）估计成为可能——我们不必等到整条轨迹结束才能计算梯度。

<a name="s3-4"></a>

## 3.4 用 Q 函数表达的策略梯度定理

回忆第 2 章的定义：$`Q^\pi(s,a) = \mathbb{E}_{\tau \sim \pi}[G_t \mid s_t = s, a_t = a]`$。对式 (3.15) 中的期望做条件分解：

$$
\begin{aligned}
\nabla_\theta J(\theta) &= \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{\infty} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right] \\
&= \sum_{t=0}^{\infty} \mathbb{E}_{s_t, a_t}\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \mathbb{E}[G_t \mid s_t, a_t]\right] \\
&= \sum_{t=0}^{\infty} \mathbb{E}_{s_t, a_t}\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot Q^{\pi_\theta}(s_t, a_t)\right]
\end{aligned} \tag{3.16}
$$

进一步，$`(s_t, a_t)`$ 的边缘分布正是 $`d^{\pi_\theta}(s) \cdot \pi_\theta(a|s) / (1-\gamma)`$（注意 $`d^{\pi_\theta}`$ 的定义中包含 $`(1-\gamma)`$ 归一化因子）。将所有时间步合并为一个期望，得到**策略梯度定理的标准形式**：

$$
\nabla_\theta J(\theta) = \frac{1}{1-\gamma} \mathbb{E}_{s \sim d^{\pi_\theta},\, a \sim \pi_\theta(\cdot|s)}\Big[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s,a)\Big] \tag{3.17}
$$

在实际算法中，常数因子 $`1/(1-\gamma)`$ 通常被吸收进学习率，因此常见的写法省略该因子：

$$
\nabla_\theta J(\theta) \propto \mathbb{E}_{s \sim d^{\pi_\theta},\, a \sim \pi_\theta(\cdot|s)}\Big[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s,a)\Big] \tag{3.18}
$$

式 (3.18) 是后续所有策略梯度算法的理论基石。它告诉我们：策略梯度的方向由 $`Q^{\pi_\theta}(s,a)`$ 加权决定——好的动作（$`Q`$ 值高）获得更大的概率提升。

<a name="s3-5"></a>

## 3.5 基线理论与优势函数

式 (3.18) 给出的梯度估计量是无偏的，但方差可能极大。本节介绍**基线（baseline）**方法，这是降低方差而不引入偏差的核心技术。

<a name="s3-5-1"></a>

### 3.5.1 基线的无偏性证明

**定理 3.2**：对于任意只依赖状态 $`s`$ 的函数 $`b(s)`$（称为基线），有

$$
\mathbb{E}_{a \sim \pi_\theta(\cdot|s)}\big[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)\big] = 0 \tag{3.19}
$$

**证明**：由于 $`b(s)`$ 不依赖 $`a`$，可以提到期望外面：

$$
b(s) \cdot \mathbb{E}_{a \sim \pi_\theta(\cdot|s)}\big[\nabla_\theta \log \pi_\theta(a|s)\big] = b(s) \cdot \nabla_\theta \sum_a \pi_\theta(a|s) = b(s) \cdot \nabla_\theta 1 = 0 \tag{3.20}
$$

**证毕。** ∎

这意味着我们可以将式 (3.18) 中的 $`Q^{\pi_\theta}(s,a)`$ 替换为 $`Q^{\pi_\theta}(s,a) - b(s)`$，梯度期望不变。特别地，取 $`b(s) = V^{\pi_\theta}(s)`$，得到：

$$
\nabla_\theta J(\theta) \propto \mathbb{E}_{s \sim d^{\pi_\theta},\, a \sim \pi_\theta(\cdot|s)}\Big[\nabla_\theta \log \pi_\theta(a|s) \cdot A^{\pi_\theta}(s,a)\Big] \tag{3.21}
$$

其中 $`A^{\pi_\theta}(s,a) = Q^{\pi_\theta}(s,a) - V^{\pi_\theta}(s)`$ 就是第 2 章定义的**优势函数**。

<a name="s3-5-2"></a>

### 3.5.2 为什么优势函数能显著降方差

直觉上，$`Q^{\pi}(s,a)`$ 的绝对值可能很大且波动剧烈，而 $`A^{\pi}(s,a)`$ 衡量的是"动作 $`a`$ 相对于平均水平的好坏"。由第 2 章的结论，$`\mathbb{E}_{a \sim \pi}[A^\pi(s,a)] = 0`$，即优势函数以零为中心，正负抵消。这使得梯度估计量的幅度大幅缩小，方差显著降低。

举例来说，如果某个状态下所有动作的 $`Q`$ 值都在 $`[100, 102]`$ 之间，直接用 $`Q`$ 加权会导致梯度被一个大常数主导；而减去 $`V \approx 101`$ 后，优势值在 $`[-1, 1]`$ 之间，梯度信号的信噪比大幅提升。

<a name="s3-5-3"></a>

### 3.5.3 最优基线的推导

既然任意 $`b(s)`$ 都不引入偏差，那么哪个 $`b(s)`$ 使方差最小？考虑单步情形（固定状态 $`s`$），梯度估计量为 $`g = \nabla_\theta \log \pi_\theta(a|s) \cdot (Q^\pi(s,a) - b(s))`$。为简化记号，令 $`\psi = \nabla_\theta \log \pi_\theta(a|s)`$，$`Q = Q^\pi(s,a)`$，则：

$$
\text{Var}[g] = \mathbb{E}_a\big[\|\psi \cdot (Q - b)\|^2\big] - \big\|\mathbb{E}_a[\psi \cdot (Q - b)]\big\|^2 \tag{3.22}
$$

第二项等于 $`\|\nabla_\theta J\|^2`$，与 $`b`$ 无关（因为基线不改变期望）。因此最小化方差等价于最小化第一项。对 $`b`$ 求导并令其为零：

$$
\frac{\partial}{\partial b} \mathbb{E}_a\big[\|\psi\|^2 (Q - b)^2\big] = -2\, \mathbb{E}_a\big[\|\psi\|^2 (Q - b)\big] = 0 \tag{3.23}
$$

解得**最优基线**：

$$
b^*(s) = \frac{\mathbb{E}_{a \sim \pi}\big[\|\nabla_\theta \log \pi_\theta(a|s)\|^2 \cdot Q^\pi(s,a)\big]}{\mathbb{E}_{a \sim \pi}\big[\|\nabla_\theta \log \pi_\theta(a|s)\|^2\big]} \tag{3.24}
$$

注意 $`b^*(s)`$ **不是** $`V^\pi(s)`$，而是被 $`\|\nabla_\theta \log \pi_\theta(a|s)\|^2`$ 加权的 $`Q`$ 值平均。只有当 $`\|\nabla_\theta \log \pi_\theta(a|s)\|^2`$ 对所有 $`a`$ 近似相等时，$`b^*(s) \approx V^\pi(s)`$。尽管如此，实践中几乎总是使用 $`V^\pi(s)`$ 作为基线，原因是：(1) $`V^\pi(s)`$ 容易通过价值网络估计；(2) 最优基线需要额外估计 $`\|\nabla \log \pi\|^2`$，计算开销大且自身有估计误差；(3) $`V^\pi(s)`$ 在实践中已经足够有效。

<a name="s3-6"></a>

## 3.6 策略梯度的偏差-方差分解框架

为了统一理解后续各种策略梯度变体（包括第 4 章的 GAE），我们建立一个一般性的分解框架。

将策略梯度估计量写成如下通用形式：

$$
\hat{g} = \sum_{t=0}^{\infty} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t \tag{3.25}
$$

其中 $`\hat{A}_t`$ 是对真实优势 $`A^{\pi_\theta}(s_t, a_t)`$ 的某种估计。不同的 $`\hat{A}_t`$ 选择构成一个**偏差-方差权衡谱**：

| $`\hat{A}_t`$ 的选择 | 偏差 | 方差 |
|---|---|---|
| $`R(\tau)`$（全轨迹回报） | 无偏 | 极高 |
| $`G_t`$（reward-to-go） | 无偏 | 高 |
| $`Q^\pi(s_t,a_t)`$（真实 Q 值） | 无偏 | 中等 |
| $`A^\pi(s_t,a_t)`$（真实优势） | 无偏 | 较低 |
| $`r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)`$（TD 残差） | 有偏（因 $`V_\phi \neq V^\pi`$） | 低 |
| $`k`$-step 回报减基线 | 偏差随 $`k`$ 递减 | 方差随 $`k`$ 递增 |

关键观察：当 $`\hat{A}_t`$ 使用了真实的 $`Q^\pi`$ 或 $`G_t`$ 时，估计量是无偏的；一旦引入近似价值函数 $`V_\phi`$ 来替代真实值，就会产生偏差。但这种偏差换来的是方差的显著降低——因为 $`V_\phi`$ 是一个平滑函数，不像采样回报那样随机波动。

**这个框架是第 4 章 GAE（Generalized Advantage Estimation）的直接入口**。GAE 通过一个参数 $`\lambda \in [0,1]`$ 在上述谱系中连续插值，实现偏差-方差的精细调控。具体的量化分析和 $`\lambda`$ 的最优选择将在第 4 章展开。

<a name="s3-7"></a>

## 3.7 Actor-Critic 架构

策略梯度定理告诉我们梯度方向由 $`Q^\pi`$ 或 $`A^\pi`$ 决定，但这些量本身是未知的。**Actor-Critic** 架构的核心思想是：用一个可学习的价值网络（critic）来估计这些量，同时用策略网络（actor）沿估计的梯度方向更新。

<a name="s3-7-1"></a>

### 3.7.1 两个网络的分工

- **Actor（策略网络）**$`\pi_\theta(a|s)`$：参数为 $`\theta`$，输出动作概率分布。按式 (3.21) 更新：
  $$
  \theta \leftarrow \theta + \alpha_\theta \cdot \nabla_\theta \log \pi_\theta(a|s) \cdot \hat{A}(s,a) \tag{3.26}
  $$

- **Critic（价值网络）**$`V_\phi(s)`$：参数为 $`\phi`$，估计状态值函数 $`V^{\pi_\theta}(s)`$。用于提供基线和/或优势估计。

<a name="s3-7-2"></a>

### 3.7.2 Critic 的损失函数

Critic 的目标是准确估计 $`V^{\pi_\theta}(s)`$。最直接的损失函数是最小化均方误差：

$$
L_V(\phi) = \mathbb{E}_{s \sim d^{\pi_\theta}}\Big[\big(V_\phi(s) - V^{\pi_\theta}(s)\big)^2\Big] \tag{3.27}
$$

但 $`V^{\pi_\theta}(s)`$ 本身未知。实践中用以下替代目标之一：

1. **Monte Carlo 目标**：$`L_V(\phi) = \mathbb{E}[(V_\phi(s_t) - G_t)^2]`$，其中 $`G_t`$ 是从 $`s_t`$ 出发的实际回报。无偏但方差高。

2. **TD 目标**：$`L_V(\phi) = \mathbb{E}[(V_\phi(s_t) - (r_t + \gamma V_\phi(s_{t+1})))^2]`$。注意这里目标中也包含 $`V_\phi`$ 自身，因此这是一个**半梯度**（semi-gradient）方法——对目标中的 $`V_\phi(s_{t+1})`$ 不反向传播梯度。这种做法引入了偏差（bootstrap bias），但大幅降低了方差，且允许在线更新。

3. **$`k`$-step TD 目标**：介于上述两者之间，用 $`k`$ 步实际回报加上 $`V_\phi(s_{t+k})`$ 作为目标。

<a name="s3-7-3"></a>

### 3.7.3 为什么两个网络的梯度必须分开

一个常见疑问是：能否把 actor 和 critic 的损失合并为一个 loss 一起反传？**不能**，原因有二：

1. **目标不同**：Actor 要最大化 $`J(\theta)`$，critic 要最小化 $`L_V(\phi)`$。两者的优化方向不一致，合并后会互相干扰。

2. **梯度耦合导致不稳定**：如果在 actor 的梯度中也对 $`\phi`$ 求导（或反过来），会产生交叉项 $`\nabla_\theta \nabla_\phi L`$，这相当于在联合参数空间上做二阶优化，极易发散。

因此标准做法是：**actor 和 critic 各自维护独立的优化器和梯度计算图**，仅在数据层面共享采样轨迹。

<a name="s3-7-4"></a>

### 3.7.4 Critic 误差对 Actor 的影响

当 $`V_\phi \neq V^{\pi_\theta}`$ 时，优势估计 $`\hat{A} = Q_{\text{est}} - V_\phi`$ 是有偏的。这种偏差会通过式 (3.26) 传导给 actor，导致策略更新偏离真实梯度方向。这是 Actor-Critic 方法的固有代价：我们用 critic 的估计误差换取了方差的降低。理论上，当 critic 收敛到真实 $`V^{\pi_\theta}`$ 时偏差消失；实践中，critic 永远只是近似，因此需要在偏差和方差之间找到平衡点。

<a name="s3-8"></a>

## 3.8 n-step 方法与 A2C/A3C

<a name="s3-8-1"></a>

### 3.8.1 A3C 的异步多 Worker 动机

策略梯度方法的一个根本问题是**样本相关性**：同一条轨迹中的连续样本高度相关，导致梯度估计的方差增大、训练不稳定。在 DQN 等 off-policy 方法中，这个问题通过经验回放（experience replay）打破相关性。但在 on-policy 策略梯度方法中，经验回放不可直接使用——因为回放的数据是由旧策略采集的，而策略梯度要求数据来自当前策略（off-policy 修正需要重要性采样，见第 5 章）。

**A3C（Asynchronous Advantage Actor-Critic）** 提出了一种巧妙的替代方案：启动多个并行的 worker，每个 worker 独立与环境交互并计算梯度，异步地将梯度累加到一个共享的全局参数上。并行采样的效果类似于经验回放——不同 worker 处于轨迹的不同阶段、不同状态，汇总后的梯度近似于从混合分布中采样，有效打破了样本相关性。

<a name="s3-8-2"></a>

### 3.8.2 A2C：同步版本

**A2C（Advantage Actor-Critic）** 是 A3C 的同步变体：所有 worker 并行采集固定长度的轨迹片段，然后同步地汇总梯度并更新全局参数，再广播新参数给所有 worker。

实践中 A2C 往往优于 A3C，原因是：(1) 同步更新避免了过时梯度（stale gradient）问题；(2) 可以使用更大的批量（batch size），提高 GPU 利用率；(3) 实现更简单，调试更容易。A3C 的异步优势在现代硬件上并不明显，因为 GPU 并行已经取代了 CPU 多核并行的角色。

<a name="s3-8-3"></a>

### 3.8.3 两者的等价性讨论

从理论上看，A2C 和 A3C 都是对同一个策略梯度目标 (3.21) 的随机近似。区别仅在于梯度估计的聚合方式：A3C 是异步随机梯度下降（asynchronous SGD），A2C 是小批量同步 SGD。在凸优化中两者收敛性有经典结果；在非凸的策略优化中，同步版本通常更稳定。

<a name="s3-9"></a>

## 3.9 REINFORCE 伪代码

以下是 REINFORCE with baseline 算法的数学伪代码，不涉及具体编程框架：

```
算法 REINFORCE with Baseline
输入：可微策略 π_θ，可微价值函数 V_φ，学习率 α_θ, α_φ，折扣因子 γ
初始化 θ, φ

重复直到收敛：
    1. 用当前策略 π_θ 采样一条完整轨迹 τ = (s_0, a_0, r_0, ..., s_T)
    2. 对 t = 0, 1, ..., T-1：
         计算 G_t = Σ_{k=t}^{T-1} γ^{k-t} r_k
    3. 计算基线 b_t = V_φ(s_t)
    4. 计算优势 Â_t = G_t - b_t
    5. 策略梯度估计：
         ĝ_θ = (1/T) Σ_{t=0}^{T-1} ∇_θ log π_θ(a_t|s_t) · Â_t
    6. 价值函数梯度估计：
         ĝ_φ = (1/T) Σ_{t=0}^{T-1} ∇_φ (V_φ(s_t) - G_t)²
    7. 更新参数：
         θ ← θ + α_θ · ĝ_θ
         φ ← φ - α_φ · ĝ_φ
```

注意步骤 5 和步骤 6 中的梯度是分别计算的，对应 3.7.3 节的讨论。

<a name="s3-10"></a>

## 3.10 策略梯度方法的固有缺陷

尽管策略梯度方法在理论上优雅且通用，但它有三个严重的实践缺陷，这些缺陷直接催生了后续的 TRPO（第 5 章）和 PPO（第 6 章）。

<a name="s3-10-1"></a>

### 3.10.1 样本效率极低

策略梯度是严格的 **on-policy** 方法：每条轨迹只能用于产生它的那个策略的梯度估计。一旦 $`\theta`$ 更新，旧轨迹就不再是当前策略下的无偏样本，不能复用。这意味着每次梯度更新都需要重新采样，样本效率远低于 off-policy 方法（如 DQN 的经验回放可以反复使用同一批数据）。

<a name="s3-10-2"></a>

### 3.10.2 对步长极其敏感

这是策略梯度最致命的问题。在监督学习中，即使学习率过大导致一步走偏，下一轮仍然可以用同样的训练数据纠正回来。但在策略梯度中，**数据分布本身随策略改变**。如果一步更新太大，新策略可能变得很差（例如在某些状态下给灾难性动作分配了高概率），而从这个差策略采集的数据只会进一步强化错误行为，形成恶性循环。这种崩溃往往是**不可恢复的**。

反过来，如果步长太小，收敛速度极慢，在高维连续控制问题中可能需要数百万步才能学到有意义的策略。

这就引出了一个核心矛盾：**我们需要一种机制，既能保证每步更新的改善幅度足够大以加速收敛，又能防止更新过大导致策略崩溃**。TRPO 通过约束 KL 散度来解决这个问题（第 5 章），PPO 通过裁剪（clip）目标函数来实现类似效果（第 6 章）。

<a name="s3-10-3"></a>

### 3.10.3 高方差

即使使用了 reward-to-go 和基线，策略梯度估计量的方差仍然远高于监督学习中的梯度。这是因为梯度估计依赖于随机采样的轨迹，而轨迹本身的随机性（来自策略和环境）会放大估计的不确定性。高方差意味着需要更多样本才能获得可靠的梯度方向，进一步加剧了样本效率问题。

这三个缺陷共同构成了 PPO 存在的直接理由：PPO 通过裁剪机制限制了每步更新的幅度（解决步长敏感性），通过多 epoch 重用数据提高了样本效率（配合裁剪保证安全性），并通过优势函数的标准化等手段降低了方差。这些解决方案的理论基础将在第 5、6 章详细展开。

<a name="s3-11"></a>

## 3.11 本章小结与自测

<a name="h119"></a>

### 要点回顾

1. **策略梯度定理**给出了 $`\nabla_\theta J(\theta)`$ 的精确表达式，推导有两条等价路径：直接展开轨迹分布梯度和似然比技巧。核心结果是 $`\nabla J \propto \mathbb{E}[\nabla \log \pi_\theta(a|s) \cdot Q^\pi(s,a)]`$。

2. **因果性**允许将全轨迹回报 $`R(\tau)`$ 替换为 reward-to-go $`G_t`$，去除过去奖励对当前梯度的噪声贡献，且不引入偏差。

3. **基线**$`b(s)`$ 减去后不改变梯度期望，最优基线被 $`\|\nabla \log \pi\|^2`$ 加权，实践中用 $`V^\pi(s)`$ 近似。减去 $`V^\pi`$ 后自然引出优势函数 $`A^\pi = Q^\pi - V^\pi`$。

4. **Actor-Critic** 用价值网络估计优势函数，critic 和 actor 必须独立优化；critic 的估计误差会给 actor 引入偏差，这是偏差-方差权衡的体现。

5. **策略梯度的三大缺陷**（样本效率低、步长敏感、高方差）是 TRPO 和 PPO 被提出的直接动机。

<a name="h120"></a>

### 自测题

**题目 1**：在策略梯度推导中，为什么 $`\nabla_\theta \log P(s_{t+1}|s_t, a_t) = 0`$？这对 model-free 方法意味着什么？

<details>
<summary>查看参考答案</summary>

参考答案：$`P(s_{t+1}|s_t, a_t)`$ 是环境动力学，不含策略参数 $`\theta`$，因此其对 $`\theta`$ 的梯度为零。这意味着策略梯度方法不需要知道环境的转移概率模型，只需通过与环境交互采样轨迹即可计算梯度，这是 model-free 方法的核心特征。

</details>

**题目 2**：证明 $`\mathbb{E}_{a \sim \pi_\theta(\cdot|s)}[\nabla_\theta \log \pi_\theta(a|s)] = 0`$，并说明这个性质在基线理论和因果性证明中分别扮演什么角色。

<details>
<summary>查看参考答案</summary>

参考答案：$`\mathbb{E}_a[\nabla \log \pi(a|s)] = \sum_a \pi(a|s) \cdot \nabla \log \pi(a|s) = \nabla \sum_a \pi(a|s) = \nabla 1 = 0`$。在基线理论中，它保证了 $`b(s) \cdot \mathbb{E}[\nabla \log \pi] = 0`$，即基线不引入偏差。在因果性证明中，它保证了 $`\mathbb{E}[\nabla \log \pi(a_t|s_t) \cdot r_{t'}] = 0`$（$`t'<t`$），因为内层期望对 $`a_t`$ 取时 $`r_{t'}`$ 是常数，提出来后乘以零向量得零。

</details>

**题目 3**：最优基线 $`b^*(s)`$ 与 $`V^\pi(s)`$ 何时相等？为什么实践中仍使用 $`V^\pi(s)`$？

<details>
<summary>查看参考答案</summary>

参考答案：当 $`\|\nabla_\theta \log \pi_\theta(a|s)\|^2`$ 对所有 $`a`$ 为常数时，$`b^*(s) = V^\pi(s)`$。实践中使用 $`V^\pi(s)`$ 是因为：最优基线需要额外估计 $`\|\nabla \log \pi\|^2`$ 的期望，计算成本高且自身有估计噪声；$`V^\pi(s)`$ 可通过 critic 网络自然获得；经验表明 $`V^\pi(s)`$ 已能有效降低方差。

</details>

**题目 4**：解释为什么策略梯度方法不能像监督学习那样对同一批数据跑多个 epoch。这与 PPO 的设计有什么关系？

<details>
<summary>查看参考答案</summary>

参考答案：策略梯度是 on-policy 的，梯度估计要求数据来自当前策略。更新 $`\theta`$ 后数据分布改变，旧数据不再提供当前策略的无偏梯度，强行复用会导致偏差甚至策略崩溃。PPO 通过裁剪目标函数限制了新旧策略之间的比率偏离程度，使得在一定范围内安全地重用数据成为可能，从而提高了样本效率。详见第 6 章。

</details>

**题目 5**：在 Actor-Critic 中，如果用 $`r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)`$ 作为优势估计，这个估计量是有偏还是无偏的？偏差来源是什么？

<details>
<summary>查看参考答案</summary>

参考答案：有偏。偏差来源于两个方面：(1) $`V_\phi`$ 是 $`V^{\pi_\theta}`$ 的近似，$`V_\phi \neq V^{\pi_\theta}`$ 导致 bootstrap 偏差；(2) TD 目标 $`r_t + \gamma V_\phi(s_{t+1})`$ 本身是 $`Q^{\pi_\theta}(s_t, a_t)`$ 的有偏估计（除非 $`V_\phi = V^{\pi_\theta}`$）。但当 $`V_\phi`$ 足够准确时，偏差很小，而方差的降低使得总体均方误差反而更小。

</details>

---

---

[上一章](chapter-02.md) · [返回目录](../README.md) · [阅读路线与符号表](README.md) · [下一章](chapter-04.md)
