[上一章](chapter-01.md) · [返回目录](../README.md) · [阅读路线与符号表](README.md) · [下一章](chapter-03.md)

---

<a id="chapter-2"></a>

# 第 2 章 强化学习核心概念

**本章目录**

- [2.1 为什么用马尔可夫决策过程建模](#user-content-s2-1)
- [2.2 轨迹的概率表达式](#user-content-s2-2)
- [2.3 回报与折扣因子](#user-content-s2-3)
- [2.4 价值函数三件套](#user-content-s2-4)
- [2.5 贝尔曼期望方程](#user-content-s2-5)
- [2.6 贝尔曼最优方程与最优策略](#user-content-s2-6)
- [2.7 时序差分误差](#user-content-s2-7)
- [2.8 蒙特卡洛估计与时序差分估计的对比](#user-content-s2-8)
- [2.9 策略的形式、分类与强化学习的根本困境](#user-content-s2-9)
- [2.10 本章小结与自测](#user-content-s2-10)

<a id="s2-1"></a>

## 2.1 为什么用马尔可夫决策过程建模

<a id="s2-1-1"></a>

### 2.1.1 Agent-Environment 交互循环

强化学习（Reinforcement Learning, RL）研究的核心问题是：一个智能体（agent）如何在与环境（environment）的持续交互中，通过试错来最大化某种累积信号。这一交互以离散时间步 $`t = 0, 1, 2, \dots`$ 展开：在每个时间步 $`t`$，agent 观察到环境的状态（state）$`s_t`$，据此选择一个动作（action）$`a_t`$；环境接收到动作后，以某个概率转移到新状态 $`s_{t+1}`$，同时向 agent 发出一个标量奖励（reward）$`r_{t+1}`$。如此往复，形成一条无限或有限的交互序列：

```math
s_0, a_0, r_1, s_1, a_1, r_2, s_2, a_2, r_3, \dots
```

为了对这一过程进行严格的数学分析，我们需要一个形式化框架。马尔可夫决策过程（Markov Decision Process, MDP）正是这样一个框架，它将上述交互抽象为一个五元组 $`(\mathcal{S}, \mathcal{A}, P, R, \gamma)`$，下面逐一给出严格定义。

<a id="s2-1-2"></a>

### 2.1.2 MDP 五元组的严格定义

**状态空间 $`\mathcal{S}`$**：所有可能状态的集合。可以是有限集、可数无限集，甚至是连续空间。本章为推导清晰，除特别说明外均假设 $`\mathcal{S}`$ 有限。

**动作空间 $`\mathcal{A}`$**：所有可能动作的集合。同样假设有限。在某些问题中，可用动作依赖于当前状态，记为 $`\mathcal{A}(s)`$；为简化记号，我们通常假定 $`\mathcal{A}(s) = \mathcal{A}`$ 对所有 $`s`$ 成立，这不影响理论结论。

**转移概率 $`P`$**：环境的动力学模型。定义为条件概率分布

```math
P(s' \mid s, a) = \Pr(s_{t+1} = s' \mid s_t = s, a_t = a),
```

满足归一化条件 $`\sum_{s' \in \mathcal{S}} P(s' \mid s, a) = 1`$ 对所有 $`(s, a)`$ 成立。注意这里只依赖当前状态-动作对 $`(s, a)`$，不依赖更早的历史——这就是马尔可夫性质，下文详述。

**奖励函数 $`R`$**：我们采用期望奖励的形式，定义为

```math
R(s, a) = \mathbb{E}[r_{t+1} \mid s_t = s, a_t = a].
```

更一般地，奖励可以是随机变量，其分布也仅由 $`(s, a)`$（或 $`(s, a, s')`$）决定。为推导简洁，本章多数地方使用期望奖励 $`R(s,a)`$；当需要区分即时奖励的随机性时，会显式写出 $`r_{t+1}`$。

**折扣因子 $`\gamma`$**：一个标量，满足 $`0 \leq \gamma \lt  1`$（在无限 horizon 设定下严格要求 $`\gamma \lt  1`$）。其作用与解释见 2.3 节。

<a id="s2-1-3"></a>

### 2.1.3 马尔可夫性质及其局限

MDP 的核心假设是**马尔可夫性质（Markov property）**：给定当前状态和动作，下一状态的条件分布与之前的全部历史无关。形式化地，对任意 $`t`$ 和任意历史 $`h_t = (s_0, a_0, r_1, \dots, s_t, a_t)`$，有

```math
\Pr(s_{t+1} \mid h_t) = \Pr(s_{t+1} \mid s_t, a_t) = P(s_{t+1} \mid s_t, a_t).
```

这意味着状态 $`s_t`$ 是对历史的「充分统计量」——它已经包含了预测未来所需的全部信息。在实际问题中，如果原始观测不满足马尔可夫性质（例如只能看到部分信息），则问题应被建模为部分可观测马尔可夫决策过程（Partially Observable MDP, POMDP）。处理 POMDP 的标准方法包括引入信念状态（belief state）或使用循环神经网络等记忆结构来隐式维护历史信息。不过，PPO 算法本身仍然建立在 MDP 框架之上；当应用于部分可观测环境时，通常将观测历史编码为网络的输入，使策略在「扩展状态」上近似满足马尔可夫性。这一话题超出本章范围，此处仅作提示。

<a id="s2-2"></a>

## 2.2 轨迹的概率表达式

<a id="s2-2-1"></a>

### 2.2.1 轨迹的定义

一条轨迹（trajectory），也称回合（episode），是从初始状态到终止状态（或无限延续）的一段完整交互记录。记有限 horizon $`H`$ 的轨迹为

```math
\tau = (s_0, a_0, r_1, s_1, a_1, r_2, \dots, s_{H-1}, a_{H-1}, r_H).
```

对于无限 horizon，$`\tau`$ 是一个无限序列；但在实际推导中，我们通常先处理有限 horizon 再取极限，或直接利用 $`\gamma \lt  1`$ 保证级数收敛。

<a id="s2-2-2"></a>

### 2.2.2 轨迹概率的连乘分解

设 agent 遵循策略 $`\pi(a \mid s)`$（即 $`\Pr(a_t = a \mid s_t = s) = \pi(a \mid s)`$），环境的初始状态分布为 $`p(s_0)`$。则轨迹 $`\tau`$ 出现的联合概率为

```math
p(\tau) = p(s_0) \prod_{t=0}^{H-1} \pi(a_t \mid s_t) \, P(s_{t+1} \mid s_t, a_t).
```

这个分解的关键在于：**环境动力学项 $`P(s_{t+1} \mid s_t, a_t)`$ 与策略项 $`\pi(a_t \mid s_t)`$ 是完全可分离的**。原因如下：

1. 根据链式法则，联合概率可以写成条件概率的连乘：

```math
p(\tau) = p(s_0) \prod_{t=0}^{H-1} p(a_t \mid s_t, \text{history before } t) \cdot p(s_{t+1} \mid s_t, a_t, \text{history before } t).
```

2. 由策略的定义，agent 选择动作只依赖当前状态：$`p(a_t \mid s_t, \text{history}) = \pi(a_t \mid s_t)`$。

3. 由马尔可夫性质，环境转移只依赖当前状态-动作对：$`p(s_{t+1} \mid s_t, a_t, \text{history}) = P(s_{t+1} \mid s_t, a_t)`$。

因此两项各自独立地只依赖 $`(s_t, a_t)`$，互不耦合。这一可分离性是第 3 章策略梯度定理推导的基石：当我们对 $`\log p(\tau)`$ 关于策略参数求导时，只有 $`\pi(a_t \mid s_t)`$ 项贡献梯度，$`P(s_{t+1} \mid s_t, a_t)`$ 项作为常数消失。这一点请读者牢记，我们在第 3 章会反复使用。

<a id="s2-3"></a>

## 2.3 回报与折扣因子

<a id="s2-3-1"></a>

### 2.3.1 折扣回报的定义

从时间步 $`t`$ 出发的折扣回报（discounted return）定义为

```math
G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1} = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots
```

其中 $`\gamma \in [0, 1)`$ 是折扣因子（discount factor）。注意下标约定：$`r_{t+1}`$ 是在时间步 $`t`$ 执行动作后、到达 $`s_{t+1}`$ 时收到的奖励。

<a id="s2-3-2"></a>

### 2.3.2 折扣因子的三重解释

$`\gamma`$ 的存在并非纯粹的数学技巧，它有至少三层含义：

1. **数学收敛性**：若即时奖励有界，即 $`|r_t| \leq R_{\max}`$ 对所有 $`t`$ 成立，则

```math
|G_t| \leq \sum_{k=0}^{\infty} \gamma^k R_{\max} = \frac{R_{\max}}{1 - \gamma}.
```
   该几何级数在 $`\gamma \lt  1`$ 时收敛，保证了 $`G_t`$ 是有界的。下面给出完整证明。

2. **时间偏好**：$`\gamma \lt  1`$ 意味着未来的奖励比眼前的奖励「价值更低」。这反映了经济学中的时间偏好：今天的 1 元比明天的 1 元更有价值。$`\gamma`$ 越接近 1，agent 越「有耐心」。

3. **不确定性建模**：可以将 $`\gamma`$ 理解为「环境在每个时间步以概率 $`\gamma`$ 继续运行、以概率 $`1-\gamma`$ 终止」的等价表述。在这种解释下，折扣回报等于未折扣回报在随机终止时间下的期望。

<a id="s2-3-3"></a>

### 2.3.3 有界性证明

**命题**：若 $`|r_t| \leq R_{\max}`$ 对所有 $`t`$ 成立且 $`0 \leq \gamma \lt  1`$，则 $`|G_t| \leq \dfrac{R_{\max}}{1-\gamma}`$。

**证明**：由三角不等式和几何级数求和公式，

```math
|G_t| = \left| \sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \right| \leq \sum_{k=0}^{\infty} \gamma^k |r_{t+k+1}| \leq \sum_{k=0}^{\infty} \gamma^k R_{\max} = R_{\max} \sum_{k=0}^{\infty} \gamma^k = \frac{R_{\max}}{1 - \gamma}.
```

最后一步利用了 $`\gamma \in [0, 1)`$ 时几何级数 $`\sum_{k=0}^\infty \gamma^k = \frac{1}{1-\gamma}`$ 的收敛性。$`\square`$

<a id="s2-3-4"></a>

### 2.3.4 $`\gamma \to 1`$ 时的行为

当 $`\gamma \to 1`$ 时，上界 $`\frac{R_{\max}}{1-\gamma} \to \infty`$，折扣回报可能发散。在有限 horizon 问题中，可以安全地令 $`\gamma = 1`$，因为求和项数有限。但在无限 horizon 设定下，$`\gamma = 1`$ 要求奖励序列本身具有足够的衰减性或零均值结构才能保证 $`G_t`$ 有意义。实践中，$`\gamma`$ 通常取 0.99 或 0.999，在「足够长远」与「数值稳定」之间折中。

<a id="s2-4"></a>

## 2.4 价值函数三件套

<a id="s2-4-1"></a>

### 2.4.1 状态价值函数

策略 $`\pi`$ 下的状态价值函数（state value function）定义为从状态 $`s`$ 出发、此后遵循策略 $`\pi`$ 所获得的期望折扣回报：

```math
V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s] = \mathbb{E}_\pi\left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \;\middle|\; s_t = s \right].
```

这里 $`\mathbb{E}_\pi[\cdot]`$ 表示期望是在策略 $`\pi`$ 和环境动力学 $`P`$ 共同诱导的轨迹分布下取的。由于 MDP 的时齐性（time-homogeneity），$`V^\pi(s)`$ 不依赖于具体的时间步 $`t`$。

<a id="s2-4-2"></a>

### 2.4.2 动作价值函数

动作价值函数（action value function）定义为在状态 $`s`$ 下先执行动作 $`a`$、此后遵循策略 $`\pi`$ 的期望折扣回报：

```math
Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a] = \mathbb{E}_\pi\left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k+1} \;\middle|\; s_t = s, a_t = a \right].
```

注意第一个动作 $`a_t = a`$ 是被指定的，不受 $`\pi`$ 控制；从 $`t+1`$ 步起才按 $`\pi`$ 行动。

<a id="s2-4-3"></a>

### 2.4.3 优势函数

优势函数（advantage function）衡量在状态 $`s`$ 下选择动作 $`a`$ 相对于「平均水平」的好坏：

```math
A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s).
```

$`A^\pi(s, a) \gt  0`$ 表示动作 $`a`$ 优于策略 $`\pi`$ 在该状态的平均表现；$`A^\pi(s, a) \lt  0`$ 则表示劣于平均。

<a id="s2-4-4"></a>

### 2.4.4 三者关系式的推导

**关系一**：$`V^\pi(s) = \mathbb{E}_{a \sim \pi(\cdot \mid s)}[Q^\pi(s, a)]`$。

**推导**：由全期望公式（law of total expectation），

```math
V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s] = \sum_{a \in \mathcal{A}} \pi(a \mid s) \, \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a] = \sum_{a \in \mathcal{A}} \pi(a \mid s) \, Q^\pi(s, a).
```

这正是 $`Q^\pi(s, a)`$ 在策略 $`\pi(\cdot \mid s)`$ 下的期望。$`\square`$

**关系二**：$`\mathbb{E}_{a \sim \pi(\cdot \mid s)}[A^\pi(s, a)] = 0`$。

**推导**：利用优势函数的定义和关系一，

```math
\begin{aligned}
\mathbb{E}_{a \sim \pi(\cdot \mid s)}[A^\pi(s, a)] &= \sum_{a \in \mathcal{A}} \pi(a \mid s) \left[ Q^\pi(s, a) - V^\pi(s) \right] \\
&= \sum_{a \in \mathcal{A}} \pi(a \mid s) Q^\pi(s, a) - V^\pi(s) \sum_{a \in \mathcal{A}} \pi(a \mid s) \\
&= V^\pi(s) - V^\pi(s) \cdot 1 \\
&= 0.
\end{aligned}
```

第二步利用了 $`\sum_a \pi(a \mid s) = 1`$。$`\square`$

这个**零均值性质**极为重要：它说明优势函数是以策略自身为基准的「去中心化」度量。在第 3 章的策略梯度和第 4 章的优势估计中，这一性质保证了基线（baseline）的引入不会改变梯度的期望，只会降低方差。

<a id="s2-5"></a>

## 2.5 贝尔曼期望方程

<a id="s2-5-1"></a>

### 2.5.1 从回报的递归结构出发

折扣回报有一个天然的递归分解：

```math
G_t = r_{t+1} + \gamma G_{t+1}.
```

这是因为

```math
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots = r_{t+1} + \gamma (r_{t+2} + \gamma r_{t+3} + \cdots) = r_{t+1} + \gamma G_{t+1}.
```

这一简单的恒等式是推导所有贝尔曼方程的起点。

<a id="s2-5-2"></a>

### 2.5.2 $`V^\pi`$ 的贝尔曼期望方程

对 $`G_t = r_{t+1} + \gamma G_{t+1}`$ 两边在 $`s_t = s`$ 条件下取期望：

```math
V^\pi(s) = \mathbb{E}_\pi[r_{t+1} + \gamma G_{t+1} \mid s_t = s].
```

展开期望：首先对 $`a_t \sim \pi(\cdot \mid s)`$ 取期望，然后对 $`s_{t+1} \sim P(\cdot \mid s, a)`$ 取期望：

```math
V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \left[ R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^\pi(s') \right].
```

这就是 $`V^\pi`$ 的**贝尔曼期望方程（Bellman expectation equation）**。它表达了 $`V^\pi(s)`$ 与后继状态价值之间的自洽关系。

<a id="s2-5-3"></a>

### 2.5.3 $`Q^\pi`$ 的贝尔曼期望方程

类似地，对 $`Q^\pi(s, a)`$ 有

```math
Q^\pi(s, a) = R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \sum_{a' \in \mathcal{A}} \pi(a' \mid s') Q^\pi(s', a').
```

或者更简洁地，利用 $`V^\pi(s') = \sum_{a'} \pi(a' \mid s') Q^\pi(s', a')`$，写成

```math
Q^\pi(s, a) = R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^\pi(s').
```

<a id="s2-5-4"></a>

### 2.5.4 算子形式与矩阵形式

定义**贝尔曼期望算子** $`T^\pi : \mathbb{R}^{|\mathcal{S}|} \to \mathbb{R}^{|\mathcal{S}|}`$：

```math
(T^\pi V)(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \left[ R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V(s') \right].
```

则贝尔曼期望方程等价于不动点方程 $`V^\pi = T^\pi V^\pi`$。

当 $`\mathcal{S}`$ 有限时，可将 $`V^\pi`$ 视为 $`|\mathcal{S}|`$ 维列向量，定义策略诱导的转移矩阵 $`P^\pi \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{S}|}`$，其元素为

```math
P^\pi(s, s') = \sum_{a \in \mathcal{A}} \pi(a \mid s) P(s' \mid s, a),
```

以及期望奖励向量 $`R^\pi \in \mathbb{R}^{|\mathcal{S}|}`$，其元素为 $`R^\pi(s) = \sum_a \pi(a \mid s) R(s, a)`$。则贝尔曼期望方程可写成紧凑的矩阵形式：

```math
V^\pi = R^\pi + \gamma P^\pi V^\pi.
```

移项得 $`(I - \gamma P^\pi) V^\pi = R^\pi`$，因此

```math
V^\pi = (I - \gamma P^\pi)^{-1} R^\pi.
```

**为什么 $`I - \gamma P^\pi`$ 可逆？** 因为 $`P^\pi`$ 是一个随机矩阵（每行元素非负且和为 1），其谱半径 $`\rho(P^\pi) \leq 1`$。当 $`\gamma \lt  1`$ 时，$`\gamma P^\pi`$ 的谱半径 $`\rho(\gamma P^\pi) \lt  1`$，因此 $`I - \gamma P^\pi`$ 的特征值全部不为零（具体地，其特征值为 $`1 - \gamma \lambda_i`$，其中 $`\lambda_i`$ 是 $`P^\pi`$ 的特征值，$`|\lambda_i| \leq 1`$，故 $`|1 - \gamma \lambda_i| \geq 1 - \gamma \gt  0`$），矩阵可逆。这也保证了 $`V^\pi`$ 是唯一确定的。

<a id="s2-6"></a>

## 2.6 贝尔曼最优方程与最优策略

<a id="s2-6-1"></a>

### 2.6.1 最优价值函数与贝尔曼最优算子

定义最优状态价值函数 $`V^*(s) = \max_\pi V^\pi(s)`$ 和最优动作价值函数 $`Q^*(s, a) = \max_\pi Q^\pi(s, a)`$。它们满足**贝尔曼最优方程（Bellman optimality equation）**：

```math
V^*(s) = \max_{a \in \mathcal{A}} \left[ R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^*(s') \right].
```

对应地，定义**贝尔曼最优算子** $`T^*`$：

```math
(T^* V)(s) = \max_{a \in \mathcal{A}} \left[ R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V(s') \right].
```

则 $`V^* = T^* V^*`$。

<a id="s2-6-2"></a>

### 2.6.2 $`\gamma`$-压缩性与 Banach 不动点定理

$`T^*`$ 是一个 $`\gamma`$-压缩映射（$`\gamma`$-contraction mapping）。具体来说，对任意两个价值函数 $`V_1, V_2 \in \mathbb{R}^{|\mathcal{S}|}`$，在无穷范数 $`\|V\|_\infty = \max_s |V(s)|`$ 下有

```math
\|T^* V_1 - T^* V_2\|_\infty \leq \gamma \|V_1 - V_2\|_\infty.
```

**证明思路**：对任意状态 $`s`$，设 $`a_1^*`$ 是 $`T^* V_1`$ 在 $`s`$ 处取 max 的动作，则

```math
(T^* V_1)(s) = R(s, a_1^*) + \gamma \sum_{s'} P(s' \mid s, a_1^*) V_1(s').
```

而 $`(T^* V_2)(s) \geq R(s, a_1^*) + \gamma \sum_{s'} P(s' \mid s, a_1^*) V_2(s')`$（因为 $`a_1^*`$ 不一定是 $`V_2`$ 的最优动作，max 给出的值不小于任何特定动作的值）。两式相减：

```math
(T^* V_1)(s) - (T^* V_2)(s) \leq \gamma \sum_{s'} P(s' \mid s, a_1^*) [V_1(s') - V_2(s')] \leq \gamma \|V_1 - V_2\|_\infty.
```

最后一步利用了 $`P(s' \mid s, a_1^*) \geq 0`$ 且 $`\sum_{s'} P(s' \mid s, a_1^*) = 1`$。交换 $`V_1, V_2`$ 的角色可得反向不等式，合起来即 $`|(T^* V_1)(s) - (T^* V_2)(s)| \leq \gamma \|V_1 - V_2\|_\infty`$ 对所有 $`s`$ 成立，取 max 即得压缩性。$`\square`$

由于 $`0 \leq \gamma \lt  1`$，根据 **Banach 不动点定理**（Banach fixed-point theorem），$`T^*`$ 在完备度量空间 $`(\mathbb{R}^{|\mathcal{S}|}, \|\cdot\|_\infty)`$ 上有**唯一**不动点 $`V^*`$，且从任意初始 $`V_0`$ 出发，迭代 $`V_{k+1} = T^* V_k`$ 都以指数速率收敛到 $`V^*`$。这就是**价值迭代（value iteration）** 算法的理论基础。同理，$`T^\pi`$ 也是 $`\gamma`$-压缩映射，保证了 $`V^\pi`$ 的唯一性和策略评估迭代的收敛性。

<a id="s2-6-3"></a>

### 2.6.3 最优策略的存在性

一旦获得 $`V^*`$ 或 $`Q^*`$，最优策略可以通过贪心选择得到：$`\pi^*(s) \in \arg\max_a Q^*(s, a)`$。可以证明，对任意有限 MDP，至少存在一个确定性最优策略。这意味着虽然 PPO 等方法使用随机策略进行搜索，但最优解本身可以在确定性策略类中找到。

<a id="s2-7"></a>

## 2.7 时序差分误差

<a id="s2-7-1"></a>

### 2.7.1 TD 误差的定义

时序差分（temporal difference, TD）误差定义为

```math
\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t),
```

其中 $`V`$ 是当前对真实价值函数的估计。TD 误差衡量的是「一步预测」与「实际观测到的即时奖励加下一步估值」之间的差异。它是几乎所有现代强化学习算法的核心信号。

<a id="s2-7-2"></a>

### 2.7.2 TD 误差与优势函数的关系

**关键命题**：当 $`V = V^\pi`$（即估值恰好等于真实状态价值）时，

```math
\mathbb{E}[\delta_t \mid s_t = s, a_t = a] = A^\pi(s, a).
```

**完整推导**：将 $`\delta_t`$ 的定义代入条件期望：

```math
\begin{aligned}
\mathbb{E}[\delta_t \mid s_t = s, a_t = a] &= \mathbb{E}[r_{t+1} + \gamma V^\pi(s_{t+1}) - V^\pi(s_t) \mid s_t = s, a_t = a] \\
&= \mathbb{E}[r_{t+1} \mid s_t = s, a_t = a] + \gamma \, \mathbb{E}[V^\pi(s_{t+1}) \mid s_t = s, a_t = a] - V^\pi(s).
\end{aligned}
```

第一项就是 $`R(s, a)`$。第二项展开为

```math
\mathbb{E}[V^\pi(s_{t+1}) \mid s_t = s, a_t = a] = \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^\pi(s').
```

回忆 2.5.3 节中 $`Q^\pi`$ 的贝尔曼期望方程：

```math
Q^\pi(s, a) = R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^\pi(s').
```

因此前两项之和恰好等于 $`Q^\pi(s, a)`$，于是

```math
\mathbb{E}[\delta_t \mid s_t = s, a_t = a] = Q^\pi(s, a) - V^\pi(s) = A^\pi(s, a).
```

$`\square`$

这个结果是第 4 章广义优势估计（Generalized Advantage Estimation, GAE）的基石：GAE 本质上是将多步 TD 误差加权求和，而当 $`V = V^\pi`$ 时，每一步 TD 误差的期望都精确等于优势函数。

<a id="s2-7-3"></a>

### 2.7.3 当 $`V \neq V^\pi`$ 时的情形

在实践中，$`V`$ 通常由神经网络参数化，不可能精确等于 $`V^\pi`$。此时 $`\mathbb{E}[\delta_t \mid s_t, a_t] = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)`$ 的期望不再等于 $`A^\pi(s_t, a_t)`$，而是包含了 $`V`$ 的近似误差。具体地，

```math
\mathbb{E}[\delta_t \mid s_t = s, a_t = a] = Q^\pi(s, a) - V(s) + \gamma \sum_{s'} P(s' \mid s, a)[V(s') - V^\pi(s')],
```

其中最后一项是 $`V`$ 在后继状态上的偏差经环境动力学传播后的结果。因此，用近似 $`V`$ 计算的 $`\delta_t`$ 是优势函数的**有偏估计**。偏差的大小取决于 $`V`$ 的拟合质量；随着训练推进、$`V \to V^\pi`$，偏差趋于零。这一偏差-方差的权衡将在第 4 章量化分析。

<a id="s2-8"></a>

## 2.8 蒙特卡洛估计与时序差分估计的对比

基于上述讨论，我们可以比较两种估计优势（或回报）的基本方法：

**蒙特卡洛（Monte Carlo, MC）估计**：直接用采样得到的完整回报 $`G_t`$ 作为 $`Q^\pi(s_t, a_t)`$ 的估计。由于 $`G_t`$ 是 $`Q^\pi(s_t, a_t)`$ 的无偏样本（$`\mathbb{E}[G_t \mid s_t, a_t] = Q^\pi(s_t, a_t)`$），MC 估计是**无偏的**。但其方差较大，因为 $`G_t`$ 包含了从 $`t`$ 到回合结束的所有随机奖励和转移的累积波动。

**时序差分（TD）估计**：用单步 TD 目标 $`r_{t+1} + \gamma V(s_{t+1})`$ 代替完整回报。当 $`V = V^\pi`$ 时，TD 估计也是无偏的；但当 $`V \neq V^\pi`$ 时，它引入了偏差。好处是方差显著小于 MC，因为它只用了一步随机奖励和一个（相对稳定的）估值，避免了长链随机变量的累积噪声。

这两种极端之间存在连续的插值空间，即 $`k`$-step 回报和 TD($`\lambda`$)，它们通过调节回溯长度来控制偏差-方差权衡。详细的量化分析和 GAE 的统一框架见第 4 章。

<a id="s2-9"></a>

## 2.9 策略的形式、分类与强化学习的根本困境

<a id="s2-9-1"></a>

### 2.9.1 随机策略与确定性策略

策略 $`\pi`$ 是从状态到动作空间上的概率分布的映射：$`\pi(\cdot \mid s)`$ 是 $`\mathcal{A}`$ 上的概率分布。这是最一般的策略形式，称为**随机策略（stochastic policy）**。

确定性策略（deterministic policy）$`\mu: \mathcal{S} \to \mathcal{A}`$ 是随机策略的特例：它可以被视为 Dirac $`\delta`$ 分布，即 $`\pi(a \mid s) = \delta(a, \mu(s))`$，将所有概率质量集中在单一动作上。在连续动作空间中，确定性策略梯度方法（如 DDPG）直接参数化 $`\mu_\theta(s)`$ 而非完整分布；但在 PPO 等 on-policy 方法中，通常使用随机策略以保持探索能力，并通过熵正则化鼓励随机性。

<a id="s2-9-2"></a>

### 2.9.2 参数化策略

在实际算法中，策略由参数 $`\theta \in \mathbb{R}^d`$ 索引，记为 $`\pi_\theta(a \mid s)`$。参数化通常通过神经网络实现：网络以状态 $`s`$ 为输入，输出动作分布的参数（例如离散动作的 logits 或连续动作的均值与标准差）。策略优化的目标就是在参数空间 $`\Theta`$ 中找到使期望回报最大的 $`\theta^*`$。这一参数化视角是第 3 章策略梯度方法的出发点。

<a id="s2-9-3"></a>

### 2.9.3 On-policy 与 Off-policy 的本质区别

**On-policy（同策略）**：用于训练的数据是由当前正在优化的策略 $`\pi_\theta`$ 自身生成的。也就是说，采样分布与目标策略分布一致。A2C、A3C、PPO 都是 on-policy 方法。

**Off-policy（异策略）**：用于训练的数据是由某个行为策略（behavior policy）$`b`$ 生成的，但目标是优化另一个目标策略（target policy）$`\pi_\theta`$，$`b \neq \pi_\theta`$。DQN、SAC、DDPG 都是 off-policy 方法。

两者的本质区别在于：**off-policy 方法面临采样分布与目标分布不匹配的问题**。为了用 $`b`$ 产生的数据来估计 $`\pi_\theta`$ 的性能，必须引入重要性采样（importance sampling）修正，这带来了额外的方差和稳定性挑战。这一问题的严格处理见第 5 章。PPO 虽然是 on-policy 方法，但它使用截断的重要性比率来限制策略更新幅度，这与 off-policy 的重要性采样有深层联系。

<a id="s2-9-4"></a>

### 2.9.4 强化学习的根本困境

至此，我们可以明确指出强化学习与监督学习的根本区别，这也是后续所有章节的理论动机所在：

> **在强化学习中，数据的分布由策略自身决定。改变策略就改变了数据分布，因此不能像监督学习那样在一个固定的数据集上反复做梯度下降。**

具体来说：
- 监督学习中，训练集 $`\{(x_i, y_i)\}`$ 是从某个固定的数据生成分布中独立同分布采样的，与模型参数无关。梯度下降的收敛理论建立在这一固定分布假设之上。
- 强化学习中，每条数据 $`(s_t, a_t, r_{t+1}, s_{t+1})`$ 都是在当前策略 $`\pi_\theta`$ 下采集的。一旦 $`\theta`$ 更新，旧数据的分布就不再代表新策略的分布。如果用旧数据对新策略做梯度更新，就会产生分布偏移（distribution shift）导致的偏差和不稳定。

这一困境催生了多种应对策略：on-policy 方法每次更新后重新采集数据；off-policy 方法通过经验回放（replay buffer）和重要性采样来缓解分布偏移；PPO 则通过裁剪机制限制单次更新的幅度，使得旧数据在新策略下仍然「大致有效」。这些方案的理论基础将在第 5 章和第 6 章展开。

<a id="s2-10"></a>

## 2.10 本章小结与自测

<a id="h87"></a>

### 要点回顾

1. **MDP 五元组** $`(\mathcal{S}, \mathcal{A}, P, R, \gamma)`$ 提供了 agent-environment 交互的严格数学框架；马尔可夫性质保证了状态是历史的充分统计量，使轨迹概率可分解为环境动力学与策略的可分离连乘。
2. **折扣回报** $`G_t`$ 在 $`\gamma \lt  1`$ 且有界奖励下有界，折扣因子兼具数学收敛性、时间偏好和不确定性三重含义。
3. **价值函数三件套** $`V^\pi, Q^\pi, A^\pi`$ 通过全期望公式相互关联，优势函数的零均值性质 $`\mathbb{E}_{a \sim \pi}[A^\pi(s,a)] = 0`$ 是基线方法和策略梯度的理论支柱。
4. **贝尔曼方程**将价值函数表达为递归不动点；$`T^\pi`$ 和 $`T^*`$ 都是 $`\gamma`$-压缩映射，Banach 不动点定理保证了唯一解和迭代收敛；矩阵形式 $`V^\pi = (I - \gamma P^\pi)^{-1} R^\pi`$ 揭示了可逆性的谱条件。
5. **TD 误差**在 $`V = V^\pi`$ 时是优势函数的无偏估计，在 $`V \neq V^\pi`$ 时是有偏估计；这一事实连接了价值函数近似与策略改进，是 GAE（第 4 章）和 PPO（第 6 章）的核心纽带。

<a id="h88"></a>

### 自测题

**题目 1**：写出有限 horizon $`H`$ 轨迹 $`\tau = (s_0, a_0, r_1, \dots, s_{H-1}, a_{H-1}, r_H)`$ 的概率表达式，并说明为什么环境动力学项与策略项是可分离的。

<details>
<summary>查看参考答案</summary>

参考答案：$`p(\tau) = p(s_0) \prod_{t=0}^{H-1} \pi(a_t \mid s_t) P(s_{t+1} \mid s_t, a_t)`$。可分离性来自两个独立性假设：策略选择动作只依赖当前状态（$`\pi(a_t \mid s_t)`$），环境转移只依赖当前状态-动作对（马尔可夫性质 $`P(s_{t+1} \mid s_t, a_t)`$），两者各自只依赖 $`(s_t, a_t)`$，互不耦合。

</details>

**题目 2**：证明 $`\mathbb{E}_{a \sim \pi(\cdot \mid s)}[A^\pi(s, a)] = 0`$，并解释这一性质的直观含义。

<details>
<summary>查看参考答案</summary>

参考答案：$`\sum_a \pi(a \mid s)[Q^\pi(s,a) - V^\pi(s)] = \sum_a \pi(a \mid s) Q^\pi(s,a) - V^\pi(s) = V^\pi(s) - V^\pi(s) = 0`$。直观含义：优势函数以策略自身的平均表现为基准，正优势和负优势恰好抵消，说明它是「去中心化」的度量。

</details>

**题目 3**：为什么矩阵 $`I - \gamma P^\pi`$ 在 $`\gamma \lt  1`$ 时可逆？

<details>
<summary>查看参考答案</summary>

参考答案：$`P^\pi`$ 是随机矩阵，谱半径 $`\rho(P^\pi) \leq 1`$。当 $`\gamma \lt  1`$ 时，$`\rho(\gamma P^\pi) \lt  1`$，故 $`I - \gamma P^\pi`$ 的所有特征值 $`1 - \gamma \lambda_i`$ 满足 $`|1 - \gamma \lambda_i| \geq 1 - \gamma \gt  0`$，没有零特征值，矩阵可逆。

</details>

**题目 4**：证明当 $`V = V^\pi`$ 时，$`\mathbb{E}[\delta_t \mid s_t = s, a_t = a] = A^\pi(s, a)`$。如果 $`V \neq V^\pi`$，这个等式还成立吗？为什么不成立？

<details>
<summary>查看参考答案</summary>

参考答案：$`\mathbb{E}[r_{t+1} + \gamma V^\pi(s_{t+1}) \mid s, a] = Q^\pi(s,a)`$（由 $`Q^\pi`$ 的贝尔曼方程），减去 $`V^\pi(s)`$ 即得 $`A^\pi(s,a)`$。当 $`V \neq V^\pi`$ 时不成立，因为 $`\mathbb{E}[\gamma V(s_{t+1}) \mid s,a] \neq \gamma \sum_{s'} P(s'|s,a) V^\pi(s')`$，额外产生了 $`V`$ 的近似偏差经环境动力学传播的误差项。

</details>

**题目 5**：用自己的话解释为什么强化学习不能像监督学习一样在固定数据集上做梯度下降。这对 PPO 的设计有什么影响？

<details>
<summary>查看参考答案</summary>

参考答案：RL 中数据由当前策略生成，更新策略后数据分布随之改变，旧数据不再代表新策略的分布，直接用会导致分布偏移和不稳定。PPO 通过裁剪重要性比率限制单次更新幅度，确保新旧策略足够接近，使旧数据在新策略下仍近似有效，从而允许对同一批数据进行多次小步更新。

</details>

---

---

[上一章](chapter-01.md) · [返回目录](../README.md) · [阅读路线与符号表](README.md) · [下一章](chapter-03.md)
