[上一章](chapter-04.md) · [返回目录](../README.md) · [阅读路线与符号表](README.md) · [下一章](chapter-06.md)

---

<a name="chapter-5"></a>

# 第 5 章 信任域方法与重要性采样

**本章目录**

- [5.1 引言：为什么需要限制策略更新幅度](#s5-1)
- [5.2 重要性采样的严格推导](#s5-2)
- [5.3 策略层面的重要性采样](#s5-3)
- [5.4 性能差异引理](#s5-4)
- [5.5 TRPO 的单调改进界](#s5-5)
- [5.6 TRPO 的实际求解形式](#s5-6)
- [5.7 自然梯度与 Fisher 信息矩阵](#s5-7)
- [5.8 共轭梯度法求解自然梯度](#s5-8)
- [5.9 TRPO 的实践代价与 PPO 的动机](#s5-9)
- [5.10 本章小结与自测](#s5-10)

<a name="s5-1"></a>

## 5.1 引言：为什么需要限制策略更新幅度

在第 3 章中，我们推导了策略梯度定理并给出了 REINFORCE 算法。然而该算法存在一个致命缺陷：对步长极度敏感。步长过大时，新策略可能跌入低回报区域且无法恢复；步长过小时，收敛速度不可接受。这一困境的根源在于：策略梯度只提供了目标函数 $J(\theta)$ 在 $\theta$ 处的**一阶局部信息**，而我们对 $J(\theta)$ 的全局几何一无所知。参数空间中的欧氏距离 $\|\Delta\theta\|$ 并不能反映新旧策略在**分布意义**上的真实差异——同一个 $\Delta\theta$ 在不同参数化下可能导致截然不同的策略变化。

本章的核心任务是建立一套数学框架，使得我们能够在保证「策略性能单调不降」的前提下尽可能大地更新策略。这条路线最终导向 TRPO（Trust Region Policy Optimization）算法，并为第 6 章 PPO 的设计提供理论基础。为此，我们需要依次解决以下问题：如何用旧策略的样本评估新策略（重要性采样）、新旧策略的性能差距如何精确表达（性能差异引理）、如何将该差距转化为可计算的优化约束（单调改进界）、以及如何在高维参数空间中高效求解该约束优化（自然梯度与共轭梯度）。

<a name="s5-2"></a>

## 5.2 重要性采样的严格推导

<a name="s5-2-1"></a>

### 5.2.1 基本恒等式与无偏性

设我们希望计算期望 $\mathbb{E}_{x \sim q}[f(x)]$，但只能从另一个分布 $p$ 中获取样本。假设 $p$ 的支撑覆盖 $q$ 的支撑，即对所有满足 $q(x) > 0$ 的 $x$ 都有 $p(x) > 0$。定义**重要性比（importance ratio）** 为：

$$
r(x) = \frac{q(x)}{p(x)}.
$$

则有如下恒等式：

$$
\mathbb{E}_{x \sim p}\left[r(x) f(x)\right] = \int p(x) \cdot \frac{q(x)}{p(x)} \cdot f(x) \, dx = \int q(x) f(x) \, dx = \mathbb{E}_{x \sim q}[f(x)].
$$

上述推导的关键步骤是在积分中乘除 $q(x)/p(x)$，然后交换测度。这要求 $p(x) > 0$ 在 $q(x) > 0$ 的所有点上成立；若存在 $q(x) > 0$ 但 $p(x) = 0$ 的点，则该点贡献被遗漏，估计量产生系统性偏差。因此在支撑条件满足时，IS 估计量是**严格无偏**的。

给定 $N$ 个独立同分布样本 $x_i \sim p$，IS 估计量为：

$$
\hat{\mu}_N = \frac{1}{N} \sum_{i=1}^{N} r(x_i) f(x_i).
$$

由上述恒等式，$\mathbb{E}[\hat{\mu}_N] = \mathbb{E}_{x \sim q}[f(x)]$，无偏性得证。

<a name="s5-2-2"></a>

### 5.2.2 方差爆炸与有效样本量

虽然 IS 估计量无偏，其方差却可能极大甚至无穷。展开方差表达式：

$$
\text{Var}_p\left[r(X)f(X)\right] = \mathbb{E}_p\left[r(X)^2 f(X)^2\right] - \left(\mathbb{E}_q[f(X)]\right)^2 = \int \frac{q(x)^2}{p(x)} f(x)^2 \, dx - \mu_q^2.
$$

当 $q/p$ 在某些区域取值很大且 $p$ 在这些区域的概率密度很小时，被积函数 $q^2/p$ 可以发散，导致方差无界。直观地说，如果行为策略 $p$ 很少访问目标策略 $q$ 认为重要的区域，那么偶尔落入这些区域的样本将获得极大的权重 $r(x)$，使估计量剧烈波动。这就是**方差爆炸**问题，也是后续所有「限制策略更新幅度」方法的根本动机。

实践中常用**有效样本量（Effective Sample Size, ESS）** 作为诊断指标：

$$
\text{ESS} = \frac{\left(\sum_{i=1}^{N} w_i\right)^2}{\sum_{i=1}^{N} w_i^2},
$$

其中 $w_i = r(x_i)$ 为非归一化权重。ESS 的取值范围为 $[1, N]$：当所有权重相等时 $\text{ESS} = N$（理想情况），当权重集中于单个样本时 $\text{ESS} \approx 1$（退化情况）。ESS 远小于 $N$ 表明 IS 估计不可靠。

<a name="s5-3"></a>

## 5.3 策略层面的重要性采样

<a name="s5-3-1"></a>

### 5.3.1 轨迹比率与单步比率

在强化学习中，我们希望用旧策略 $\pi_{\theta_\text{old}}$ 采集的轨迹来估计新策略 $\pi_\theta$ 的目标函数或梯度。最直接的想法是使用完整轨迹的重要性比：

$$
\rho(\tau) = \prod_{t=0}^{T-1} \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}.
$$

然而这个乘积的方差随时间步 $T$ **指数级增长**。粗略分析如下：若每步比率 $r_t$ 独立且 $\text{Var}(r_t) = \sigma^2$，则 $\text{Var}(\prod_t r_t) \approx (1+\sigma^2)^T - 1$，呈指数增长。即使各步比率不完全独立，在实际 MDP 中这种指数趋势依然显著。因此轨迹级别的 IS 在 horizon 稍大时完全不可用。

<a name="s5-3-2"></a>

### 5.3.2 为什么单步比率就够了

幸运的是，策略梯度定理（见第 3 章）告诉我们：

$$
\nabla J(\theta) = \mathbb{E}_{s \sim d^\pi, a \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^\pi(s,a)\right].
$$

注意这里的状态分布是 $d^\pi$，即当前策略 $\pi_\theta$ 自身的稳态（或折扣加权）状态分布。当我们用 $\pi_{\theta_\text{old}}$ 的样本来估计上式时，只需对动作采样做修正：

$$
\mathbb{E}_{s \sim d^{\pi_\text{old}}, a \sim \pi_{\theta_\text{old}}}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_\text{old}}(a|s)} \nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\text{old}}(s,a)\right].
$$

这里只出现了**单步比率** $r_t(\theta) = \pi_\theta(a_t|s_t)/\pi_{\theta_\text{old}}(a_t|s_t)$，避免了轨迹比率的指数方差。

但这引入了一个关键近似：我们用 $d^{\pi_\text{old}}$ 替代了 $d^{\pi_\theta}$。当 $\theta$ 偏离 $\theta_\text{old}$ 较远时，两个状态分布的差异不可忽视。这正是**性能差异引理**要解决的问题——它精确刻画了状态分布失配带来的误差。

<a name="s5-4"></a>

## 5.4 性能差异引理

<a name="s5-4-1"></a>

### 5.4.1 引理陈述

**性能差异引理**（Kakade & Langford, 2002）给出了任意两个策略 $\pi$ 和 $\pi'$ 之间期望回报差的精确表达式：

$$
J(\pi') - J(\pi) = \mathbb{E}_{s \sim d^{\pi'}}\left[\sum_{a} \pi'(a|s) A^\pi(s,a)\right],
$$

其中 $d^{\pi'}$ 是新策略 $\pi'$ 的折扣加权状态访问频率，$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ 是旧策略 $\pi$ 的优势函数。

<a name="s5-4-2"></a>

### 5.4.2 完整推导

我们从回报的定义出发。记 $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$ 为从时刻 $t$ 开始的折扣回报。对于策略 $\pi'$ 产生的轨迹，有：

$$
V^{\pi'}(s_0) = \mathbb{E}_{\tau \sim \pi'}[G_0 | s_0].
$$

考虑沿 $\pi'$ 轨迹的 telescoping 分解。注意到：

$$
V^{\pi'}(s_t) = \mathbb{E}_{\tau \sim \pi'}[G_t | s_t],
$$

因此 $G_t - V^{\pi'}(s_t)$ 是一个均值为零的鞅差序列。但更有用的是将 $V^\pi$ 而非 $V^{\pi'}$ 插入 telescoping：

$$
V^{\pi'}(s_0) - V^\pi(s_0) = \mathbb{E}_{\tau \sim \pi'}\left[\sum_{t=0}^{\infty} \gamma^t \left(r_t + \gamma V^\pi(s_{t+1}) - V^\pi(s_t)\right)\right].
$$

**验证此式**：展开右边求和，得到

$$
\mathbb{E}_{\tau \sim \pi'}\left[\sum_{t=0}^{\infty} \gamma^t r_t + \sum_{t=0}^{\infty} \gamma^{t+1} V^\pi(s_{t+1}) - \sum_{t=0}^{\infty} \gamma^t V^\pi(s_t)\right].
$$

后两项构成 telescoping 级数：$\sum_{t=0}^{\infty} \gamma^{t+1} V^\pi(s_{t+1}) - \sum_{t=0}^{\infty} \gamma^t V^\pi(s_t) = -V^\pi(s_0)$（因为 $\lim_{T\to\infty} \gamma^T V^\pi(s_T) = 0$，假定 $\gamma < 1$ 且 $V^\pi$ 有界）。因此右边等于 $\mathbb{E}_{\tau \sim \pi'}[G_0] - V^\pi(s_0) = V^{\pi'}(s_0) - V^\pi(s_0)$，恒等式成立。

接下来，注意到括号内的表达式正是 TD 误差 $\delta_t = r_t + \gamma V^\pi(s_{t+1}) - V^\pi(s_t)$，而由第 2 章结论，$\mathbb{E}[\delta_t | s_t, a_t] = A^\pi(s_t, a_t)$。因此：

$$
V^{\pi'}(s_0) - V^\pi(s_0) = \mathbb{E}_{\tau \sim \pi'}\left[\sum_{t=0}^{\infty} \gamma^t A^\pi(s_t, a_t)\right].
$$

对初始状态 $s_0 \sim \rho_0$ 取期望，并将轨迹期望改写为状态-动作边际分布的形式：

$$
J(\pi') - J(\pi) = \sum_{t=0}^{\infty} \gamma^t \mathbb{E}_{s_t \sim d_t^{\pi'}, a_t \sim \pi'}\left[A^\pi(s_t, a_t)\right] = \mathbb{E}_{s \sim d^{\pi'}}\left[\sum_a \pi'(a|s) A^\pi(s,a)\right],
$$

其中 $d^{\pi'}(s) = (1-\gamma)\sum_{t=0}^{\infty} \gamma^t d_t^{\pi'}(s)$ 为折扣加权状态分布（归一化为概率分布）。引理得证。

<a name="s5-4-3"></a>

### 5.4.3 关键观察

引理右侧使用的是 **$d^{\pi'}$（新策略的状态分布）** 而非 $d^\pi$。这意味着要精确计算性能差异，我们需要知道新策略会访问哪些状态——而这恰恰是我们尚未执行的策略。这是 off-policy 评估的根本困难，也是下一节推导单调改进界的出发点。

<a name="s5-5"></a>

## 5.5 TRPO 的单调改进界

本节是本章核心，遵循 Schulman et al. (2015) 的推导路径。

<a name="s5-5-1"></a>

### 5.5.1 第一步：将 $d^{\pi'}$ 替换为 $d^\pi$ 并量化误差

由性能差异引理：

$$
J(\pi') - J(\pi) = \mathbb{E}_{s \sim d^{\pi'}}\left[\sum_a \pi'(a|s) A^\pi(s,a)\right].
$$

我们希望将其中的 $d^{\pi'}$ 替换为可计算的 $d^\pi$。为此引入替代目标：

$$
\eta(\pi') = \mathbb{E}_{s \sim d^{\pi}}\left[\sum_a \pi'(a|s) A^\pi(s,a)\right].
$$

注意 $\eta(\pi')$ 只涉及旧策略的状态分布 $d^\pi$，可以用旧策略采集的样本通过重要性采样估计。定义近似目标函数：

$$
L_\pi(\pi') = J(\pi) + \eta(\pi').
$$

现在量化 $J(\pi') - L_\pi(\pi')$ 的误差。由性能差异引理和 $\eta$ 的定义：

$$
J(\pi') - L_\pi(\pi') = \mathbb{E}_{s \sim d^{\pi'}}\left[\sum_a \pi'(a|s) A^\pi(s,a)\right] - \mathbb{E}_{s \sim d^{\pi}}\left[\sum_a \pi'(a|s) A^\pi(s,a)\right].
$$

记 $f(s) = \sum_a \pi'(a|s) A^\pi(s,a)$，则误差为 $(d^{\pi'} - d^\pi)^T f$。Schulman et al. (2015) 证明了以下关键不等式：

$$
\left|J(\pi') - L_\pi(\pi')\right| \leq C \cdot \max_s D_\text{KL}\left(\pi'(\cdot|s) \| \pi(\cdot|s)\right),
$$

其中 $C = \frac{4\varepsilon\gamma}{(1-\gamma)^2}$，$\varepsilon = \max_{s,a} |A^\pi(s,a)|$。

**$C$ 的来历**：该常数来自将状态分布差异 $\|d^{\pi'} - d^\pi\|_1$ 用逐状态 KL 散度控制的推导链。具体地，(i) 由 Pinsker 不等式，$\|d^{\pi'} - d^\pi\|_1 \leq \sqrt{2 D_\text{KL}(d^{\pi'} \| d^\pi)}$；(ii) 折扣 MDP 中状态分布的 KL 散度可以被逐状态策略 KL 的加权和所界定，权重涉及 $\gamma/(1-\gamma)$；(iii) 结合 $|f(s)| \leq \max_a |A^\pi(s,a)| \leq \varepsilon$ 以及额外的 $\gamma/(1-\gamma)$ 因子（来自双重求和中折扣因子的累积），最终得到 $C = 4\varepsilon\gamma/(1-\gamma)^2$。完整的常数推导较为冗长，核心直觉是：$\gamma$ 接近 1 时，策略微小差异经过长时间累积会被放大，$(1-\gamma)^{-2}$ 反映了这种累积效应的二阶性质。

<a name="s5-5-2"></a>

### 5.5.2 第二步：单调改进下界

由上述误差界，直接得到：

$$
J(\pi') \geq L_\pi(\pi') - C \cdot \max_s D_\text{KL}\left(\pi'(\cdot|s) \| \pi(\cdot|s)\right).
$$

记右端为 $\underline{J}_\pi(\pi')$，它是 $J(\pi')$ 的一个**可计算下界**。

<a name="s5-5-3"></a>

### 5.5.3 第三步：下界的关键性质与信任域的含义

该下界具有两个至关重要的性质：

**性质 1（紧性）**：当 $\pi' = \pi$ 时，$\eta(\pi) = \mathbb{E}_{s \sim d^\pi}[\sum_a \pi(a|s) A^\pi(s,a)] = \mathbb{E}_{s \sim d^\pi}[0] = 0$（因为 $\mathbb{E}_{a \sim \pi}[A^\pi(s,a)] = 0$，见第 2 章），故 $L_\pi(\pi) = J(\pi)$，且 $\max_s D_\text{KL}(\pi\|\pi) = 0$。因此 $\underline{J}_\pi(\pi) = J(\pi)$，下界在 $\pi'=\pi$ 处**取等号**。

**性质 2（梯度一致性）**：在 $\pi' = \pi$ 处，$\nabla_{\pi'} L_\pi(\pi')|_{\pi'=\pi} = \nabla_{\pi'} \eta(\pi')|_{\pi'=\pi} = \nabla J(\pi)$（因为 $\nabla J(\pi)$ 的策略梯度形式恰好等于 $\nabla \eta(\pi')|_{\pi'=\pi}$，见第 3 章策略梯度定理）。因此下界函数与真实目标在当前策略处不仅值相同，梯度也相同。

这两个性质的推论是：**只要我们能保证 $\max_s D_\text{KL}(\pi'\|\pi)$ 足够小，使得 $\underline{J}_\pi(\pi') > J(\pi)$，就必然有 $J(\pi') > J(\pi)$**。换言之，在 KL 球约束内优化替代目标 $L_\pi$ 就能保证真实性能单调不降。这个 KL 球就是所谓的**信任域（trust region）**——在该区域内，我们对替代目标的优化是「可信的」。

<a name="s5-6"></a>

## 5.6 TRPO 的实际求解形式

<a name="s5-6-1"></a>

### 5.6.1 约束优化问题

将上述理论转化为实际算法，TRPO 求解如下约束优化：

$$
\max_\theta \quad \mathbb{E}_{s \sim d^{\pi_{\theta_\text{old}}}, a \sim \pi_{\theta_\text{old}}}\left[r_t(\theta) \cdot \hat{A}_t\right] \quad \text{s.t.} \quad \mathbb{E}_{s \sim d^{\pi_{\theta_\text{old}}}}\left[D_\text{KL}\left(\pi_{\theta_\text{old}}(\cdot|s) \| \pi_\theta(\cdot|s)\right)\right] \leq \delta,
$$

其中 $r_t(\theta) = \pi_\theta(a_t|s_t)/\pi_{\theta_\text{old}}(a_t|s_t)$ 为单步重要性比，$\hat{A}_t$ 为 GAE 优势估计（见第 4 章），$\delta$ 为预设的 KL 阈值（通常取 0.01）。

<a name="s5-6-2"></a>

### 5.6.2 从 max KL 到平均 KL 的松弛

理论下界使用的是 $\max_s D_\text{KL}$，而实践中使用的是关于 $d^{\pi_{\theta_\text{old}}}$ 的**平均 KL**。这是一个重要的松弛：平均 KL 约束不再保证 $\max_s$ KL 有界，因此**单调改进的下界在松弛后不再严格成立**。这是 TRPO 理论与实践之间的一个已知缺口。实践中，平均 KL 约束仍然有效，因为极端大的逐状态 KL 通常也会推高平均 KL；但在 pathological 情况下，理论上不能排除违反单调性的可能。

<a name="s5-7"></a>

## 5.7 自然梯度与 Fisher 信息矩阵

<a name="s5-7-1"></a>

### 5.7.1 参数空间 vs. 分布空间

标准梯度下降在参数空间中以欧氏度量 $\|\Delta\theta\|^2$ 衡量步长。但同一个 $\Delta\theta$ 在不同参数化下引起的分布变化完全不同。例如，对于 Bernoulli 分布 $\text{Ber}(p)$，参数 $p=0.5$ 附近 $\Delta p = 0.01$ 引起的分布变化很小，但在 $p=0.01$ 附近同样的 $\Delta p = 0.01$ 意味着分布的剧烈改变（从几乎确定输出 0 变为有一定概率输出 1）。我们需要一种**参数化无关**的距离度量，而 KL 散度恰好提供了这样的度量。

<a name="s5-7-2"></a>

### 5.7.2 KL 散度的二阶展开与 Fisher 矩阵

考虑 $D_\text{KL}(\pi_\theta \| \pi_{\theta+\Delta\theta})$ 关于 $\Delta\theta$ 的泰勒展开。由第 1 章结论：

- **零阶项**：$D_\text{KL}(\pi_\theta \| \pi_\theta) = 0$。
- **一阶项**：$\nabla_{\Delta\theta} D_\text{KL}(\pi_\theta \| \pi_{\theta+\Delta\theta})|_{\Delta\theta=0} = -\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s)] = 0$，因为 score function 的期望为零（见第 1 章似然比技巧部分）。
- **二阶项**：Hessian 为 $\nabla^2_{\Delta\theta} D_\text{KL}(\pi_\theta \| \pi_{\theta+\Delta\theta})|_{\Delta\theta=0} = \mathbb{E}_{a \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)^\top\right] = F(\theta)$，

其中 $F(\theta)$ 即为 **Fisher 信息矩阵**。因此：

$$
D_\text{KL}(\pi_\theta \| \pi_{\theta+\Delta\theta}) \approx \frac{1}{2} \Delta\theta^\top F(\theta) \Delta\theta.
$$

这说明 Fisher 矩阵定义了参数空间中 KL 散度的**局部二次近似**，即以 Fisher 度量的二次型来衡量分布间的距离。

<a name="s5-7-3"></a>

### 5.7.3 自然梯度的变分推导

现在我们求解如下约束优化：在 KL 球内找到使目标增长最快的方向：

$$
\max_{\Delta\theta} \quad g^\top \Delta\theta \quad \text{s.t.} \quad \frac{1}{2} \Delta\theta^\top F \Delta\theta \leq \delta,
$$

其中 $g = \nabla_\theta J(\theta)$ 为标准策略梯度。构造拉格朗日函数：

$$
\mathcal{L}(\Delta\theta, \lambda) = g^\top \Delta\theta - \lambda \left(\frac{1}{2} \Delta\theta^\top F \Delta\theta - \delta\right).
$$

对 $\Delta\theta$ 求导令其为零：$g - \lambda F \Delta\theta = 0$，解得 $\Delta\theta = \frac{1}{\lambda} F^{-1} g$。代入约束的等式边界 $\frac{1}{2}\Delta\theta^\top F \Delta\theta = \delta$ 可确定 $\lambda$，但方向已经确定：

$$
\Delta\theta \propto F^{-1} g.
$$

这就是**自然梯度（natural gradient）**：$\tilde{\nabla} J(\theta) = F(\theta)^{-1} \nabla J(\theta)$。它是在 KL 约束下使目标增长最快的方向，且与参数化选择无关（Fisher 矩阵在重参数化下协变变换，使得 $F^{-1}g$ 的方向不变）。

<a name="s5-7-4"></a>

### 5.7.4 常用策略的解析 KL 公式

以下两个公式在第 6、7 章的实现中直接使用。

**对角 Gaussian 策略**：设 $\pi = \mathcal{N}(\mu, \text{diag}(\sigma^2))$，$\pi' = \mathcal{N}(\mu', \text{diag}(\sigma'^2))$，维度为 $d$，则：

$$
D_\text{KL}(\pi \| \pi') = \sum_{i=1}^{d} \left[\log\frac{\sigma'_i}{\sigma_i} + \frac{\sigma_i^2 + (\mu_i - \mu'_i)^2}{2\sigma'^2_i} - \frac{1}{2}\right].
$$

**Categorical 策略**：设 $\pi(a) = p_a$，$\pi'(a) = q_a$，则：

$$
D_\text{KL}(\pi \| \pi') = \sum_a p_a \log \frac{p_a}{q_a}.
$$

这两个公式均可直接微分以计算 KL 约束的梯度与 Hessian-vector product。

<a name="s5-8"></a>

## 5.8 共轭梯度法求解自然梯度

<a name="s5-8-1"></a>

### 5.8.1 为什么不能直接求逆

Fisher 矩阵 $F$ 的大小为 $n \times n$，其中 $n$ 是策略网络的参数量（通常 $10^5 \sim 10^7$）。显式构造和存储 $F$ 需要 $O(n^2)$ 内存，求逆需要 $O(n^3)$ 计算，完全不可行。

<a name="s5-8-2"></a>

### 5.8.2 共轭梯度法的适用性

共轭梯度法（Conjugate Gradient, CG）是求解线性系统 $Fx = g$ 的迭代方法，其关键优势在于**只需要矩阵-向量乘积 $Fv$**，而不需要显式构造 $F$。对于 $n$ 维问题，CG 最多迭代 $n$ 次即可得到精确解；实践中由于 $F$ 的有效秩远小于 $n$，通常 $10 \sim 20$ 次迭代即可获得足够精度的近似解。

<a name="s5-8-3"></a>

### 5.8.3 Fisher-向量乘积的高效计算

$Fv$ 的计算利用以下恒等式。记 $g(\theta) = \nabla_\theta J(\theta)$，则：

$$
F(\theta) v = \nabla_\theta \left(g(\theta)^\top v\right),
$$

这是因为 $F = \mathbb{E}[\nabla \log \pi \nabla \log \pi^\top]$ 且 $\nabla(g^\top v) = \nabla(\mathbb{E}[\cdots]^\top v) = \mathbb{E}[\nabla(\nabla \log \pi^\top v)] = Fv$（在 $v$ 与 $\theta$ 无关时）。实际操作中：先计算标量 $g^\top v$（一次前向+反向传播），再对该标量关于 $\theta$ 求梯度（又一次反向传播），总代价约为两次梯度计算，远低于构造 $F$。

另一种方法是 Hutchinson 估计：$Fv = \mathbb{E}_z[(z^\top F v) z]$ 其中 $z$ 为 Rademacher 随机向量，但实践中上述确定性方法更常用。

<a name="s5-8-4"></a>

### 5.8.4 线搜索保证 KL 约束

CG 给出自然梯度方向 $\Delta\theta = F^{-1}g$ 后，还需确定步长 $\alpha$。TRPO 使用回溯线搜索：从 $\alpha = 1$ 开始，逐步减半直到满足 KL 约束 $\bar{D}_\text{KL}(\pi_{\theta+\alpha\Delta\theta} \| \pi_\theta) \leq \delta$ 且目标函数有改善。这保证了每次更新都在信任域内。

<a name="s5-9"></a>

## 5.9 TRPO 的实践代价与 PPO 的动机

TRPO 虽然在理论上优美，但实践中存在显著代价：

1. **实现复杂**：需要手动实现 CG 求解器、线搜索、Fisher-向量乘积，代码量大且容易出错。
2. **计算开销**：每步更新需要多次梯度计算（CG 迭代 × 2 + 线搜索评估），吞吐量低于纯一阶方法。
3. **架构兼容性差**：CG 要求目标函数对参数二阶可微且 Fisher 矩阵正定，这与 dropout、batch normalization、参数共享、辅助损失等现代深度学习技术难以兼容。

PPO 的设计目标正是用**一阶优化方法**（SGD/Adam）达到「近似信任域」的效果，同时避免 TRPO 的上述缺点。PPO 提出了两条路线：

- **PPO-Penalty**：将 KL 约束转化为自适应惩罚项加入目标函数，通过调节惩罚系数间接控制 KL。
- **PPO-Clip**：直接截断重要性比 $r_t(\theta)$，使其落在 $[1-\epsilon, 1+\epsilon]$ 区间内，从根本上阻止过大的策略更新。

这两条路线的具体公式、损失函数设计与超参数机理将在第 6 章详细展开。本章建立的数学框架——重要性采样、性能差异引理、单调改进界、自然梯度的几何意义——构成了理解 PPO 设计选择的理论基础。

<a name="s5-10"></a>

## 5.10 本章小结与自测

<a name="h184"></a>

### 要点回顾

1. **重要性采样**提供了用旧分布样本估计新分布期望的无偏方法，但方差可能爆炸；有效样本量 ESS 是关键诊断指标。
2. **策略层面只用单步比率**而非轨迹比率，因为策略梯度定理已对状态分布取期望；代价是引入了状态分布失配误差。
3. **性能差异引理**精确表达了 $J(\pi') - J(\pi)$ 为新策略状态下优势函数的期望，揭示了 off-policy 评估的核心困难在于 $d^{\pi'}$ 未知。
4. **TRPO 单调改进界**通过将 $d^{\pi'}$ 替换为 $d^\pi$ 并用 KL 散度控制误差，建立了可计算的替代目标与真实性能之间的下界关系；在 $\pi'=\pi$ 处下界紧且梯度一致，保证了信任域内优化的可靠性。
5. **自然梯度** $F^{-1}\nabla J$ 是 KL 约束下的最优更新方向，通过共轭梯度法仅需 Fisher-向量乘积即可高效求解；但 TRPO 的实现复杂性催生了 PPO。

<a name="h185"></a>

### 自测题

**题目 1**：为什么 IS 估计量要求行为分布 $p$ 的支撑覆盖目标分布 $q$ 的支撑？若不满足，估计量是有偏还是无偏？

<details>
<summary>查看参考答案</summary>

参考答案：若存在 $q(x)>0$ 但 $p(x)=0$ 的区域，该区域对 $\mathbb{E}_q[f]$ 的贡献永远无法被采样到，IS 估计量系统性低估真实期望，产生偏差。

</details>

**题目 2**：在策略梯度中使用单步重要性比而非轨迹比的根本原因是什么？单步比率忽略了什么误差来源？

<details>
<summary>查看参考答案</summary>

参考答案：轨迹比的方差随 horizon 指数增长。单步比率之所以成立是因为策略梯度定理已对状态分布取期望；忽略的误差来源是新旧策略状态分布 $d^{\pi_\theta}$ 与 $d^{\pi_{\theta_\text{old}}}$ 的差异。

</details>

**题目 3**：性能差异引理 $J(\pi')-J(\pi) = \mathbb{E}_{s\sim d^{\pi'}}[\sum_a \pi'(a|s)A^\pi(s,a)]$ 中为什么出现的是 $d^{\pi'}$ 而非 $d^\pi$？这对算法设计有什么影响？

<details>
<summary>查看参考答案</summary>

参考答案：推导中 telescoping 分解沿 $\pi'$ 轨迹进行，状态自然服从 $d^{\pi'}$。这使得精确计算性能差需要知道新策略的状态分布，导致无法直接用旧数据评估；TRPO 通过 KL 约束控制 $d^{\pi'}$ 与 $d^\pi$ 的差异来解决此问题。

</details>

**题目 4**：证明自然梯度方向 $F^{-1}g$ 是约束优化 $\max_{\Delta\theta} g^\top\Delta\theta$ s.t. $\frac{1}{2}\Delta\theta^\top F\Delta\theta \leq \delta$ 的解。

<details>
<summary>查看参考答案</summary>

参考答案：拉格朗日函数 $\mathcal{L}=g^\top\Delta\theta-\lambda(\frac{1}{2}\Delta\theta^\top F\Delta\theta-\delta)$，对 $\Delta\theta$ 求导得 $g-\lambda F\Delta\theta=0$，故 $\Delta\theta=\lambda^{-1}F^{-1}g$，方向为 $F^{-1}g$。

</details>

**题目 5**：TRPO 实践中将 $\max_s D_\text{KL}$ 松弛为平均 KL 后，单调改进保证是否仍然严格成立？为什么？

<details>
<summary>查看参考答案</summary>

参考答案：不再严格成立。平均 KL 有界不能推出每个状态的 KL 都有界，可能存在某些状态的 KL 远超平均值，使理论下界失效。实践中该松弛通常有效，但理论上存在缺口。

</details>

---

---

[上一章](chapter-04.md) · [返回目录](../README.md) · [阅读路线与符号表](README.md) · [下一章](chapter-06.md)
