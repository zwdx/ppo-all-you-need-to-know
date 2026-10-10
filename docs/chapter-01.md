[返回目录](../README.md) · [阅读路线与符号表](README.md) · [下一章](chapter-02.md)

---

<a id="chapter-1"></a>

# 第 1 章 数学基础

**本章目录**

- [1.1 随机变量与概率分布](#s1-1)
- [1.2 期望、方差与条件期望](#s1-2)
- [1.3 采样估计与蒙特卡洛方法](#s1-3)
- [1.4 信息论基础](#s1-4)
- [1.5 Jensen 不等式与常用不等式](#s1-5)
- [1.6 优化基础](#s1-6)
- [1.7 随机梯度下降与 Adam 优化器](#s1-7)
- [1.8 似然比技巧（Likelihood Ratio Trick）](#s1-8)
- [1.9 本章小结与自测](#s1-9)

本章为后续 PPO（Proximal Policy Optimization，近端策略优化）算法的推导建立必要的数学工具。我们不追求泛泛的数学复习，而是精准地铺设 PPO 及其前置理论（策略梯度、TRPO、自然梯度）所依赖的每一块基石。每个定义和定理都会说明它在强化学习中的具体用途，避免"学了不知道干什么"的感觉。

**记号约定**：标量用小写字母 $`x, \theta`$；向量用粗体小写 $`\boldsymbol{x}, \boldsymbol{\mu}`$；矩阵用粗体大写 $`\boldsymbol{H}, \boldsymbol{F}`$。随机变量与其取值用同一符号，通过上下文区分。$`\nabla_{\boldsymbol{\theta}}`$ 表示对参数向量 $`\boldsymbol{\theta}`$ 的梯度算子，结果为列向量。$`\mathbb{E}[\cdot]`$ 与 $`\mathrm{Var}(\cdot)`$ 分别表示期望与方差，下标标明分布时写作 $`\mathbb{E}_{x \sim p}[f(x)]`$。$`\log`$ 一律指自然对数 $`\ln`$。

<a id="s1-1"></a>

## 1.1 随机变量与概率分布

<a id="s1-1-1"></a>

### 1.1.1 概率质量函数与概率密度函数

离散随机变量 $`X`$ 取值为 $`\{x_1, x_2, \dots\}`$，其概率质量函数（probability mass function, PMF）定义为 $`p(x_i) = P(X = x_i)`$，满足 $`p(x_i) \geq 0`$ 且 $`\sum_i p(x_i) = 1`$。

连续随机变量 $`X`$ 的概率密度函数（probability density function, PDF）$`p(x)`$ 满足 $`p(x) \geq 0`$ 且 $`\int_{-\infty}^{\infty} p(x)\,dx = 1`$，事件概率由积分给出：$`P(a \leq X \leq b) = \int_a^b p(x)\,dx`$。

在强化学习中，策略 $`\pi_{\boldsymbol{\theta}}(a|s)`$ 就是一个以状态 $`s`$ 为条件、以动作 $`a`$ 为变量的概率分布，参数化为 $`\boldsymbol{\theta}`$。当动作空间离散时用 PMF，连续时用 PDF。理解这两类分布的参数化形式及其梯度，是后续策略梯度推导的前提。

<a id="s1-1-2"></a>

### 1.1.2 联合分布、条件分布与边缘化

对于两个随机变量 $`X, Y`$，联合分布 $`p(x, y)`$ 描述它们同时取特定值的概率。条件分布定义为：

$$
p(y|x) = \frac{p(x, y)}{p(x)}, \quad \text{当 } p(x) > 0
$$

边缘分布通过对另一个变量求和（离散）或积分（连续）得到：

$$
p(x) = \sum_y p(x, y) \quad \text{或} \quad p(x) = \int p(x, y)\,dy
$$

这三个概念的关系即贝叶斯公式的基础：$`p(x, y) = p(y|x)\,p(x) = p(x|y)\,p(y)`$。在 RL 中，轨迹的概率分解正是反复应用这一链式法则的结果——一条轨迹 $`(s_0, a_0, s_1, a_1, \dots)`$ 的联合概率被拆分为初始状态分布、策略条件和环境转移条件的乘积。

<a id="s1-1-3"></a>

### 1.1.3 Categorical 分布：离散动作空间的策略

Categorical 分布（也称多项分布的单次试验特例）是离散动作空间中最常用的策略分布。设动作空间大小为 $`K`$，策略网络输出一个 $`K`$ 维向量 $`\boldsymbol{z} = (z_1, \dots, z_K)`$，称为 logits（未归一化的对数分数）。通过 softmax 函数将其转化为合法概率：

$$
\pi(a=k|\boldsymbol{z}) = \frac{\exp(z_k)}{\sum_{j=1}^{K} \exp(z_j)}, \quad k = 1, \dots, K
$$

**为什么用 logits 而非直接输出概率？** 因为 softmax 将无约束的 $`\mathbb{R}^K`$ 映射到概率单纯形，使网络输出无需额外约束即可保证合法性。更重要的是，softmax 关于 logits 的梯度有简洁形式。记 $`\pi_k = \pi(a=k|\boldsymbol{z})`$，则：

$$
\frac{\partial \pi_k}{\partial z_j} = \pi_k (\delta_{kj} - \pi_j)
$$

其中 $`\delta_{kj}`$ 是 Kronecker delta。这个结果在对策略梯度做反向传播时直接出现。此外，$`\log \pi_k = z_k - \log \sum_j \exp(z_j)`$，其对 $`z_j`$ 的梯度为：

$$
\frac{\partial \log \pi_k}{\partial z_j} = \delta_{kj} - \pi_j
$$

这正是策略梯度定理中 $`\nabla_{\boldsymbol{\theta}} \log \pi(a|s)`$ 的具体形式（见第 3 章）。注意该梯度只依赖于 $`\pi`$ 本身，不显式涉及 $`z`$，数值稳定性好。

<a id="s1-1-4"></a>

### 1.1.4 对角 Gaussian 分布：连续动作空间的策略

当动作空间为 $`\mathbb{R}^d`$ 时，最常用的策略分布是对角多元高斯分布（diagonal multivariate Gaussian）。其 PDF 为：

$$
\pi(\boldsymbol{a}|\boldsymbol{s}) = \prod_{i=1}^{d} \frac{1}{\sqrt{2\pi}\sigma_i} \exp\!\left(-\frac{(a_i - \mu_i)^2}{2\sigma_i^2}\right)
$$

其中均值向量 $`\boldsymbol{\mu} = (\mu_1, \dots, \mu_d)`$ 和对数标准差向量 $`\log \boldsymbol{\sigma} = (\log \sigma_1, \dots, \log \sigma_d)`$ 均由策略网络输出。

**为什么参数化 $`\log \sigma`$ 而非 $`\sigma`$？** 标准差必须为正，而神经网络输出是无约束实数。若直接输出 $`\sigma`$，需加 softplus 或 exp 变换来保证正性，这会引入额外的非线性并可能导致梯度消失。使用 $`\log \sigma`$ 作为参数，$`\sigma = \exp(\log \sigma)`$ 自动为正，且对数似然关于 $`\log \sigma`$ 的梯度形式更干净：

$$
\log \pi(\boldsymbol{a}|\boldsymbol{s}) = -\frac{d}{2}\log(2\pi) - \sum_{i=1}^{d} \log \sigma_i - \sum_{i=1}^{d} \frac{(a_i - \mu_i)^2}{2\sigma_i^2}
$$

对 $`\mu_i`$ 求导得 $`\frac{\partial \log \pi}{\partial \mu_i} = \frac{a_i - \mu_i}{\sigma_i^2}`$，对 $`\log \sigma_i`$ 求导得 $`\frac{\partial \log \pi}{\partial \log \sigma_i} = -1 + \frac{(a_i - \mu_i)^2}{\sigma_i^2}`$。这些表达式在实现连续动作空间的策略梯度时直接使用。

**为什么假设对角协方差？** 完整协方差矩阵有 $`O(d^2)`$ 个参数，且需保证正定性，计算和存储代价高。对角假设意味着各动作维度独立，虽然牺牲了建模能力，但大幅简化了采样和梯度计算。实践中这一假设通常足够，因为策略网络的隐层已经捕获了维度间的相关性。

<a id="s1-2"></a>

## 1.2 期望、方差与条件期望

<a id="s1-2-1"></a>

### 1.2.1 期望的定义与线性性

离散情形：$`\mathbb{E}[f(X)] = \sum_x f(x)\,p(x)`$。连续情形：$`\mathbb{E}[f(X)] = \int f(x)\,p(x)\,dx`$。

期望的核心性质是**线性性**：对任意常数 $`a, b`$ 和函数 $`f, g`$，

$$
\mathbb{E}[a f(X) + b g(Y)] = a\,\mathbb{E}[f(X)] + b\,\mathbb{E}[g(Y)]
$$

这一性质不要求 $`X, Y`$ 独立，是后续所有期望运算的基础。特别地，在策略梯度推导中，我们需要将 $`\mathbb{E}[\sum_t \nabla \log \pi \cdot r_t]`$ 拆分为各项之和，线性性是这一步成立的唯一依据。

<a id="s1-2-2"></a>

### 1.2.2 方差与协方差

方差定义为 $`\mathrm{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2`$。

**方差的可加性**仅在变量不相关时成立：若 $`\mathrm{Cov}(X_i, X_j) = 0`$ 对所有 $`i \neq j`$，则 $`\mathrm{Var}(\sum_i X_i) = \sum_i \mathrm{Var}(X_i)`$。一般情况下：

$$
\mathrm{Var}\!\left(\sum_{i=1}^{n} X_i\right) = \sum_{i=1}^{n} \mathrm{Var}(X_i) + 2\sum_{i<j} \mathrm{Cov}(X_i, X_j)
$$

在 RL 中，回报 $`G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}`$ 的各项之间高度相关（因为共享未来的随机性），因此回报的方差远大于各项方差之和。这是理解「为什么回报估计噪声大」以及「为什么需要基线函数减方差」（见第 3 章）的关键背景。

<a id="s1-2-3"></a>

### 1.2.3 条件期望与塔性质

条件期望 $`\mathbb{E}[Y|X=x] = \sum_y y\,p(y|x)`$（离散）或对应积分形式。它是一个关于 $`x`$ 的函数，因此 $`\mathbb{E}[Y|X]`$ 本身是一个随机变量。

**塔性质（Law of Total Expectation）**：

$$
\mathbb{E}[Y] = \mathbb{E}_X[\mathbb{E}[Y|X]]
$$

即先对 $`Y`$ 在给定 $`X`$ 下求条件期望，再对 $`X`$ 求边际期望，等于直接对 $`Y`$ 求期望。这个性质在 RL 中无处不在：贝尔曼方程本质上就是塔性质的应用——将未来回报的期望分解为当前奖励加上下一状态价值的期望（见第 2 章）。

**条件方差的分解**（Law of Total Variance）也值得记住：

$$
\mathrm{Var}(Y) = \mathbb{E}_X[\mathrm{Var}(Y|X)] + \mathrm{Var}_X(\mathbb{E}[Y|X])
$$

它说明总方差可以分解为「组内方差」和「组间方差」。在分析 TD 学习与蒙特卡洛方法的方差差异时，这个分解提供了精确的解释框架。

<a id="s1-3"></a>

## 1.3 采样估计与蒙特卡洛方法

<a id="s1-3-1"></a>

### 1.3.1 样本均值估计

设 $`X_1, \dots, X_N \overset{\text{i.i.d.}}{\sim} p`$，用样本均值 $`\hat{\mu}_N = \frac{1}{N}\sum_{i=1}^{N} X_i`$ 估计 $`\mu = \mathbb{E}[X]`$。

**无偏性**：$`\mathbb{E}[\hat{\mu}_N] = \frac{1}{N}\sum_{i=1}^{N}\mathbb{E}[X_i] = \frac{1}{N} \cdot N\mu = \mu`$。第二步用了期望的线性性和同分布假设。

**方差衰减**：由于独立性，$`\mathrm{Var}(\hat{\mu}_N) = \frac{1}{N^2}\sum_{i=1}^{N}\mathrm{Var}(X_i) = \frac{\sigma^2}{N}`$，其中 $`\sigma^2 = \mathrm{Var}(X)`$。标准差以 $`O(1/\sqrt{N})`$ 衰减。

这就是「为什么要跑很多 rollout」的数学根源：要将估计精度提高 10 倍，样本量需增加 100 倍。RL 中每次 rollout 成本高昂（需要与环境交互），因此降低方差（通过基线函数、优势估计等）比单纯增加样本量更有效率。这也是 PPO 采用 mini-batch 更新而非单次全量估计的直接动机之一。

<a id="s1-3-2"></a>

### 1.3.2 从期望到采样的桥梁

策略梯度的核心困难在于：目标函数 $`J(\boldsymbol{\theta}) = \mathbb{E}_{\tau \sim p_{\boldsymbol{\theta}}}[R(\tau)]`$ 的期望是对 $`\boldsymbol{\theta}`$ 相关的分布取的，不能直接对 $`\boldsymbol{\theta}`$ 求导后交换期望与微分的顺序（因为分布本身依赖 $`\boldsymbol{\theta}`$）。似然比技巧（1.7 节）解决了这个问题，但其结果是另一个期望，仍然无法解析计算。采样估计提供了最终的数值解法：用有限条轨迹的经验平均代替理论期望。整个链条是：

$$
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\tau \sim p_{\boldsymbol{\theta}}}[R(\tau) \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(\tau)] \approx \frac{1}{N}\sum_{i=1}^{N} R(\tau_i) \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(\tau_i)
$$

左边的等号来自似然比技巧，右边的近似来自蒙特卡洛估计。两者缺一不可。

<a id="s1-4"></a>

## 1.4 信息论基础

<a id="s1-4-1"></a>

### 1.4.1 熵与最大熵性质

离散分布 $`p`$ 的香农熵（Shannon entropy）定义为：

$$
H(p) = -\sum_x p(x) \log p(x) = \mathbb{E}_{x \sim p}[-\log p(x)]
$$

连续分布对应微分熵（differential entropy），形式类似但用积分替代求和，且不再具有非负性等离散熵的部分性质。

**最大熵性质**：在所有支撑集相同的分布中，均匀分布的熵最大。对于支撑集为 $`\{1, \dots, K\}`$ 的离散分布，$`H(p) \leq \log K`$，等号当且仅当 $`p(x) = 1/K`$ 时成立。证明可用 Jensen 不等式（1.5 节）或拉格朗日乘子法。

在 RL 中，熵正则化项 $`\beta H(\pi(\cdot|s))`$ 被加入目标函数以鼓励探索：熵越大，策略越接近均匀分布，探索越充分。PPO 的实现中几乎总是包含这一项（见第 6 章）。

<a id="s1-4-2"></a>

### 1.4.2 交叉熵

分布 $`p`$ 相对于 $`q`$ 的交叉熵（cross-entropy）定义为：

$$
H(p, q) = -\sum_x p(x) \log q(x) = \mathbb{E}_{x \sim p}[-\log q(x)]
$$

注意期望是对 $`p`$ 取的，不是 $`q`$。交叉熵衡量的是「用 $`q`$ 编码服从 $`p`$ 的数据时的平均码长」。在机器学习中，分类任务的损失函数就是真实标签分布与模型预测分布之间的交叉熵。

<a id="s1-4-3"></a>

### 1.4.3 KL 散度：定义与非负性证明

KL 散度（Kullback-Leibler divergence）定义为交叉熵减去熵：

$$
D_{\mathrm{KL}}(p \| q) = H(p, q) - H(p) = \sum_x p(x) \log \frac{p(x)}{q(x)} = \mathbb{E}_{x \sim p}\!\left[\log \frac{p(x)}{q(x)}\right]
$$

**定理（KL 非负性 / Gibbs 不等式）**：$`D_{\mathrm{KL}}(p \| q) \geq 0`$，等号当且仅当 $`p = q`$（几乎处处）时成立。

**证明**：考虑函数 $`f(t) = -\log t`$，它是严格凸函数（$`f''(t) = 1/t^2 > 0`$ 对 $`t > 0`$）。由 Jensen 不等式（1.5 节），对凸函数 $`f`$ 有 $`f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]`$。令 $`X = q(x)/p(x)`$，期望对 $`p`$ 取：

$$
-\log \mathbb{E}_{x \sim p}\!\left[\frac{q(x)}{p(x)}\right] \leq \mathbb{E}_{x \sim p}\!\left[-\log \frac{q(x)}{p(x)}\right]
$$

左边：$`\mathbb{E}_{x \sim p}[q(x)/p(x)] = \sum_x p(x) \cdot q(x)/p(x) = \sum_x q(x) = 1`$，故左边为 $`-\log 1 = 0`$。

右边：$`\mathbb{E}_{x \sim p}[-\log(q(x)/p(x))] = \mathbb{E}_{x \sim p}[\log(p(x)/q(x))] = D_{\mathrm{KL}}(p \| q)`$。

因此 $`0 \leq D_{\mathrm{KL}}(p \| q)`$。等号成立条件是 Jensen 不等式取等号，即 $`q(x)/p(x)`$ 为常数（几乎处处），结合归一化条件得 $`p = q`$。$`\blacksquare`$

<a id="s1-4-4"></a>

### 1.4.4 KL 散度的不对称性与前向/反向 KL

KL 散度**不是**距离度量：$`D_{\mathrm{KL}}(p \| q) \neq D_{\mathrm{KL}}(q \| p)`$。这种不对称性导致两种 KL 在优化中有截然不同的行为：

- **前向 KL（forward KL）**：$`D_{\mathrm{KL}}(p \| q)`$，对 $`q`$ 最小化时，$`q`$ 倾向于覆盖 $`p`$ 的所有支撑区域（mean-seeking）。若 $`p`$ 在某处为零而 $`q`$ 不为零，该项贡献有限；但若 $`p`$ 在某处非零而 $`q`$ 趋近于零，$`\log(p/q) \to +\infty`$，惩罚极大。因此 $`q`$ 被迫在 $`p`$ 有质量的地方也分配质量。

- **反向 KL（reverse KL）**：$`D_{\mathrm{KL}}(q \| p)`$，对 $`q`$ 最小化时，$`q`$ 倾向于集中在 $`p`$ 的某个峰值上（mode-seeking）。若 $`q`$ 在某处非零而 $`p`$ 趋近于零，$`\log(q/p) \to +\infty`$，惩罚极大；但 $`q`$ 在 $`p`$ 的其他峰处为零不会受罚。因此 $`q`$ 可能忽略 $`p`$ 的部分模式。

**PPO 使用的是哪种？** PPO 的策略更新约束使用 $`D_{\mathrm{KL}}(\pi_{\boldsymbol{\theta}_{\text{old}}} \| \pi_{\boldsymbol{\theta}})`$，即以旧策略为参考、新策略为优化变量。这是前向 KL 的形式（第一个参数固定为旧策略）。选择前向 KL 的原因是：我们希望新策略不要偏离旧策略太远，同时保留探索能力（mean-seeking 行为避免新策略过早坍缩到某个确定性动作）。TRPO 的理论界同样基于这一方向的 KL（见第 5 章）。不过在实践中，PPO-Clip 变体并不直接优化 KL，而是通过裁剪比率间接控制更新幅度；KL 仅作为诊断指标或在 penalty 变体中使用（见第 6 章）。

<a id="s1-5"></a>

## 1.5 Jensen 不等式与常用不等式

<a id="s1-5-1"></a>

### 1.5.1 Jensen 不等式

**定理**：设 $`f: \mathcal{D} \to \mathbb{R}`$ 是凸函数，$`X`$ 是取值于 $`\mathcal{D}`$ 的随机变量，则：

$$
f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]
$$

若 $`f`$ 是严格凸函数，等号当且仅当 $`X`$ 为常数（几乎处处）时成立。若 $`f`$ 是凹函数，不等号反向。

凸函数的等价定义：$`f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)`$ 对所有 $`\lambda \in [0,1]`$ 成立。Jensen 不等式是该定义向一般期望的推广。

**在 RL 推导中的典型用法**：
1. 上述 KL 非负性的证明（1.4.3 节）。
2. 证据下界（ELBO）的推导：$`\log \mathbb{E}[X] \geq \mathbb{E}[\log X]`$（因为 $`\log`$ 是凹函数），这是变分推断的基础，也与策略梯度中的重要性加权估计有关。
3. TRPO 单调改进界的推导中，利用凸性将复杂的性能差异界简化为可计算的代理目标（见第 5 章）。

<a id="s1-5-2"></a>

### 1.5.2 其他常用不等式

- **Cauchy-Schwarz 不等式**：$`|\mathbb{E}[XY]| \leq \sqrt{\mathbb{E}[X^2]\,\mathbb{E}[Y^2]}`$。在分析梯度估计的方差上界时使用。
- **Markov 不等式**：$`P(X \geq a) \leq \mathbb{E}[X]/a`$（$`X \geq 0, a > 0`$）。概率界的基本工具。
- **三角不等式**：$`|a + b| \leq |a| + |b|`$。在误差累积分析中频繁出现。

<a id="s1-6"></a>

## 1.6 优化基础

<a id="s1-6-1"></a>

### 1.6.1 梯度与方向导数

标量函数 $`f: \mathbb{R}^n \to \mathbb{R}`$ 在点 $`\boldsymbol{\theta}`$ 处的梯度 $`\nabla_{\boldsymbol{\theta}} f(\boldsymbol{\theta})`$ 是一个向量，其第 $`i`$ 个分量为偏导数 $`\partial f / \partial \theta_i`$。梯度的几何意义是函数增长最快的方向。

沿单位向量 $`\boldsymbol{v}`$ 的方向导数为 $`D_{\boldsymbol{v}} f(\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}} f(\boldsymbol{\theta})^\top \boldsymbol{v}`$。当 $`\boldsymbol{v}`$ 与梯度同向时方向导数最大，这解释了梯度上升为何选择梯度方向。

<a id="s1-6-2"></a>

### 1.6.2 Hessian 矩阵与二阶泰勒展开

Hessian 矩阵 $`\boldsymbol{H}(\boldsymbol{\theta})`$ 的元素为 $`H_{ij} = \frac{\partial^2 f}{\partial \theta_i \partial \theta_j}`$。若 $`f`$ 二阶连续可微，$`\boldsymbol{H}`$ 对称。

**一阶泰勒展开**（带余项）：

$$
f(\boldsymbol{\theta} + \Delta \boldsymbol{\theta}) = f(\boldsymbol{\theta}) + \nabla f(\boldsymbol{\theta})^\top \Delta \boldsymbol{\theta} + o(\|\Delta \boldsymbol{\theta}\|)
$$

**二阶泰勒展开**（带余项）：

$$
f(\boldsymbol{\theta} + \Delta \boldsymbol{\theta}) = f(\boldsymbol{\theta}) + \nabla f(\boldsymbol{\theta})^\top \Delta \boldsymbol{\theta} + \frac{1}{2} \Delta \boldsymbol{\theta}^\top \boldsymbol{H}(\boldsymbol{\theta}) \Delta \boldsymbol{\theta} + o(\|\Delta \boldsymbol{\theta}\|^2)
$$

这里 $`o(\|\Delta \boldsymbol{\theta}\|^k)`$ 表示当 $`\|\Delta \boldsymbol{\theta}\| \to 0`$ 时比 $`\|\Delta \boldsymbol{\theta}\|^k`$ 更快趋于零的量。

**在 RL 中的关键作用**：TRPO 的核心思想是将策略更新的性能改进用二阶泰勒展开近似，其中 Hessian 被替换为 Fisher 信息矩阵（见第 5 章）。自然梯度方向 $`\boldsymbol{F}^{-1} \nabla J`$ 正是在 KL 散度的二阶近似约束下最大化线性化目标的解。没有二阶展开，就无法理解为什么自然梯度比普通梯度更合理。

<a id="s1-6-3"></a>

### 1.6.3 凸函数与局部近似

若 $`f`$ 是凸函数，则一阶泰勒展开给出全局下界：$`f(\boldsymbol{y}) \geq f(\boldsymbol{x}) + \nabla f(\boldsymbol{x})^\top (\boldsymbol{y} - \boldsymbol{x})`$。这意味着在凸优化中，局部信息足以指导全局搜索。

然而，RL 的目标函数 $`J(\boldsymbol{\theta})`$ 通常**非凸**。策略参数的微小变化可能导致策略行为的剧烈改变（尤其在深度网络参数化下）。因此，我们不能信赖全局凸性，只能在信任域（trust region）内依赖局部近似。这正是 TRPO 和 PPO 的设计哲学：限制每步更新的大小，使局部近似有效。

<a id="s1-7"></a>

## 1.7 随机梯度下降与 Adam 优化器

<a id="s1-7-1"></a>

### 1.7.1 Mini-batch 梯度的偏差与可行性

真梯度 $`\nabla J(\boldsymbol{\theta}) = \mathbb{E}_{x \sim p}[g(x)]`$ 通常无法解析计算，我们用 mini-batch 估计 $`\hat{g} = \frac{1}{B}\sum_{i=1}^{B} g(x_i)`$ 替代。若 $`x_i \overset{\text{i.i.d.}}{\sim} p`$，则 $`\mathbb{E}[\hat{g}] = \nabla J(\boldsymbol{\theta})`$，即 mini-batch 梯度是真梯度的**无偏估计**。

但在 RL 中情况更复杂：策略梯度估计 $`\hat{g} = \frac{1}{N}\sum_i R(\tau_i) \nabla \log p_{\boldsymbol{\theta}}(\tau_i)`$ 中的轨迹 $`\tau_i`$ 是由当前策略 $`\pi_{\boldsymbol{\theta}}`$ 生成的。当我们用同一批数据做多次 mini-batch 更新时（如 PPO 的多 epoch 训练），数据不再是当前参数的无偏样本——因为参数已经在更新，而数据仍是旧参数生成的。这引入了偏差。PPO 通过裁剪机制来控制这种偏差的影响（见第 6 章），这也是 PPO 被称为"近端"优化的原因之一。

<a id="s1-7-2"></a>

### 1.7.2 Adam 优化器

Adam（Adaptive Moment Estimation）结合了动量和自适应学习率。维护两个指数移动平均：

$$
\boldsymbol{m}_t = \beta_1 \boldsymbol{m}_{t-1} + (1 - \beta_1) \boldsymbol{g}_t \quad \text{（一阶矩估计，即梯度的 EMA）}
$$

$$
\boldsymbol{v}_t = \beta_2 \boldsymbol{v}_{t-1} + (1 - \beta_2) \boldsymbol{g}_t^2 \quad \text{（二阶矩估计，即梯度平方的 EMA）}
$$

偏差修正后：$`\hat{\boldsymbol{m}}_t = \boldsymbol{m}_t / (1 - \beta_1^t)`$，$`\hat{\boldsymbol{v}}_t = \boldsymbol{v}_t / (1 - \beta_2^t)`$。

参数更新：$`\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \alpha \cdot \hat{\boldsymbol{m}}_t / (\sqrt{\hat{\boldsymbol{v}}_t} + \epsilon)`$。

默认超参数：$`\beta_1 = 0.9`$，$`\beta_2 = 0.999`$，$`\epsilon = 10^{-8}`$。$`\boldsymbol{m}_t`$ 提供平滑的梯度方向（类似动量），$`\sqrt{\hat{\boldsymbol{v}}_t}`$ 对每个参数维度做自适应缩放——梯度方差大的维度学习率被缩小，方差小的维度被放大。这使得 Adam 在非凸、非平稳的 RL 优化中表现稳健。

<a id="s1-7-3"></a>

### 1.7.3 学习率与梯度裁剪的数学含义

学习率 $`\alpha`$ 控制每步更新的幅度。在泰勒展开视角下，$`\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \alpha \boldsymbol{g}_t`$ 意味着我们在一阶近似有效的范围内移动。若 $`\alpha`$ 过大，高阶项不可忽略，近似失效，训练发散。

梯度裁剪（gradient clipping）将梯度范数限制在阈值 $`c`$ 以内：$`\boldsymbol{g}_{\text{clipped}} = \boldsymbol{g} \cdot \min(1, c / \|\boldsymbol{g}\|)`$。这等价于在球约束 $`\|\Delta \boldsymbol{\theta}\| \leq \alpha c`$ 下做投影梯度步。在 RL 中，梯度裁剪防止偶尔出现的极大梯度（例如当 $`\log \pi`$ 的梯度在某些状态下异常大时）破坏训练稳定性。PPO 的 clip 操作作用于概率比率而非梯度本身，但精神一致：限制单步变化的幅度。

<a id="s1-8"></a>

## 1.8 似然比技巧（Likelihood Ratio Trick）

本节是第 1 章的核心，也是连接概率论与策略梯度方法的桥梁。

<a id="s1-8-1"></a>

### 1.8.1 基本恒等式：得分函数的期望为零

**命题**：设 $`p_{\boldsymbol{\theta}}(x)`$ 是关于参数 $`\boldsymbol{\theta}`$ 可微的概率分布族，且满足正则条件（支撑集不依赖 $`\boldsymbol{\theta}`$，可在积分号下求导），则：

$$
\mathbb{E}_{x \sim p_{\boldsymbol{\theta}}}[\nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(x)] = \boldsymbol{0}
$$

**证明**：由 $`\int p_{\boldsymbol{\theta}}(x)\,dx = 1`$ 两边对 $`\boldsymbol{\theta}`$ 求导：

$$
\nabla_{\boldsymbol{\theta}} \int p_{\boldsymbol{\theta}}(x)\,dx = \nabla_{\boldsymbol{\theta}} 1 = \boldsymbol{0}
$$

在正则条件下交换微分与积分：

$$
\int \nabla_{\boldsymbol{\theta}} p_{\boldsymbol{\theta}}(x)\,dx = \boldsymbol{0}
$$

利用恒等式 $`\nabla_{\boldsymbol{\theta}} p_{\boldsymbol{\theta}}(x) = p_{\boldsymbol{\theta}}(x) \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(x)`$（这是链式法则的直接结果：$`\nabla \log p = (\nabla p) / p`$），代入得：

$$
\int p_{\boldsymbol{\theta}}(x) \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(x)\,dx = \mathbb{E}_{x \sim p_{\boldsymbol{\theta}}}[\nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(x)] = \boldsymbol{0}
$$

$`\blacksquare`$

量 $`\nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(x)`$ 称为**得分函数（score function）**。该命题说明得分函数在自身分布下的期望为零。直觉上，$`\log p`$ 的梯度指向使概率增大的方向，但正负区域恰好抵消。

<a id="s1-8-2"></a>

### 1.8.2 似然比技巧的核心公式

**定理（似然比技巧 / Log-derivative trick）**：在同样的正则条件下，对任意与 $`\boldsymbol{\theta}`$ 无关的函数 $`f(x)`$：

$$
\nabla_{\boldsymbol{\theta}} \mathbb{E}_{x \sim p_{\boldsymbol{\theta}}}[f(x)] = \mathbb{E}_{x \sim p_{\boldsymbol{\theta}}}[f(x) \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(x)]
$$

**证明**：展开期望的定义：

$$
\nabla_{\boldsymbol{\theta}} \mathbb{E}_{x \sim p_{\boldsymbol{\theta}}}[f(x)] = \nabla_{\boldsymbol{\theta}} \int f(x) p_{\boldsymbol{\theta}}(x)\,dx
$$

$`f(x)`$ 不依赖 $`\boldsymbol{\theta}`$，故可移入梯度算子内部（再次交换微分与积分）：

$$
= \int f(x) \nabla_{\boldsymbol{\theta}} p_{\boldsymbol{\theta}}(x)\,dx
$$

代入 $`\nabla_{\boldsymbol{\theta}} p_{\boldsymbol{\theta}}(x) = p_{\boldsymbol{\theta}}(x) \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(x)`$：

$$
= \int f(x) p_{\boldsymbol{\theta}}(x) \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(x)\,dx = \mathbb{E}_{x \sim p_{\boldsymbol{\theta}}}[f(x) \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(x)]
$$

$`\blacksquare`$

<a id="s1-8-3"></a>

### 1.8.3 为什么这个技巧至关重要

这个公式解决了一个根本性的困难：**如何对一个依赖于参数的分布的期望求导？**

直接方法是先算出期望的解析表达式再求导，但这在 RL 中不可能——轨迹分布 $`p_{\boldsymbol{\theta}}(\tau)`$ 极其复杂，期望无法解析求出。

似然比技巧的精妙之处在于：它将「对分布求导」转化为「在同一分布下求另一个量的期望」。右边的期望仍然无法解析计算，但现在可以用蒙特卡洛方法估计——只需从 $`p_{\boldsymbol{\theta}}`$ 中采样 $`x_1, \dots, x_N`$，计算 $`\frac{1}{N}\sum_i f(x_i) \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(x_i)`$ 即可。

换句话说，似然比技巧把「微分期望」变成了「期望的微分」，而后者可以通过采样来数值逼近。这就是 REINFORCE 算法（见第 3 章）的数学基础，也是所有策略梯度方法的起点。

**正则条件的重要性**：交换微分与积分要求支撑集不随 $`\boldsymbol{\theta}`$ 变化。对于 Categorical 分布和 Gaussian 分布，这一条件满足。但对于某些参数化（如均匀分布 $`U(0, \theta)`$），支撑集依赖 $`\theta`$，似然比技巧不直接适用。在实际 RL 中，我们选择的策略分布族都满足正则条件。

<a id="s1-8-4"></a>

### 1.8.4 对数似然的二阶矩与 Fisher 信息

得分函数的协方差矩阵定义为 **Fisher 信息矩阵（Fisher Information Matrix, FIM）**：

$$
\boldsymbol{F}(\boldsymbol{\theta}) = \mathbb{E}_{x \sim p_{\boldsymbol{\theta}}}[\nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(x) \, \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(x)^\top]
$$

由于得分函数均值为零（1.8.1 节），这就是其二阶矩矩阵。Fisher 信息矩阵也可以写成 KL 散度的 Hessian：

$$
\boldsymbol{F}(\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}} \nabla_{\boldsymbol{\theta}'} D_{\mathrm{KL}}(p_{\boldsymbol{\theta}} \| p_{\boldsymbol{\theta}'}) \big|_{\boldsymbol{\theta}' = \boldsymbol{\theta}}
$$

这个关系揭示了 Fisher 信息的几何含义：它度量了参数空间中分布族的局部曲率。在 $`\boldsymbol{\theta}`$ 附近，KL 散度的二阶近似为 $`D_{\mathrm{KL}}(p_{\boldsymbol{\theta}} \| p_{\boldsymbol{\theta} + \Delta \boldsymbol{\theta}}) \approx \frac{1}{2} \Delta \boldsymbol{\theta}^\top \boldsymbol{F}(\boldsymbol{\theta}) \Delta \boldsymbol{\theta}`$。

**与后续章节的联系**：自然梯度（Natural Gradient）定义为 $`\tilde{\nabla} J = \boldsymbol{F}^{-1} \nabla J`$，它在 Fisher 信息度量下的最速上升方向。TRPO 使用 KL 散度的二阶近似（即 Fisher 矩阵）作为约束，推导出自然梯度方向的更新规则（见第 5 章）。本节只建立 Fisher 信息的定义和基本性质，完整的自然梯度理论和 TRPO 推导留给第 5 章。

<a id="s1-9"></a>

## 1.9 本章小结与自测

<a id="h47"></a>

### 要点回顾

1. **两类策略分布**：离散动作用 Categorical（logits + softmax），连续动作用对角 Gaussian（均值 + 对数标准差）。两者的 $`\nabla \log \pi`$ 都有简洁的闭式表达，是策略梯度实现的基础。
2. **期望的线性性与塔性质**是 RL 中几乎所有推导的底层工具。方差的不可加性（因相关性）解释了回报估计的高方差问题。
3. **KL 散度非负性**由 Jensen 不等式证明；前向 KL（mean-seeking）与反向 KL（mode-seeking）的行为差异决定了 PPO 选择前向 KL 作为约束方向。
4. **似然比技巧**将 $`\nabla_{\boldsymbol{\theta}} \mathbb{E}_{p_{\boldsymbol{\theta}}}[f]`$ 转化为 $`\mathbb{E}_{p_{\boldsymbol{\theta}}}[f \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}]`$，使策略梯度可通过采样估计。这是策略梯度方法的数学根基。
5. **Fisher 信息矩阵**是得分函数的二阶矩，也是 KL 散度的局部曲率。它为自然梯度和 TRPO 提供了几何框架（详见第 5 章）。

<a id="h48"></a>

### 自测题

**题目 1**：设策略为 Categorical 分布，logits 为 $`\boldsymbol{z}`$，证明 $`\nabla_{z_j} \log \pi(a=k|\boldsymbol{z}) = \delta_{kj} - \pi_j`$，其中 $`\pi_j = \text{softmax}(\boldsymbol{z})_j`$。

<details>
<summary>查看参考答案</summary>

参考答案：$`\log \pi_k = z_k - \log \sum_l \exp(z_l)`$。对 $`z_j`$ 求导：$`\partial z_k / \partial z_j = \delta_{kj}`$，$`\partial \log \sum_l \exp(z_l) / \partial z_j = \exp(z_j) / \sum_l \exp(z_l) = \pi_j`$。两者相减得 $`\delta_{kj} - \pi_j`$。

</details>

**题目 2**：用 Jensen 不等式证明 $`D_{\mathrm{KL}}(p \| q) \geq 0`$，并指出等号成立的条件。

<details>
<summary>查看参考答案</summary>

参考答案：$`-\log`$ 是严格凸函数。由 Jensen 不等式，$`-\log \mathbb{E}_p[q/p] \leq \mathbb{E}_p[-\log(q/p)]`$。左边 $`= -\log 1 = 0`$，右边 $`= D_{\mathrm{KL}}(p\|q)`$。等号当且仅当 $`q/p`$ 为常数（几乎处处），即 $`p = q`$。

</details>

**题目 3**：解释为什么 mini-batch SGD 在 RL 中可能产生有偏梯度估计，而在监督学习中通常是无偏的。

<details>
<summary>查看参考答案</summary>

参考答案：监督学习中数据集固定，mini-batch 是从固定分布中 i.i.d. 采样的，梯度估计无偏。RL 中数据由当前策略生成，当用同一批数据做多次更新时，参数已变但数据未重新采集，数据不再服从当前策略分布，导致偏差。PPO 通过裁剪比率控制此偏差的影响。

</details>

**题目 4**：写出 $`f(\boldsymbol{\theta} + \Delta \boldsymbol{\theta})`$ 的二阶泰勒展开（含余项），并说明为什么它在 TRPO 中至关重要。

<details>
<summary>查看参考答案</summary>

参考答案：$`f(\boldsymbol{\theta} + \Delta \boldsymbol{\theta}) = f(\boldsymbol{\theta}) + \nabla f^\top \Delta \boldsymbol{\theta} + \frac{1}{2} \Delta \boldsymbol{\theta}^\top \boldsymbol{H} \Delta \boldsymbol{\theta} + o(\|\Delta \boldsymbol{\theta}\|^2)`$。TRPO 用此展开近似性能改进，并将 Hessian 替换为 Fisher 信息矩阵作为 KL 约束的二次近似，从而导出自然梯度更新方向。没有二阶展开就无法建立 KL 约束与参数更新之间的定量关系。

</details>

**题目 5**：证明 $`\mathbb{E}_{x \sim p_{\boldsymbol{\theta}}}[\nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(x)] = \boldsymbol{0}`$，并解释这个结果在策略梯度估计中的作用。

<details>
<summary>查看参考答案</summary>

参考答案：由 $`\int p_{\boldsymbol{\theta}} dx = 1`$ 两边对 $`\boldsymbol{\theta}`$ 求导，交换微分积分得 $`\int \nabla p_{\boldsymbol{\theta}} dx = 0`$。代入 $`\nabla p = p \nabla \log p`$ 即得证。该结果表明得分函数均值为零，因此在策略梯度估计 $`\mathbb{E}[R \nabla \log \pi]`$ 中加入与 $`\nabla \log \pi`$ 无关的常数基线 $`b`$ 不改变期望值（因为 $`\mathbb{E}[b \nabla \log \pi] = b \cdot \boldsymbol{0} = 0`$），但可以减小方差。这是基线函数方法的理论基础（见第 3 章）。

</details>

---

---

[返回目录](../README.md) · [阅读路线与符号表](README.md) · [下一章](chapter-02.md)
