<a id="h1"></a>

# PPO 系统学习文档

<!-- toc:start -->
[返回项目首页](README.md) · [阅读路线](#user-content-h5) · [全文通用符号约定](#user-content-h6)

## 可点击目录

- [第 1 章 数学基础](#user-content-chapter-1)
- [第 2 章 强化学习核心概念](#user-content-chapter-2)
- [第 3 章 策略梯度方法](#user-content-chapter-3)
- [第 4 章 优势估计与广义优势估计（GAE）](#user-content-chapter-4)
- [第 5 章 信任域方法与重要性采样](#user-content-chapter-5)
- [第 6 章 PPO 算法本体](#user-content-chapter-6)
- [第 7 章 工程实现与实践](#user-content-chapter-7)

<details>
<summary>展开小节目录</summary>

- [1.1 随机变量与概率分布](#user-content-s1-1)
  - [1.1.1 概率质量函数与概率密度函数](#user-content-s1-1-1)
  - [1.1.2 联合分布、条件分布与边缘化](#user-content-s1-1-2)
  - [1.1.3 Categorical 分布：离散动作空间的策略](#user-content-s1-1-3)
  - [1.1.4 对角 Gaussian 分布：连续动作空间的策略](#user-content-s1-1-4)
- [1.2 期望、方差与条件期望](#user-content-s1-2)
  - [1.2.1 期望的定义与线性性](#user-content-s1-2-1)
  - [1.2.2 方差与协方差](#user-content-s1-2-2)
  - [1.2.3 条件期望与塔性质](#user-content-s1-2-3)
- [1.3 采样估计与蒙特卡洛方法](#user-content-s1-3)
  - [1.3.1 样本均值估计](#user-content-s1-3-1)
  - [1.3.2 从期望到采样的桥梁](#user-content-s1-3-2)
- [1.4 信息论基础](#user-content-s1-4)
  - [1.4.1 熵与最大熵性质](#user-content-s1-4-1)
  - [1.4.2 交叉熵](#user-content-s1-4-2)
  - [1.4.3 KL 散度：定义与非负性证明](#user-content-s1-4-3)
  - [1.4.4 KL 散度的不对称性与前向/反向 KL](#user-content-s1-4-4)
- [1.5 Jensen 不等式与常用不等式](#user-content-s1-5)
  - [1.5.1 Jensen 不等式](#user-content-s1-5-1)
  - [1.5.2 其他常用不等式](#user-content-s1-5-2)
- [1.6 优化基础](#user-content-s1-6)
  - [1.6.1 梯度与方向导数](#user-content-s1-6-1)
  - [1.6.2 Hessian 矩阵与二阶泰勒展开](#user-content-s1-6-2)
  - [1.6.3 凸函数与局部近似](#user-content-s1-6-3)
- [1.7 随机梯度下降与 Adam 优化器](#user-content-s1-7)
  - [1.7.1 Mini-batch 梯度的偏差与可行性](#user-content-s1-7-1)
  - [1.7.2 Adam 优化器](#user-content-s1-7-2)
  - [1.7.3 学习率与梯度裁剪的数学含义](#user-content-s1-7-3)
- [1.8 似然比技巧（Likelihood Ratio Trick）](#user-content-s1-8)
  - [1.8.1 基本恒等式：得分函数的期望为零](#user-content-s1-8-1)
  - [1.8.2 似然比技巧的核心公式](#user-content-s1-8-2)
  - [1.8.3 为什么这个技巧至关重要](#user-content-s1-8-3)
  - [1.8.4 对数似然的二阶矩与 Fisher 信息](#user-content-s1-8-4)
- [1.9 本章小结与自测](#user-content-s1-9)
  - [要点回顾](#user-content-h47)
  - [自测题](#user-content-h48)
- [2.1 为什么用马尔可夫决策过程建模](#user-content-s2-1)
  - [2.1.1 Agent-Environment 交互循环](#user-content-s2-1-1)
  - [2.1.2 MDP 五元组的严格定义](#user-content-s2-1-2)
  - [2.1.3 马尔可夫性质及其局限](#user-content-s2-1-3)
- [2.2 轨迹的概率表达式](#user-content-s2-2)
  - [2.2.1 轨迹的定义](#user-content-s2-2-1)
  - [2.2.2 轨迹概率的连乘分解](#user-content-s2-2-2)
- [2.3 回报与折扣因子](#user-content-s2-3)
  - [2.3.1 折扣回报的定义](#user-content-s2-3-1)
  - [2.3.2 折扣因子的三重解释](#user-content-s2-3-2)
  - [2.3.3 有界性证明](#user-content-s2-3-3)
  - [2.3.4 $`\gamma \to 1`$ 时的行为](#user-content-s2-3-4)
- [2.4 价值函数三件套](#user-content-s2-4)
  - [2.4.1 状态价值函数](#user-content-s2-4-1)
  - [2.4.2 动作价值函数](#user-content-s2-4-2)
  - [2.4.3 优势函数](#user-content-s2-4-3)
  - [2.4.4 三者关系式的推导](#user-content-s2-4-4)
- [2.5 贝尔曼期望方程](#user-content-s2-5)
  - [2.5.1 从回报的递归结构出发](#user-content-s2-5-1)
  - [2.5.2 $`V^\pi`$ 的贝尔曼期望方程](#user-content-s2-5-2)
  - [2.5.3 $`Q^\pi`$ 的贝尔曼期望方程](#user-content-s2-5-3)
  - [2.5.4 算子形式与矩阵形式](#user-content-s2-5-4)
- [2.6 贝尔曼最优方程与最优策略](#user-content-s2-6)
  - [2.6.1 最优价值函数与贝尔曼最优算子](#user-content-s2-6-1)
  - [2.6.2 $`\gamma`$-压缩性与 Banach 不动点定理](#user-content-s2-6-2)
  - [2.6.3 最优策略的存在性](#user-content-s2-6-3)
- [2.7 时序差分误差](#user-content-s2-7)
  - [2.7.1 TD 误差的定义](#user-content-s2-7-1)
  - [2.7.2 TD 误差与优势函数的关系](#user-content-s2-7-2)
  - [2.7.3 当 $`V \neq V^\pi`$ 时的情形](#user-content-s2-7-3)
- [2.8 蒙特卡洛估计与时序差分估计的对比](#user-content-s2-8)
- [2.9 策略的形式、分类与强化学习的根本困境](#user-content-s2-9)
  - [2.9.1 随机策略与确定性策略](#user-content-s2-9-1)
  - [2.9.2 参数化策略](#user-content-s2-9-2)
  - [2.9.3 On-policy 与 Off-policy 的本质区别](#user-content-s2-9-3)
  - [2.9.4 强化学习的根本困境](#user-content-s2-9-4)
- [2.10 本章小结与自测](#user-content-s2-10)
  - [要点回顾](#user-content-h87)
  - [自测题](#user-content-h88)
- [3.1 优化目标的两种等价形式](#user-content-s3-1)
- [3.2 策略梯度定理的完整推导](#user-content-s3-2)
  - [3.2.1 路径一：直接展开轨迹分布的梯度](#user-content-s3-2-1)
  - [3.2.2 路径二：似然比技巧一步得到](#user-content-s3-2-2)
  - [3.2.3 策略梯度定理的标准形式](#user-content-s3-2-3)
- [3.3 因果性与 Reward-to-Go](#user-content-s3-3)
  - [3.3.1 因果性定理及其证明](#user-content-s3-3-1)
  - [3.3.2 Reward-to-Go 形式](#user-content-s3-3-2)
- [3.4 用 Q 函数表达的策略梯度定理](#user-content-s3-4)
- [3.5 基线理论与优势函数](#user-content-s3-5)
  - [3.5.1 基线的无偏性证明](#user-content-s3-5-1)
  - [3.5.2 为什么优势函数能显著降方差](#user-content-s3-5-2)
  - [3.5.3 最优基线的推导](#user-content-s3-5-3)
- [3.6 策略梯度的偏差-方差分解框架](#user-content-s3-6)
- [3.7 Actor-Critic 架构](#user-content-s3-7)
  - [3.7.1 两个网络的分工](#user-content-s3-7-1)
  - [3.7.2 Critic 的损失函数](#user-content-s3-7-2)
  - [3.7.3 为什么两个网络的梯度必须分开](#user-content-s3-7-3)
  - [3.7.4 Critic 误差对 Actor 的影响](#user-content-s3-7-4)
- [3.8 n-step 方法与 A2C/A3C](#user-content-s3-8)
  - [3.8.1 A3C 的异步多 Worker 动机](#user-content-s3-8-1)
  - [3.8.2 A2C：同步版本](#user-content-s3-8-2)
  - [3.8.3 两者的等价性讨论](#user-content-s3-8-3)
- [3.9 REINFORCE 伪代码](#user-content-s3-9)
- [3.10 策略梯度方法的固有缺陷](#user-content-s3-10)
  - [3.10.1 样本效率极低](#user-content-s3-10-1)
  - [3.10.2 对步长极其敏感](#user-content-s3-10-2)
  - [3.10.3 高方差](#user-content-s3-10-3)
- [3.11 本章小结与自测](#user-content-s3-11)
  - [要点回顾](#user-content-h119)
  - [自测题](#user-content-h120)
- [4.1 本章起点：从策略梯度到优势估计器](#user-content-s4-1)
- [4.2 k-step 优势估计器的定义与推导](#user-content-s4-2)
  - [4.2.1 TD 残差与 1-step 优势估计](#user-content-s4-2-1)
  - [4.2.2 2-step 优势估计及其等价形式](#user-content-s4-2-2)
  - [4.2.3 一般 k-step 优势估计器](#user-content-s4-2-3)
  - [4.2.4 无穷步极限：蒙特卡洛优势估计](#user-content-s4-2-4)
- [4.3 偏差-方差的量化分析](#user-content-s4-3)
  - [4.3.1 偏差的来源与表达式](#user-content-s4-3-1)
  - [4.3.2 方差的来源与量级](#user-content-s4-3-2)
  - [4.3.3 偏差-方差权衡总结表](#user-content-s4-3-3)
- [4.4 TD($`\lambda`$) 回顾：$`\lambda`$-return 与资格迹](#user-content-s4-4)
  - [4.4.1 前向视角：$`\lambda`$-return](#user-content-s4-4-1)
  - [4.4.2 后向视角与资格迹](#user-content-s4-4-2)
- [4.5 广义优势估计（GAE）的完整推导](#user-content-s4-5)
  - [4.5.1 动机](#user-content-s4-5-1)
  - [4.5.2 GAE 的定义](#user-content-s4-5-2)
  - [4.5.3 核心推导：交换求和顺序](#user-content-s4-5-3)
  - [4.5.4 两个极端情形的验证](#user-content-s4-5-4)
  - [4.5.5 $`\gamma`$ 与 $`\lambda`$ 的角色区分](#user-content-s4-5-5)
- [4.6 递推实现形式与实践处理](#user-content-s4-6)
  - [4.6.1 从无穷级数到递推公式](#user-content-s4-6-1)
  - [4.6.2 配套的 Critic 回归目标](#user-content-s4-6-2)
  - [4.6.3 Episode 边界处理](#user-content-s4-6-3)
  - [4.6.4 有限 Horizon 下的等价性说明](#user-content-s4-6-4)
- [4.7 实践中的数学处理与 GAE 和 PPO 的关系](#user-content-s4-7)
  - [4.7.1 优势归一化](#user-content-s4-7-1)
  - [4.7.2 截断与终止的数学区别再述](#user-content-s4-7-2)
  - [4.7.3 GAE 与 PPO 的正交关系](#user-content-s4-7-3)
- [4.8 本章小结与自测](#user-content-s4-8)
  - [要点回顾](#user-content-h151)
  - [自测题](#user-content-h152)
- [5.1 引言：为什么需要限制策略更新幅度](#user-content-s5-1)
- [5.2 重要性采样的严格推导](#user-content-s5-2)
  - [5.2.1 基本恒等式与无偏性](#user-content-s5-2-1)
  - [5.2.2 方差爆炸与有效样本量](#user-content-s5-2-2)
- [5.3 策略层面的重要性采样](#user-content-s5-3)
  - [5.3.1 轨迹比率与单步比率](#user-content-s5-3-1)
  - [5.3.2 为什么单步比率就够了](#user-content-s5-3-2)
- [5.4 性能差异引理](#user-content-s5-4)
  - [5.4.1 引理陈述](#user-content-s5-4-1)
  - [5.4.2 完整推导](#user-content-s5-4-2)
  - [5.4.3 关键观察](#user-content-s5-4-3)
- [5.5 TRPO 的单调改进界](#user-content-s5-5)
  - [5.5.1 第一步：将 $`d^{\pi'}`$ 替换为 $`d^\pi`$ 并量化误差](#user-content-s5-5-1)
  - [5.5.2 第二步：单调改进下界](#user-content-s5-5-2)
  - [5.5.3 第三步：下界的关键性质与信任域的含义](#user-content-s5-5-3)
- [5.6 TRPO 的实际求解形式](#user-content-s5-6)
  - [5.6.1 约束优化问题](#user-content-s5-6-1)
  - [5.6.2 从 max KL 到平均 KL 的松弛](#user-content-s5-6-2)
- [5.7 自然梯度与 Fisher 信息矩阵](#user-content-s5-7)
  - [5.7.1 参数空间 vs. 分布空间](#user-content-s5-7-1)
  - [5.7.2 KL 散度的二阶展开与 Fisher 矩阵](#user-content-s5-7-2)
  - [5.7.3 自然梯度的变分推导](#user-content-s5-7-3)
  - [5.7.4 常用策略的解析 KL 公式](#user-content-s5-7-4)
- [5.8 共轭梯度法求解自然梯度](#user-content-s5-8)
  - [5.8.1 为什么不能直接求逆](#user-content-s5-8-1)
  - [5.8.2 共轭梯度法的适用性](#user-content-s5-8-2)
  - [5.8.3 Fisher-向量乘积的高效计算](#user-content-s5-8-3)
  - [5.8.4 线搜索保证 KL 约束](#user-content-s5-8-4)
- [5.9 TRPO 的实践代价与 PPO 的动机](#user-content-s5-9)
- [5.10 本章小结与自测](#user-content-s5-10)
  - [要点回顾](#user-content-h184)
  - [自测题](#user-content-h185)
- [6.1 PPO 的问题定位：从信任域到一阶近似](#user-content-s6-1)
- [6.2 PPO-Clip 的完整构造](#user-content-s6-2)
  - [6.2.1 目标函数的定义](#user-content-s6-2-1)
  - [6.2.2 六种情形的完整分析](#user-content-s6-2-2)
  - [6.2.3 悲观下界的设计哲学与不对称性](#user-content-s6-2-3)
  - [6.2.4 梯度一致性验证](#user-content-s6-2-4)
- [6.3 PPO-Penalty：自适应 KL 惩罚](#user-content-s6-3)
  - [6.3.1 目标函数](#user-content-s6-3-1)
  - [6.3.2 自适应 β 更新规则](#user-content-s6-3-2)
  - [6.3.3 PPO-Clip 与 PPO-Penalty 的对比](#user-content-s6-3-3)
- [6.4 完整的 PPO 损失函数](#user-content-s6-4)
  - [6.4.1 三项损失的组合](#user-content-s6-4-1)
  - [6.4.2 价值函数裁剪的争议](#user-content-s6-4-2)
  - [6.4.3 Actor 与 Critic 的网络共享](#user-content-s6-4-3)
- [6.5 PPO 完整算法伪代码](#user-content-s6-5)
- [6.6 PPO 的理论局限与已知问题](#user-content-s6-6)
  - [6.6.1 Clip 不保证单调改进](#user-content-s6-6-1)
  - [6.6.2 梯度截断引入偏差](#user-content-s6-6-2)
  - [6.6.3 工程细节的重要性](#user-content-s6-6-3)
  - [6.6.4 状态分布不匹配](#user-content-s6-6-4)
  - [6.6.5 后续改进方向简述](#user-content-s6-6-5)
- [6.7 PPO 在 RLHF 中的应用](#user-content-s6-7)
  - [6.7.1 RLHF 场景的特殊性](#user-content-s6-7-1)
  - [6.7.2 逐 Token 的比率与优势](#user-content-s6-7-2)
  - [6.7.3 KL 惩罚的双重角色](#user-content-s6-7-3)
  - [6.7.4 Reward Hacking 与 Mode Collapse](#user-content-s6-7-4)
  - [6.7.5 GRPO 对 PPO 的替代趋势](#user-content-s6-7-5)
- [6.8 与其他方法的定位关系](#user-content-s6-8)
- [6.9 本章小结与自测](#user-content-s6-9)
  - [要点回顾](#user-content-h216)
  - [自测题](#user-content-h217)
- [7.1 环境接口与 Gymnasium](#user-content-s7-1)
  - [7.1.1 Gymnasium API 概述](#user-content-s7-1-1)
  - [7.1.2 terminated 与 truncated 的语义区别](#user-content-s7-1-2)
- [7.2 向量环境（Vectorized Environments）](#user-content-s7-2)
  - [7.2.1 为什么 PPO 必须用向量环境](#user-content-s7-2-1)
  - [7.2.2 SyncVectorEnv 与 AsyncVectorEnv](#user-content-s7-2-2)
  - [7.2.3 Rollout Buffer 的形状约定](#user-content-s7-2-3)
  - [7.2.4 异步环境的自动 reset 语义](#user-content-s7-2-4)
- [7.3 网络结构设计的数学考量](#user-content-s7-3)
  - [7.3.1 MLP 主干配置](#user-content-s7-3-1)
  - [7.3.2 初始化对策略梯度的特殊影响](#user-content-s7-3-2)
  - [7.3.3 连续动作空间的对数标准差参数化](#user-content-s7-3-3)
  - [7.3.4 Actor 与 Critic 共享主干 vs 分离主干](#user-content-s7-3-4)
- [7.4 PPO 完整参考实现](#user-content-s7-4)
  - [7.4.1 策略网络与价值网络](#user-content-s7-4-1)
  - [7.4.2 Rollout 存储](#user-content-s7-4-2)
  - [7.4.3 GAE 与 Return Target 计算](#user-content-s7-4-3)
  - [7.4.4 Mini-batch 更新循环](#user-content-s7-4-4)
  - [7.4.5 学习率线性衰减](#user-content-s7-4-5)
  - [7.4.6 梯度裁剪澄清](#user-content-s7-4-6)
- [7.5 数值稳定性与实现细节的数学解释](#user-content-s7-5)
  - [7.5.1 对数域计算比率](#user-content-s7-5-1)
  - [7.5.2 近似 KL 的选择](#user-content-s7-5-2)
  - [7.5.3 优势归一化的位置](#user-content-s7-5-3)
  - [7.5.4 价值裁剪](#user-content-s7-5-4)
- [7.6 训练诊断指标与调参方法论](#user-content-s7-6)
  - [7.6.1 必须监控的量及其健康范围](#user-content-s7-6-1)
  - [7.6.2 调参优先级](#user-content-s7-6-2)
  - [7.6.3 训练崩溃排查清单](#user-content-s7-6-3)
  - [7.6.4 观测归一化与奖励归一化](#user-content-s7-6-4)
- [7.7 常见坑清单](#user-content-s7-7)
- [7.8 推荐学习资源与进阶路径](#user-content-s7-8)
  - [7.8.1 论文清单（按阅读顺序）](#user-content-s7-8-1)
  - [7.8.2 代码库](#user-content-s7-8-2)
  - [7.8.3 务实的学习路径建议](#user-content-s7-8-3)
- [7.9 本章小结与下一步](#user-content-s7-9)

</details>

<!-- toc:end -->


> 从零到能独立实现与调优近端策略优化（Proximal Policy Optimization）的完整理论路径

---

<a id="h2"></a>

## 关于这份文档

这份文档的目标是：**让一个有微积分、线性代数、概率论基础，会一点 Python，但完全没学过强化学习的人，能够真正理解 PPO 的每一个公式是怎么来的，并且能自己把它写出来、调好。**

写作定位是**偏数学与理论推导**。文档不满足于「给出公式然后告诉你怎么用」，而是坚持回答三个问题：

- 这个公式**从哪里出发**推导出来？
- 中间做了**什么近似或假设**？近似掉的那部分去了哪里？
- 为什么**必须是这个形式**，换一种写法会出什么问题？

因此你会看到大量逐步展开的推导，而不是「易得」「显然」。凡是实现与理论有出入的地方（数值稳定性处理、截断、bootstrap 边界），文档都会显式标注差异并说明原因。

文档由 7 章组成，采用严格的依赖链结构：**后一章的推导会直接使用前一章的结论**。建议顺序阅读；如果只想快速抵达 PPO 本体，最短路径见下文「阅读路线」。

---

<a id="h3"></a>

## 全文逻辑主线

PPO 不是一组孤立的技巧，而是一条清晰的因果链。整份文档就是沿着这条链展开的：

```
想优化策略 → 但策略决定数据分布，没有固定数据集
          ↓  （第 2 章：RL 的根本困境）
用似然比技巧把「对分布求导」变成「在分布下求期望」，于是能用采样估计
          ↓  （第 1 章：似然比技巧 → 第 3 章：策略梯度定理）
REINFORCE 能跑，但方差巨大、样本只能用一次
          ↓  （第 3 章：基线 → 优势函数 A = Q - V）
优势怎么估？k 步小则偏差大方差小，k 步大则相反
          ↓  （第 4 章：GAE 用指数加权把所有 k 融合成一个可调旋钮 λ）
一批数据只更新一次太浪费，但多更新几次策略就会崩溃
          ↓  （第 5 章：TRPO 证明「只要 KL 足够小，性能就单调不降」）
TRPO 的信任域要用二阶优化 + 共轭梯度，实现复杂、开销大
          ↓  （第 6 章：PPO 用一阶方法近似信任域 —— clip 掉比率，直接截断梯度）
理论落地时到处是与公式不一致的细节
          ↓  （第 7 章：terminated/truncated、初始化、归一化、诊断指标）
```

一句话概括 PPO 的定位：**它用一阶优化方法（普通的 Adam）近似实现了 TRPO 的信任域约束，代价是失去了理论上的单调改进保证，换来的是实现简单、可扩展、能与各种网络架构自由组合。** 这个取舍是否值得，第 6.6 节会诚实讨论（包括社区的重要质疑）。

---

<a id="h4"></a>

## 各章依赖关系

| 章节 | 标题 | 依赖 | 核心产出 |
|---|---|---|---|
| 第 1 章 | 数学基础 | — | 似然比技巧、KL 散度、二阶泰勒展开、Fisher 信息 |
| 第 2 章 | 强化学习核心概念 | 第 1 章 | MDP、V/Q/A 函数、贝尔曼方程、TD 误差 |
| 第 3 章 | 策略梯度方法 | 第 1、2 章 | 策略梯度定理、基线理论、Actor-Critic |
| 第 4 章 | 优势估计与 GAE | 第 2、3 章 | GAE 闭式解与递推式、偏差-方差分析框架 |
| 第 5 章 | 信任域与重要性采样 | 第 1、2、3 章 | 重要性采样、性能差异引理、TRPO 单调改进界、自然梯度 |
| 第 6 章 | PPO 算法本体 | 第 1–5 章 | PPO-Clip / PPO-Penalty、完整损失函数、伪代码、RLHF 应用 |
| 第 7 章 | 工程实现与实践 | 第 1–6 章 | 可运行的 PyTorch 实现、诊断指标、常见坑、进阶资源 |

---

<a id="h5"></a>

## 阅读路线

**完整路线（推荐）**：第 1 → 2 → 3 → 4 → 5 → 6 → 7 章顺序读完。这是理解最扎实的走法。

**最短路线（想尽快看到 PPO）**：第 2.2、2.4、2.7 节（轨迹概率、V/Q/A、TD 误差）→ 第 3.2、3.5 节（策略梯度定理、基线）→ 第 4.2、4.5、4.6 节（k-step 优势、GAE 推导与递推）→ 第 5.2 节（重要性采样）→ 第 6 章全部 → 第 7.4 节（参考实现）。走完这条线大约能覆盖 60% 的内容，但会损失对「为什么」的理解。

**实践优先路线（想先跑起来再回头补理论）**：第 7 章 → 第 6 章 → 第 4 章 → 第 3 章 → 第 5 章 → 第 2 章 → 第 1 章。先用第 7.4 节的代码把 CartPole 跑通，建立直觉，再逐层往回追根溯源。这条路线对「已经会写 RL 代码但不懂原理」的人最有效。

**只关心 RLHF / 大模型对齐**：第 1.4 节（KL 散度）→ 第 5.2 节（重要性采样）→ 第 6 章全部，重点是 6.7 节。可以跳过第 2、4、7 章的环境相关部分。

---

<a id="h6"></a>

## 全文通用符号约定

以下为贯穿全文的记号。**各章可能有局部的补充记号，以章内首次定义为准。**

<a id="h7"></a>

### 基本对象

| 符号 | 含义 |
|---|---|
| $`s_t`$, $`a_t`$, $`r_t`$ | 时刻 $`t`$ 的状态、动作、奖励 |
| $`\mathcal{S}`$, $`\mathcal{A}`$ | 状态空间、动作空间 |
| $`\pi_\theta(a\mid s)`$ | 由参数 $`\theta`$ 决定的策略，输出在状态 $`s`$ 下采取动作 $`a`$ 的概率 |
| $`\pi_{\theta_{\text{old}}}`$ | 采集数据时使用的旧策略（行为策略） |
| $`V^\pi(s)`$ | 状态价值函数：从 $`s`$ 出发、遵循 $`\pi`$ 的期望回报 |
| $`Q^\pi(s,a)`$ | 动作价值函数：在 $`s`$ 先执行 $`a`$、之后遵循 $`\pi`$ 的期望回报 |
| $`A^\pi(s,a)`$ | 优势函数：$`A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)`$ |
| $`\hat{A}_t`$ | 优势函数的估计量（带帽子表示估计值） |
| $`G_t`$ | 从时刻 $`t`$ 开始的回报（return） |
| $`\delta_t`$ | 时序差分误差：$`\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)`$ |

<a id="h8"></a>

### 标量与常数

| 符号 | 含义 | 典型取值 |
|---|---|---|
| $`\gamma`$ | 折扣因子，定义问题本身的时间偏好 | 0.99 |
| $`\lambda`$ | GAE 的偏差-方差调节旋钮（**纯人为引入，与问题定义无关**） | 0.95 |
| $`\varepsilon`$ | PPO-Clip 的截断半径 | 0.2 |
| $`\beta`$ | KL 惩罚系数 | 自适应 |
| $`c_1`$, $`c_2`$ | 价值损失系数、熵奖励系数 | 0.5, 0.01 |
| $`K`$ | 每批数据的复用 epoch 数 | 3–10 |
| $`N`$ | 并行环境数 | 8–64 |
| $`T`$ | 每个环境的 rollout 步数 | 128–2048 |

<a id="h9"></a>

### 算子与矩阵

| 符号 | 含义 |
|---|---|
| $`\nabla_\theta`$ | 对参数 $`\theta`$ 的梯度 |
| $`D_{\mathrm{KL}}(p \,\|\, q)`$ | KL 散度（注意不对称，顺序有意义） |
| $`\mathbb{E}`$, $`\mathrm{Var}`$ | 期望、方差 |
| $`T^\pi`$, $`T^*`$ | 贝尔曼期望算子、贝尔曼最优算子 |
| $`F(\theta)`$ | Fisher 信息矩阵 |
| $`r_t(\theta)`$ | 重要性采样比率：$`\dfrac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}`$ |
| $`d^\pi(s)`$ | 折扣状态访问分布 |

<a id="h10"></a>

### 记号约定

- **向量与矩阵**：小写粗体或普通字母视上下文而定，各章首次使用时说明。矩阵用大写。
- **估计量与真值**：真值无上标（如 $`V^\pi`$），估计量带帽子（如 $`\hat{A}_t`$）或下标 $`\phi`$（价值网络的参数，如 $`V_\phi`$）。
- **$`\theta`$ 与 $`\phi`$**：$`\theta`$ 恒指 actor（策略网络）参数，$`\phi`$ 恒指 critic（价值网络）参数。
- **两个 "clip" 的区别**：第 6 章的 `clip` 指 PPO 对重要性采样比率 $`r_t(\theta)`$ 的截断（算法机制）；第 7 章的 `clip_grad_norm_` 指对梯度全局 L2 范数的裁剪（数值稳定性手段）。两者毫无关系，详见 7.4 节。

---

<a id="h11"></a>

## 数学公式说明

- 所有公式使用 LaTeX 书写。行内公式为 `$...$`，独立公式为 `$$...$$`。
- 若你的 Markdown 阅读器不渲染 LaTeX，建议用 Typora、Obsidian、VS Code（装 Markdown+Math 插件）或任何支持 KaTeX/MathJax 的工具打开。
- 关键推导均逐步展开，中间步骤不省略。建议**手推一遍**再往下读——策略梯度和 GAE 这两处推导，看懂和推出来是两回事。

---

<a id="chapter-1"></a>

# 第 1 章 数学基础

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

```math
p(y|x) = \frac{p(x, y)}{p(x)}, \quad \text{当 } p(x) > 0
```

边缘分布通过对另一个变量求和（离散）或积分（连续）得到：

```math
p(x) = \sum_y p(x, y) \quad \text{或} \quad p(x) = \int p(x, y)\,dy
```

这三个概念的关系即贝叶斯公式的基础：$`p(x, y) = p(y|x)\,p(x) = p(x|y)\,p(y)`$。在 RL 中，轨迹的概率分解正是反复应用这一链式法则的结果——一条轨迹 $`(s_0, a_0, s_1, a_1, \dots)`$ 的联合概率被拆分为初始状态分布、策略条件和环境转移条件的乘积。

<a id="s1-1-3"></a>

### 1.1.3 Categorical 分布：离散动作空间的策略

Categorical 分布（也称多项分布的单次试验特例）是离散动作空间中最常用的策略分布。设动作空间大小为 $`K`$，策略网络输出一个 $`K`$ 维向量 $`\boldsymbol{z} = (z_1, \dots, z_K)`$，称为 logits（未归一化的对数分数）。通过 softmax 函数将其转化为合法概率：

```math
\pi(a=k|\boldsymbol{z}) = \frac{\exp(z_k)}{\sum_{j=1}^{K} \exp(z_j)}, \quad k = 1, \dots, K
```

**为什么用 logits 而非直接输出概率？** 因为 softmax 将无约束的 $`\mathbb{R}^K`$ 映射到概率单纯形，使网络输出无需额外约束即可保证合法性。更重要的是，softmax 关于 logits 的梯度有简洁形式。记 $`\pi_k = \pi(a=k|\boldsymbol{z})`$，则：

```math
\frac{\partial \pi_k}{\partial z_j} = \pi_k (\delta_{kj} - \pi_j)
```

其中 $`\delta_{kj}`$ 是 Kronecker delta。这个结果在对策略梯度做反向传播时直接出现。此外，$`\log \pi_k = z_k - \log \sum_j \exp(z_j)`$，其对 $`z_j`$ 的梯度为：

```math
\frac{\partial \log \pi_k}{\partial z_j} = \delta_{kj} - \pi_j
```

这正是策略梯度定理中 $`\nabla_{\boldsymbol{\theta}} \log \pi(a|s)`$ 的具体形式（见第 3 章）。注意该梯度只依赖于 $`\pi`$ 本身，不显式涉及 $`z`$，数值稳定性好。

<a id="s1-1-4"></a>

### 1.1.4 对角 Gaussian 分布：连续动作空间的策略

当动作空间为 $`\mathbb{R}^d`$ 时，最常用的策略分布是对角多元高斯分布（diagonal multivariate Gaussian）。其 PDF 为：

```math
\pi(\boldsymbol{a}|\boldsymbol{s}) = \prod_{i=1}^{d} \frac{1}{\sqrt{2\pi}\sigma_i} \exp\!\left(-\frac{(a_i - \mu_i)^2}{2\sigma_i^2}\right)
```

其中均值向量 $`\boldsymbol{\mu} = (\mu_1, \dots, \mu_d)`$ 和对数标准差向量 $`\log \boldsymbol{\sigma} = (\log \sigma_1, \dots, \log \sigma_d)`$ 均由策略网络输出。

**为什么参数化 $`\log \sigma`$ 而非 $`\sigma`$？** 标准差必须为正，而神经网络输出是无约束实数。若直接输出 $`\sigma`$，需加 softplus 或 exp 变换来保证正性，这会引入额外的非线性并可能导致梯度消失。使用 $`\log \sigma`$ 作为参数，$`\sigma = \exp(\log \sigma)`$ 自动为正，且对数似然关于 $`\log \sigma`$ 的梯度形式更干净：

```math
\log \pi(\boldsymbol{a}|\boldsymbol{s}) = -\frac{d}{2}\log(2\pi) - \sum_{i=1}^{d} \log \sigma_i - \sum_{i=1}^{d} \frac{(a_i - \mu_i)^2}{2\sigma_i^2}
```

对 $`\mu_i`$ 求导得 $`\frac{\partial \log \pi}{\partial \mu_i} = \frac{a_i - \mu_i}{\sigma_i^2}`$，对 $`\log \sigma_i`$ 求导得 $`\frac{\partial \log \pi}{\partial \log \sigma_i} = -1 + \frac{(a_i - \mu_i)^2}{\sigma_i^2}`$。这些表达式在实现连续动作空间的策略梯度时直接使用。

**为什么假设对角协方差？** 完整协方差矩阵有 $`O(d^2)`$ 个参数，且需保证正定性，计算和存储代价高。对角假设意味着各动作维度独立，虽然牺牲了建模能力，但大幅简化了采样和梯度计算。实践中这一假设通常足够，因为策略网络的隐层已经捕获了维度间的相关性。

<a id="s1-2"></a>

## 1.2 期望、方差与条件期望

<a id="s1-2-1"></a>

### 1.2.1 期望的定义与线性性

离散情形：$`\mathbb{E}[f(X)] = \sum_x f(x)\,p(x)`$。连续情形：$`\mathbb{E}[f(X)] = \int f(x)\,p(x)\,dx`$。

期望的核心性质是**线性性**：对任意常数 $`a, b`$ 和函数 $`f, g`$，

```math
\mathbb{E}[a f(X) + b g(Y)] = a\,\mathbb{E}[f(X)] + b\,\mathbb{E}[g(Y)]
```

这一性质不要求 $`X, Y`$ 独立，是后续所有期望运算的基础。特别地，在策略梯度推导中，我们需要将 $`\mathbb{E}[\sum_t \nabla \log \pi \cdot r_t]`$ 拆分为各项之和，线性性是这一步成立的唯一依据。

<a id="s1-2-2"></a>

### 1.2.2 方差与协方差

方差定义为 $`\mathrm{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2`$。

**方差的可加性**仅在变量不相关时成立：若 $`\mathrm{Cov}(X_i, X_j) = 0`$ 对所有 $`i \neq j`$，则 $`\mathrm{Var}(\sum_i X_i) = \sum_i \mathrm{Var}(X_i)`$。一般情况下：

```math
\mathrm{Var}\!\left(\sum_{i=1}^{n} X_i\right) = \sum_{i=1}^{n} \mathrm{Var}(X_i) + 2\sum_{i<j} \mathrm{Cov}(X_i, X_j)
```

在 RL 中，回报 $`G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}`$ 的各项之间高度相关（因为共享未来的随机性），因此回报的方差远大于各项方差之和。这是理解「为什么回报估计噪声大」以及「为什么需要基线函数减方差」（见第 3 章）的关键背景。

<a id="s1-2-3"></a>

### 1.2.3 条件期望与塔性质

条件期望 $`\mathbb{E}[Y|X=x] = \sum_y y\,p(y|x)`$（离散）或对应积分形式。它是一个关于 $`x`$ 的函数，因此 $`\mathbb{E}[Y|X]`$ 本身是一个随机变量。

**塔性质（Law of Total Expectation）**：

```math
\mathbb{E}[Y] = \mathbb{E}_X[\mathbb{E}[Y|X]]
```

即先对 $`Y`$ 在给定 $`X`$ 下求条件期望，再对 $`X`$ 求边际期望，等于直接对 $`Y`$ 求期望。这个性质在 RL 中无处不在：贝尔曼方程本质上就是塔性质的应用——将未来回报的期望分解为当前奖励加上下一状态价值的期望（见第 2 章）。

**条件方差的分解**（Law of Total Variance）也值得记住：

```math
\mathrm{Var}(Y) = \mathbb{E}_X[\mathrm{Var}(Y|X)] + \mathrm{Var}_X(\mathbb{E}[Y|X])
```

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

```math
\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}) = \mathbb{E}_{\tau \sim p_{\boldsymbol{\theta}}}[R(\tau) \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(\tau)] \approx \frac{1}{N}\sum_{i=1}^{N} R(\tau_i) \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(\tau_i)
```

左边的等号来自似然比技巧，右边的近似来自蒙特卡洛估计。两者缺一不可。

<a id="s1-4"></a>

## 1.4 信息论基础

<a id="s1-4-1"></a>

### 1.4.1 熵与最大熵性质

离散分布 $`p`$ 的香农熵（Shannon entropy）定义为：

```math
H(p) = -\sum_x p(x) \log p(x) = \mathbb{E}_{x \sim p}[-\log p(x)]
```

连续分布对应微分熵（differential entropy），形式类似但用积分替代求和，且不再具有非负性等离散熵的部分性质。

**最大熵性质**：在所有支撑集相同的分布中，均匀分布的熵最大。对于支撑集为 $`\{1, \dots, K\}`$ 的离散分布，$`H(p) \leq \log K`$，等号当且仅当 $`p(x) = 1/K`$ 时成立。证明可用 Jensen 不等式（1.5 节）或拉格朗日乘子法。

在 RL 中，熵正则化项 $`\beta H(\pi(\cdot|s))`$ 被加入目标函数以鼓励探索：熵越大，策略越接近均匀分布，探索越充分。PPO 的实现中几乎总是包含这一项（见第 6 章）。

<a id="s1-4-2"></a>

### 1.4.2 交叉熵

分布 $`p`$ 相对于 $`q`$ 的交叉熵（cross-entropy）定义为：

```math
H(p, q) = -\sum_x p(x) \log q(x) = \mathbb{E}_{x \sim p}[-\log q(x)]
```

注意期望是对 $`p`$ 取的，不是 $`q`$。交叉熵衡量的是「用 $`q`$ 编码服从 $`p`$ 的数据时的平均码长」。在机器学习中，分类任务的损失函数就是真实标签分布与模型预测分布之间的交叉熵。

<a id="s1-4-3"></a>

### 1.4.3 KL 散度：定义与非负性证明

KL 散度（Kullback-Leibler divergence）定义为交叉熵减去熵：

```math
D_{\mathrm{KL}}(p \| q) = H(p, q) - H(p) = \sum_x p(x) \log \frac{p(x)}{q(x)} = \mathbb{E}_{x \sim p}\!\left[\log \frac{p(x)}{q(x)}\right]
```

**定理（KL 非负性 / Gibbs 不等式）**：$`D_{\mathrm{KL}}(p \| q) \geq 0`$，等号当且仅当 $`p = q`$（几乎处处）时成立。

**证明**：考虑函数 $`f(t) = -\log t`$，它是严格凸函数（$`f''(t) = 1/t^2 > 0`$ 对 $`t > 0`$）。由 Jensen 不等式（1.5 节），对凸函数 $`f`$ 有 $`f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]`$。令 $`X = q(x)/p(x)`$，期望对 $`p`$ 取：

```math
-\log \mathbb{E}_{x \sim p}\!\left[\frac{q(x)}{p(x)}\right] \leq \mathbb{E}_{x \sim p}\!\left[-\log \frac{q(x)}{p(x)}\right]
```

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

```math
f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]
```

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

```math
f(\boldsymbol{\theta} + \Delta \boldsymbol{\theta}) = f(\boldsymbol{\theta}) + \nabla f(\boldsymbol{\theta})^\top \Delta \boldsymbol{\theta} + o(\|\Delta \boldsymbol{\theta}\|)
```

**二阶泰勒展开**（带余项）：

```math
f(\boldsymbol{\theta} + \Delta \boldsymbol{\theta}) = f(\boldsymbol{\theta}) + \nabla f(\boldsymbol{\theta})^\top \Delta \boldsymbol{\theta} + \frac{1}{2} \Delta \boldsymbol{\theta}^\top \boldsymbol{H}(\boldsymbol{\theta}) \Delta \boldsymbol{\theta} + o(\|\Delta \boldsymbol{\theta}\|^2)
```

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

```math
\boldsymbol{m}_t = \beta_1 \boldsymbol{m}_{t-1} + (1 - \beta_1) \boldsymbol{g}_t \quad \text{（一阶矩估计，即梯度的 EMA）}
```

```math
\boldsymbol{v}_t = \beta_2 \boldsymbol{v}_{t-1} + (1 - \beta_2) \boldsymbol{g}_t^2 \quad \text{（二阶矩估计，即梯度平方的 EMA）}
```

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

```math
\mathbb{E}_{x \sim p_{\boldsymbol{\theta}}}[\nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(x)] = \boldsymbol{0}
```

**证明**：由 $`\int p_{\boldsymbol{\theta}}(x)\,dx = 1`$ 两边对 $`\boldsymbol{\theta}`$ 求导：

```math
\nabla_{\boldsymbol{\theta}} \int p_{\boldsymbol{\theta}}(x)\,dx = \nabla_{\boldsymbol{\theta}} 1 = \boldsymbol{0}
```

在正则条件下交换微分与积分：

```math
\int \nabla_{\boldsymbol{\theta}} p_{\boldsymbol{\theta}}(x)\,dx = \boldsymbol{0}
```

利用恒等式 $`\nabla_{\boldsymbol{\theta}} p_{\boldsymbol{\theta}}(x) = p_{\boldsymbol{\theta}}(x) \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(x)`$（这是链式法则的直接结果：$`\nabla \log p = (\nabla p) / p`$），代入得：

```math
\int p_{\boldsymbol{\theta}}(x) \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(x)\,dx = \mathbb{E}_{x \sim p_{\boldsymbol{\theta}}}[\nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(x)] = \boldsymbol{0}
```

$`\blacksquare`$

量 $`\nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(x)`$ 称为**得分函数（score function）**。该命题说明得分函数在自身分布下的期望为零。直觉上，$`\log p`$ 的梯度指向使概率增大的方向，但正负区域恰好抵消。

<a id="s1-8-2"></a>

### 1.8.2 似然比技巧的核心公式

**定理（似然比技巧 / Log-derivative trick）**：在同样的正则条件下，对任意与 $`\boldsymbol{\theta}`$ 无关的函数 $`f(x)`$：

```math
\nabla_{\boldsymbol{\theta}} \mathbb{E}_{x \sim p_{\boldsymbol{\theta}}}[f(x)] = \mathbb{E}_{x \sim p_{\boldsymbol{\theta}}}[f(x) \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(x)]
```

**证明**：展开期望的定义：

```math
\nabla_{\boldsymbol{\theta}} \mathbb{E}_{x \sim p_{\boldsymbol{\theta}}}[f(x)] = \nabla_{\boldsymbol{\theta}} \int f(x) p_{\boldsymbol{\theta}}(x)\,dx
```

$`f(x)`$ 不依赖 $`\boldsymbol{\theta}`$，故可移入梯度算子内部（再次交换微分与积分）：

```math
= \int f(x) \nabla_{\boldsymbol{\theta}} p_{\boldsymbol{\theta}}(x)\,dx
```

代入 $`\nabla_{\boldsymbol{\theta}} p_{\boldsymbol{\theta}}(x) = p_{\boldsymbol{\theta}}(x) \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(x)`$：

```math
= \int f(x) p_{\boldsymbol{\theta}}(x) \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(x)\,dx = \mathbb{E}_{x \sim p_{\boldsymbol{\theta}}}[f(x) \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(x)]
```

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

```math
\boldsymbol{F}(\boldsymbol{\theta}) = \mathbb{E}_{x \sim p_{\boldsymbol{\theta}}}[\nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(x) \, \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(x)^\top]
```

由于得分函数均值为零（1.8.1 节），这就是其二阶矩矩阵。Fisher 信息矩阵也可以写成 KL 散度的 Hessian：

```math
\boldsymbol{F}(\boldsymbol{\theta}) = \nabla_{\boldsymbol{\theta}} \nabla_{\boldsymbol{\theta}'} D_{\mathrm{KL}}(p_{\boldsymbol{\theta}} \| p_{\boldsymbol{\theta}'}) \big|_{\boldsymbol{\theta}' = \boldsymbol{\theta}}
```

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

参考答案：$`\log \pi_k = z_k - \log \sum_l \exp(z_l)`$。对 $`z_j`$ 求导：$`\partial z_k / \partial z_j = \delta_{kj}`$，$`\partial \log \sum_l \exp(z_l) / \partial z_j = \exp(z_j) / \sum_l \exp(z_l) = \pi_j`$。两者相减得 $`\delta_{kj} - \pi_j`$。

**题目 2**：用 Jensen 不等式证明 $`D_{\mathrm{KL}}(p \| q) \geq 0`$，并指出等号成立的条件。

参考答案：$`-\log`$ 是严格凸函数。由 Jensen 不等式，$`-\log \mathbb{E}_p[q/p] \leq \mathbb{E}_p[-\log(q/p)]`$。左边 $`= -\log 1 = 0`$，右边 $`= D_{\mathrm{KL}}(p\|q)`$。等号当且仅当 $`q/p`$ 为常数（几乎处处），即 $`p = q`$。

**题目 3**：解释为什么 mini-batch SGD 在 RL 中可能产生有偏梯度估计，而在监督学习中通常是无偏的。

参考答案：监督学习中数据集固定，mini-batch 是从固定分布中 i.i.d. 采样的，梯度估计无偏。RL 中数据由当前策略生成，当用同一批数据做多次更新时，参数已变但数据未重新采集，数据不再服从当前策略分布，导致偏差。PPO 通过裁剪比率控制此偏差的影响。

**题目 4**：写出 $`f(\boldsymbol{\theta} + \Delta \boldsymbol{\theta})`$ 的二阶泰勒展开（含余项），并说明为什么它在 TRPO 中至关重要。

参考答案：$`f(\boldsymbol{\theta} + \Delta \boldsymbol{\theta}) = f(\boldsymbol{\theta}) + \nabla f^\top \Delta \boldsymbol{\theta} + \frac{1}{2} \Delta \boldsymbol{\theta}^\top \boldsymbol{H} \Delta \boldsymbol{\theta} + o(\|\Delta \boldsymbol{\theta}\|^2)`$。TRPO 用此展开近似性能改进，并将 Hessian 替换为 Fisher 信息矩阵作为 KL 约束的二次近似，从而导出自然梯度更新方向。没有二阶展开就无法建立 KL 约束与参数更新之间的定量关系。

**题目 5**：证明 $`\mathbb{E}_{x \sim p_{\boldsymbol{\theta}}}[\nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(x)] = \boldsymbol{0}`$，并解释这个结果在策略梯度估计中的作用。

参考答案：由 $`\int p_{\boldsymbol{\theta}} dx = 1`$ 两边对 $`\boldsymbol{\theta}`$ 求导，交换微分积分得 $`\int \nabla p_{\boldsymbol{\theta}} dx = 0`$。代入 $`\nabla p = p \nabla \log p`$ 即得证。该结果表明得分函数均值为零，因此在策略梯度估计 $`\mathbb{E}[R \nabla \log \pi]`$ 中加入与 $`\nabla \log \pi`$ 无关的常数基线 $`b`$ 不改变期望值（因为 $`\mathbb{E}[b \nabla \log \pi] = b \cdot \boldsymbol{0} = 0`$），但可以减小方差。这是基线函数方法的理论基础（见第 3 章）。

---

<a id="chapter-2"></a>

# 第 2 章 强化学习核心概念

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

**折扣因子 $`\gamma`$**：一个标量，满足 $`0 \leq \gamma < 1`$（在无限 horizon 设定下严格要求 $`\gamma < 1`$）。其作用与解释见 2.3 节。

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

对于无限 horizon，$`\tau`$ 是一个无限序列；但在实际推导中，我们通常先处理有限 horizon 再取极限，或直接利用 $`\gamma < 1`$ 保证级数收敛。

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
   该几何级数在 $`\gamma < 1`$ 时收敛，保证了 $`G_t`$ 是有界的。下面给出完整证明。

2. **时间偏好**：$`\gamma < 1`$ 意味着未来的奖励比眼前的奖励「价值更低」。这反映了经济学中的时间偏好：今天的 1 元比明天的 1 元更有价值。$`\gamma`$ 越接近 1，agent 越「有耐心」。

3. **不确定性建模**：可以将 $`\gamma`$ 理解为「环境在每个时间步以概率 $`\gamma`$ 继续运行、以概率 $`1-\gamma`$ 终止」的等价表述。在这种解释下，折扣回报等于未折扣回报在随机终止时间下的期望。

<a id="s2-3-3"></a>

### 2.3.3 有界性证明

**命题**：若 $`|r_t| \leq R_{\max}`$ 对所有 $`t`$ 成立且 $`0 \leq \gamma < 1`$，则 $`|G_t| \leq \dfrac{R_{\max}}{1-\gamma}`$。

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

$`A^\pi(s, a) > 0`$ 表示动作 $`a`$ 优于策略 $`\pi`$ 在该状态的平均表现；$`A^\pi(s, a) < 0`$ 则表示劣于平均。

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

**为什么 $`I - \gamma P^\pi`$ 可逆？** 因为 $`P^\pi`$ 是一个随机矩阵（每行元素非负且和为 1），其谱半径 $`\rho(P^\pi) \leq 1`$。当 $`\gamma < 1`$ 时，$`\gamma P^\pi`$ 的谱半径 $`\rho(\gamma P^\pi) < 1`$，因此 $`I - \gamma P^\pi`$ 的特征值全部不为零（具体地，其特征值为 $`1 - \gamma \lambda_i`$，其中 $`\lambda_i`$ 是 $`P^\pi`$ 的特征值，$`|\lambda_i| \leq 1`$，故 $`|1 - \gamma \lambda_i| \geq 1 - \gamma > 0`$），矩阵可逆。这也保证了 $`V^\pi`$ 是唯一确定的。

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

由于 $`0 \leq \gamma < 1`$，根据 **Banach 不动点定理**（Banach fixed-point theorem），$`T^*`$ 在完备度量空间 $`(\mathbb{R}^{|\mathcal{S}|}, \|\cdot\|_\infty)`$ 上有**唯一**不动点 $`V^*`$，且从任意初始 $`V_0`$ 出发，迭代 $`V_{k+1} = T^* V_k`$ 都以指数速率收敛到 $`V^*`$。这就是**价值迭代（value iteration）** 算法的理论基础。同理，$`T^\pi`$ 也是 $`\gamma`$-压缩映射，保证了 $`V^\pi`$ 的唯一性和策略评估迭代的收敛性。

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
2. **折扣回报** $`G_t`$ 在 $`\gamma < 1`$ 且有界奖励下有界，折扣因子兼具数学收敛性、时间偏好和不确定性三重含义。
3. **价值函数三件套** $`V^\pi, Q^\pi, A^\pi`$ 通过全期望公式相互关联，优势函数的零均值性质 $`\mathbb{E}_{a \sim \pi}[A^\pi(s,a)] = 0`$ 是基线方法和策略梯度的理论支柱。
4. **贝尔曼方程**将价值函数表达为递归不动点；$`T^\pi`$ 和 $`T^*`$ 都是 $`\gamma`$-压缩映射，Banach 不动点定理保证了唯一解和迭代收敛；矩阵形式 $`V^\pi = (I - \gamma P^\pi)^{-1} R^\pi`$ 揭示了可逆性的谱条件。
5. **TD 误差**在 $`V = V^\pi`$ 时是优势函数的无偏估计，在 $`V \neq V^\pi`$ 时是有偏估计；这一事实连接了价值函数近似与策略改进，是 GAE（第 4 章）和 PPO（第 6 章）的核心纽带。

<a id="h88"></a>

### 自测题

**题目 1**：写出有限 horizon $`H`$ 轨迹 $`\tau = (s_0, a_0, r_1, \dots, s_{H-1}, a_{H-1}, r_H)`$ 的概率表达式，并说明为什么环境动力学项与策略项是可分离的。

参考答案：$`p(\tau) = p(s_0) \prod_{t=0}^{H-1} \pi(a_t \mid s_t) P(s_{t+1} \mid s_t, a_t)`$。可分离性来自两个独立性假设：策略选择动作只依赖当前状态（$`\pi(a_t \mid s_t)`$），环境转移只依赖当前状态-动作对（马尔可夫性质 $`P(s_{t+1} \mid s_t, a_t)`$），两者各自只依赖 $`(s_t, a_t)`$，互不耦合。

**题目 2**：证明 $`\mathbb{E}_{a \sim \pi(\cdot \mid s)}[A^\pi(s, a)] = 0`$，并解释这一性质的直观含义。

参考答案：$`\sum_a \pi(a \mid s)[Q^\pi(s,a) - V^\pi(s)] = \sum_a \pi(a \mid s) Q^\pi(s,a) - V^\pi(s) = V^\pi(s) - V^\pi(s) = 0`$。直观含义：优势函数以策略自身的平均表现为基准，正优势和负优势恰好抵消，说明它是「去中心化」的度量。

**题目 3**：为什么矩阵 $`I - \gamma P^\pi`$ 在 $`\gamma < 1`$ 时可逆？

参考答案：$`P^\pi`$ 是随机矩阵，谱半径 $`\rho(P^\pi) \leq 1`$。当 $`\gamma < 1`$ 时，$`\rho(\gamma P^\pi) < 1`$，故 $`I - \gamma P^\pi`$ 的所有特征值 $`1 - \gamma \lambda_i`$ 满足 $`|1 - \gamma \lambda_i| \geq 1 - \gamma > 0`$，没有零特征值，矩阵可逆。

**题目 4**：证明当 $`V = V^\pi`$ 时，$`\mathbb{E}[\delta_t \mid s_t = s, a_t = a] = A^\pi(s, a)`$。如果 $`V \neq V^\pi`$，这个等式还成立吗？为什么不成立？

参考答案：$`\mathbb{E}[r_{t+1} + \gamma V^\pi(s_{t+1}) \mid s, a] = Q^\pi(s,a)`$（由 $`Q^\pi`$ 的贝尔曼方程），减去 $`V^\pi(s)`$ 即得 $`A^\pi(s,a)`$。当 $`V \neq V^\pi`$ 时不成立，因为 $`\mathbb{E}[\gamma V(s_{t+1}) \mid s,a] \neq \gamma \sum_{s'} P(s'|s,a) V^\pi(s')`$，额外产生了 $`V`$ 的近似偏差经环境动力学传播的误差项。

**题目 5**：用自己的话解释为什么强化学习不能像监督学习一样在固定数据集上做梯度下降。这对 PPO 的设计有什么影响？

参考答案：RL 中数据由当前策略生成，更新策略后数据分布随之改变，旧数据不再代表新策略的分布，直接用会导致分布偏移和不稳定。PPO 通过裁剪重要性比率限制单次更新幅度，确保新旧策略足够接近，使旧数据在新策略下仍近似有效，从而允许对同一批数据进行多次小步更新。

---

<a id="chapter-3"></a>

# 第 3 章 策略梯度方法

<a id="s3-1"></a>

## 3.1 优化目标的两种等价形式

在第 2 章中，我们已经建立了马尔可夫决策过程（Markov Decision Process, MDP）的基本框架，并定义了状态值函数 $`V^\pi(s)`$、动作值函数 $`Q^\pi(s,a)`$ 以及优势函数（advantage function）$`A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)`$。本章的核心任务是回答一个问题：**如何直接对策略 $`\pi_\theta`$ 的参数 $`\theta`$ 求梯度，从而用梯度上升改进策略？**

首先明确优化目标。设 $`\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots)`$ 为一条轨迹（trajectory），其累积折扣回报为：

```math
R(\tau) = \sum_{t=0}^{\infty} \gamma^t r_t
```

其中 $`\gamma \in [0,1]`$ 为折扣因子（见第 2 章）。策略 $`\pi_\theta`$ 下的期望回报定义为：

```math
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\big[R(\tau)\big] = \int p_\theta(\tau)\, R(\tau)\, d\tau \tag{3.1}
```

这里 $`p_\theta(\tau)`$ 是策略 $`\pi_\theta`$ 生成的轨迹分布。具体地，若初始状态分布为 $`\rho_0(s_0)`$，环境动力学（transition dynamics）为 $`P(s_{t+1}|s_t, a_t)`$，则：

```math
p_\theta(\tau) = \rho_0(s_0) \prod_{t=0}^{\infty} P(s_{t+1}|s_t, a_t)\, \pi_\theta(a_t|s_t) \tag{3.2}
```

式 (3.1) 是策略梯度的**轨迹形式**，也是后续推导的出发点。

另一种等价写法利用**折扣状态访问分布**（discounted state visitation distribution）$`d^\pi(s)`$，其定义为：

```math
d^\pi(s) = (1-\gamma) \sum_{t=0}^{\infty} \gamma^t \Pr(s_t = s \mid \pi) \tag{3.3}
```

系数 $`(1-\gamma)`$ 使 $`d^\pi(s)`$ 归一化为概率分布（当 $`\gamma < 1`$ 时）。可以证明（推导见第 2 章）：

```math
J(\theta) = \frac{1}{1-\gamma} \mathbb{E}_{s \sim d^{\pi_\theta}}\big[V^{\pi_\theta}(s)\big] \tag{3.4}
```

式 (3.4) 在理论分析中非常方便，但在实际计算策略梯度时，我们通常使用式 (3.1) 的轨迹形式。原因在于：$`d^{\pi_\theta}`$ 本身依赖于 $`\theta`$，对式 (3.4) 直接求梯度会产生 $`\nabla_\theta d^{\pi_\theta}`$ 项，该项涉及整个状态空间的无穷级数，难以处理。而策略梯度定理的精妙之处恰恰在于：**对式 (3.1) 求梯度后，结果恰好可以用 $`d^{\pi_\theta}`$ 下的期望来表达**，从而绕开了对 $`d^{\pi_\theta}`$ 直接求导的困难。

<a id="s3-2"></a>

## 3.2 策略梯度定理的完整推导

本节是本章的核心。我们给出两条等价的推导路径，最终得到同一个结果。

<a id="s3-2-1"></a>

### 3.2.1 路径一：直接展开轨迹分布的梯度

从式 (3.1) 出发，假设积分与微分可交换（正则性条件，要求 $`\pi_\theta(a|s)`$ 关于 $`\theta`$ 连续可微且支撑集不随 $`\theta`$ 变化），有：

```math
\nabla_\theta J(\theta) = \nabla_\theta \int p_\theta(\tau)\, R(\tau)\, d\tau = \int \nabla_\theta p_\theta(\tau)\, R(\tau)\, d\tau \tag{3.5}
```

关键技巧是**乘除 $`p_\theta(\tau)`$**（即似然比技巧 / likelihood ratio trick，其数学基础见第 1 章）：

```math
\nabla_\theta p_\theta(\tau) = p_\theta(\tau) \cdot \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} = p_\theta(\tau) \cdot \nabla_\theta \log p_\theta(\tau) \tag{3.6}
```

代入式 (3.5)：

```math
\nabla_\theta J(\theta) = \int p_\theta(\tau) \cdot \nabla_\theta \log p_\theta(\tau) \cdot R(\tau)\, d\tau = \mathbb{E}_{\tau \sim \pi_\theta}\big[R(\tau) \cdot \nabla_\theta \log p_\theta(\tau)\big] \tag{3.7}
```

现在展开 $`\log p_\theta(\tau)`$。由式 (3.2) 取对数：

```math
\log p_\theta(\tau) = \log \rho_0(s_0) + \sum_{t=0}^{\infty} \Big[\log P(s_{t+1}|s_t, a_t) + \log \pi_\theta(a_t|s_t)\Big] \tag{3.8}
```

对 $`\theta`$ 求梯度时，注意 $`\rho_0(s_0)`$ 和 $`P(s_{t+1}|s_t, a_t)`$ **均不含 $`\theta`$**——它们由环境和初始分布决定，与策略参数无关。这正是 **model-free**（模型无关）方法的关键：我们不需要知道环境动力学的解析形式，也不需要对其求导。因此：

```math
\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{\infty} \nabla_\theta \log \pi_\theta(a_t|s_t) \tag{3.9}
```

将式 (3.9) 代入式 (3.7)，得到 **REINFORCE** 算法的策略梯度估计量：

```math
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[R(\tau) \sum_{t=0}^{\infty} \nabla_\theta \log \pi_\theta(a_t|s_t)\right] \tag{3.10}
```

这就是策略梯度定理的最基本形式。它的含义非常直观：如果某条轨迹的回报 $`R(\tau)`$ 较高，就增大该轨迹上所有动作的概率；反之则减小。

<a id="s3-2-2"></a>

### 3.2.2 路径二：似然比技巧一步得到

第 1 章已经证明了似然比恒等式：对于任意关于 $`\theta`$ 可微的概率密度 $`p_\theta(x)`$ 和函数 $`f(x)`$，只要正则性条件满足，就有：

```math
\nabla_\theta \mathbb{E}_{x \sim p_\theta}[f(x)] = \mathbb{E}_{x \sim p_\theta}\big[f(x) \cdot \nabla_\theta \log p_\theta(x)\big] \tag{3.11}
```

直接将 $`x = \tau`$、$`f(\tau) = R(\tau)`$、$`p_\theta = p_\theta(\tau)`$ 代入式 (3.11)，立即得到式 (3.7)。再将式 (3.9) 代入，同样得到式 (3.10)。

两条路径完全等价。路径一的优势是展示了"为什么环境动力学项消失"这一关键细节；路径二的优势是简洁，且揭示了策略梯度本质上是似然比估计量的一个特例。

<a id="s3-2-3"></a>

### 3.2.3 策略梯度定理的标准形式

式 (3.10) 虽然正确，但还不是最常用的形式。我们将在 3.3 节引入因果性简化后，在 3.4 节将其改写为用 $`Q^\pi`$ 表达的标准形式。

<a id="s3-3"></a>

## 3.3 因果性与 Reward-to-Go

式 (3.10) 中，每个时刻 $`t`$ 的策略梯度 $`\nabla_\theta \log \pi_\theta(a_t|s_t)`$ 被整条轨迹的总回报 $`R(\tau)`$ 加权。然而直觉告诉我们：**时刻 $`t`$ 的动作不可能影响 $`t`$ 之前的奖励**。下面的定理将这一因果性（causality）严格化。

<a id="s3-3-1"></a>

### 3.3.1 因果性定理及其证明

**定理 3.1（因果性）**：对于 $`t' < t`$，有

```math
\mathbb{E}_{\tau \sim \pi_\theta}\big[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot r_{t'}\big] = 0 \tag{3.12}
```

**证明**：利用条件期望的塔性质（tower property / law of iterated expectations）。记 $`\tau_{0:t} = (s_0, a_0, r_0, \dots, s_t, a_t)`$ 为截至时刻 $`t`$ 的历史。注意 $`r_{t'}`$ 在 $`t' < t`$ 时是 $`\tau_{0:t}`$ 的确定性函数，而 $`\nabla_\theta \log \pi_\theta(a_t|s_t)`$ 仅依赖于 $`(s_t, a_t)`$。我们有：

```math
\begin{aligned}
\mathbb{E}_{\tau}\big[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot r_{t'}\big] &= \mathbb{E}_{\tau_{0:t}}\left[r_{t'} \cdot \mathbb{E}_{a_t \sim \pi_\theta(\cdot|s_t)}\big[\nabla_\theta \log \pi_\theta(a_t|s_t) \,\big|\, \tau_{0:t-1}, s_t\big]\right] \\
&= \mathbb{E}_{\tau_{0:t}}\left[r_{t'} \cdot \nabla_\theta \sum_{a_t} \pi_\theta(a_t|s_t)\right] \\
&= \mathbb{E}_{\tau_{0:t}}\left[r_{t'} \cdot \nabla_\theta 1\right] \\
&= 0
\end{aligned} \tag{3.13}
```

第二步利用了 $`s_t`$ 在给定 $`\tau_{0:t-1}`$ 后已由环境动力学确定（$`s_t`$ 是 $`\tau_{0:t-1}`$ 的函数），因此内层期望只对 $`a_t`$ 取；第三步利用了概率归一化 $`\sum_a \pi_\theta(a|s) = 1`$。**证毕。** ∎

这个证明的核心洞察是：$`\nabla_\theta \log \pi_\theta(a|s)`$ 关于 $`a \sim \pi_\theta(\cdot|s)`$ 的期望恒为零向量。这是 score function（得分函数）的基本性质。

<a id="s3-3-2"></a>

### 3.3.2 Reward-to-Go 形式

由因果性定理，式 (3.10) 中 $`R(\tau) = \sum_{k=0}^{\infty} \gamma^k r_k`$ 里所有 $`k < t`$ 的项对第 $`t`$ 步梯度的期望贡献为零，可以安全去掉。定义 **reward-to-go**（未来回报）：

```math
G_t = \sum_{k=t}^{\infty} \gamma^{k-t} r_k \tag{3.14}
```

注意这里 $`G_t`$ 从 $`r_t`$ 开始累加，且以 $`\gamma^{k-t}`$ 而非 $`\gamma^k`$ 加权（即相对于时刻 $`t`$ 重新归一化了折扣）。于是策略梯度可以写为：

```math
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{\infty} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right] \tag{3.15}
```

**关于折扣权重的说明**：有些文献使用 $`\tilde{G}_t = \sum_{k=t}^{\infty} \gamma^k r_k`$（保留全局折扣权重 $`\gamma^k`$），此时式 (3.15) 变为 $`\nabla_\theta J(\theta) = \mathbb{E}[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \tilde{G}_t]`$。这两种写法都是无偏的（因为被去掉的项期望为零），但方差特性略有不同。实践中更常用式 (3.14) 的定义，因为它与 $`Q^\pi(s_t, a_t)`$ 的尺度一致。

Reward-to-go 的意义不仅是降方差：它还将全轨迹回报替换为一个**只依赖当前及未来信息**的量，这使得在线（online）估计成为可能——我们不必等到整条轨迹结束才能计算梯度。

<a id="s3-4"></a>

## 3.4 用 Q 函数表达的策略梯度定理

回忆第 2 章的定义：$`Q^\pi(s,a) = \mathbb{E}_{\tau \sim \pi}[G_t \mid s_t = s, a_t = a]`$。对式 (3.15) 中的期望做条件分解：

```math
\begin{aligned}
\nabla_\theta J(\theta) &= \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{\infty} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right] \\
&= \sum_{t=0}^{\infty} \mathbb{E}_{s_t, a_t}\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \mathbb{E}[G_t \mid s_t, a_t]\right] \\
&= \sum_{t=0}^{\infty} \mathbb{E}_{s_t, a_t}\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot Q^{\pi_\theta}(s_t, a_t)\right]
\end{aligned} \tag{3.16}
```

进一步，$`(s_t, a_t)`$ 的边缘分布正是 $`d^{\pi_\theta}(s) \cdot \pi_\theta(a|s) / (1-\gamma)`$（注意 $`d^{\pi_\theta}`$ 的定义中包含 $`(1-\gamma)`$ 归一化因子）。将所有时间步合并为一个期望，得到**策略梯度定理的标准形式**：

```math
\nabla_\theta J(\theta) = \frac{1}{1-\gamma} \mathbb{E}_{s \sim d^{\pi_\theta},\, a \sim \pi_\theta(\cdot|s)}\Big[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s,a)\Big] \tag{3.17}
```

在实际算法中，常数因子 $`1/(1-\gamma)`$ 通常被吸收进学习率，因此常见的写法省略该因子：

```math
\nabla_\theta J(\theta) \propto \mathbb{E}_{s \sim d^{\pi_\theta},\, a \sim \pi_\theta(\cdot|s)}\Big[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s,a)\Big] \tag{3.18}
```

式 (3.18) 是后续所有策略梯度算法的理论基石。它告诉我们：策略梯度的方向由 $`Q^{\pi_\theta}(s,a)`$ 加权决定——好的动作（$`Q`$ 值高）获得更大的概率提升。

<a id="s3-5"></a>

## 3.5 基线理论与优势函数

式 (3.18) 给出的梯度估计量是无偏的，但方差可能极大。本节介绍**基线（baseline）**方法，这是降低方差而不引入偏差的核心技术。

<a id="s3-5-1"></a>

### 3.5.1 基线的无偏性证明

**定理 3.2**：对于任意只依赖状态 $`s`$ 的函数 $`b(s)`$（称为基线），有

```math
\mathbb{E}_{a \sim \pi_\theta(\cdot|s)}\big[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)\big] = 0 \tag{3.19}
```

**证明**：由于 $`b(s)`$ 不依赖 $`a`$，可以提到期望外面：

```math
b(s) \cdot \mathbb{E}_{a \sim \pi_\theta(\cdot|s)}\big[\nabla_\theta \log \pi_\theta(a|s)\big] = b(s) \cdot \nabla_\theta \sum_a \pi_\theta(a|s) = b(s) \cdot \nabla_\theta 1 = 0 \tag{3.20}
```

**证毕。** ∎

这意味着我们可以将式 (3.18) 中的 $`Q^{\pi_\theta}(s,a)`$ 替换为 $`Q^{\pi_\theta}(s,a) - b(s)`$，梯度期望不变。特别地，取 $`b(s) = V^{\pi_\theta}(s)`$，得到：

```math
\nabla_\theta J(\theta) \propto \mathbb{E}_{s \sim d^{\pi_\theta},\, a \sim \pi_\theta(\cdot|s)}\Big[\nabla_\theta \log \pi_\theta(a|s) \cdot A^{\pi_\theta}(s,a)\Big] \tag{3.21}
```

其中 $`A^{\pi_\theta}(s,a) = Q^{\pi_\theta}(s,a) - V^{\pi_\theta}(s)`$ 就是第 2 章定义的**优势函数**。

<a id="s3-5-2"></a>

### 3.5.2 为什么优势函数能显著降方差

直觉上，$`Q^{\pi}(s,a)`$ 的绝对值可能很大且波动剧烈，而 $`A^{\pi}(s,a)`$ 衡量的是"动作 $`a`$ 相对于平均水平的好坏"。由第 2 章的结论，$`\mathbb{E}_{a \sim \pi}[A^\pi(s,a)] = 0`$，即优势函数以零为中心，正负抵消。这使得梯度估计量的幅度大幅缩小，方差显著降低。

举例来说，如果某个状态下所有动作的 $`Q`$ 值都在 $`[100, 102]`$ 之间，直接用 $`Q`$ 加权会导致梯度被一个大常数主导；而减去 $`V \approx 101`$ 后，优势值在 $`[-1, 1]`$ 之间，梯度信号的信噪比大幅提升。

<a id="s3-5-3"></a>

### 3.5.3 最优基线的推导

既然任意 $`b(s)`$ 都不引入偏差，那么哪个 $`b(s)`$ 使方差最小？考虑单步情形（固定状态 $`s`$），梯度估计量为 $`g = \nabla_\theta \log \pi_\theta(a|s) \cdot (Q^\pi(s,a) - b(s))`$。为简化记号，令 $`\psi = \nabla_\theta \log \pi_\theta(a|s)`$，$`Q = Q^\pi(s,a)`$，则：

```math
\text{Var}[g] = \mathbb{E}_a\big[\|\psi \cdot (Q - b)\|^2\big] - \big\|\mathbb{E}_a[\psi \cdot (Q - b)]\big\|^2 \tag{3.22}
```

第二项等于 $`\|\nabla_\theta J\|^2`$，与 $`b`$ 无关（因为基线不改变期望）。因此最小化方差等价于最小化第一项。对 $`b`$ 求导并令其为零：

```math
\frac{\partial}{\partial b} \mathbb{E}_a\big[\|\psi\|^2 (Q - b)^2\big] = -2\, \mathbb{E}_a\big[\|\psi\|^2 (Q - b)\big] = 0 \tag{3.23}
```

解得**最优基线**：

```math
b^*(s) = \frac{\mathbb{E}_{a \sim \pi}\big[\|\nabla_\theta \log \pi_\theta(a|s)\|^2 \cdot Q^\pi(s,a)\big]}{\mathbb{E}_{a \sim \pi}\big[\|\nabla_\theta \log \pi_\theta(a|s)\|^2\big]} \tag{3.24}
```

注意 $`b^*(s)`$ **不是** $`V^\pi(s)`$，而是被 $`\|\nabla_\theta \log \pi_\theta(a|s)\|^2`$ 加权的 $`Q`$ 值平均。只有当 $`\|\nabla_\theta \log \pi_\theta(a|s)\|^2`$ 对所有 $`a`$ 近似相等时，$`b^*(s) \approx V^\pi(s)`$。尽管如此，实践中几乎总是使用 $`V^\pi(s)`$ 作为基线，原因是：(1) $`V^\pi(s)`$ 容易通过价值网络估计；(2) 最优基线需要额外估计 $`\|\nabla \log \pi\|^2`$，计算开销大且自身有估计误差；(3) $`V^\pi(s)`$ 在实践中已经足够有效。

<a id="s3-6"></a>

## 3.6 策略梯度的偏差-方差分解框架

为了统一理解后续各种策略梯度变体（包括第 4 章的 GAE），我们建立一个一般性的分解框架。

将策略梯度估计量写成如下通用形式：

```math
\hat{g} = \sum_{t=0}^{\infty} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t \tag{3.25}
```

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

<a id="s3-7"></a>

## 3.7 Actor-Critic 架构

策略梯度定理告诉我们梯度方向由 $`Q^\pi`$ 或 $`A^\pi`$ 决定，但这些量本身是未知的。**Actor-Critic** 架构的核心思想是：用一个可学习的价值网络（critic）来估计这些量，同时用策略网络（actor）沿估计的梯度方向更新。

<a id="s3-7-1"></a>

### 3.7.1 两个网络的分工

- **Actor（策略网络）**$`\pi_\theta(a|s)`$：参数为 $`\theta`$，输出动作概率分布。按式 (3.21) 更新：
  ```math
  \theta \leftarrow \theta + \alpha_\theta \cdot \nabla_\theta \log \pi_\theta(a|s) \cdot \hat{A}(s,a) \tag{3.26}
  ```

- **Critic（价值网络）**$`V_\phi(s)`$：参数为 $`\phi`$，估计状态值函数 $`V^{\pi_\theta}(s)`$。用于提供基线和/或优势估计。

<a id="s3-7-2"></a>

### 3.7.2 Critic 的损失函数

Critic 的目标是准确估计 $`V^{\pi_\theta}(s)`$。最直接的损失函数是最小化均方误差：

```math
L_V(\phi) = \mathbb{E}_{s \sim d^{\pi_\theta}}\Big[\big(V_\phi(s) - V^{\pi_\theta}(s)\big)^2\Big] \tag{3.27}
```

但 $`V^{\pi_\theta}(s)`$ 本身未知。实践中用以下替代目标之一：

1. **Monte Carlo 目标**：$`L_V(\phi) = \mathbb{E}[(V_\phi(s_t) - G_t)^2]`$，其中 $`G_t`$ 是从 $`s_t`$ 出发的实际回报。无偏但方差高。

2. **TD 目标**：$`L_V(\phi) = \mathbb{E}[(V_\phi(s_t) - (r_t + \gamma V_\phi(s_{t+1})))^2]`$。注意这里目标中也包含 $`V_\phi`$ 自身，因此这是一个**半梯度**（semi-gradient）方法——对目标中的 $`V_\phi(s_{t+1})`$ 不反向传播梯度。这种做法引入了偏差（bootstrap bias），但大幅降低了方差，且允许在线更新。

3. **$`k`$-step TD 目标**：介于上述两者之间，用 $`k`$ 步实际回报加上 $`V_\phi(s_{t+k})`$ 作为目标。

<a id="s3-7-3"></a>

### 3.7.3 为什么两个网络的梯度必须分开

一个常见疑问是：能否把 actor 和 critic 的损失合并为一个 loss 一起反传？**不能**，原因有二：

1. **目标不同**：Actor 要最大化 $`J(\theta)`$，critic 要最小化 $`L_V(\phi)`$。两者的优化方向不一致，合并后会互相干扰。

2. **梯度耦合导致不稳定**：如果在 actor 的梯度中也对 $`\phi`$ 求导（或反过来），会产生交叉项 $`\nabla_\theta \nabla_\phi L`$，这相当于在联合参数空间上做二阶优化，极易发散。

因此标准做法是：**actor 和 critic 各自维护独立的优化器和梯度计算图**，仅在数据层面共享采样轨迹。

<a id="s3-7-4"></a>

### 3.7.4 Critic 误差对 Actor 的影响

当 $`V_\phi \neq V^{\pi_\theta}`$ 时，优势估计 $`\hat{A} = Q_{\text{est}} - V_\phi`$ 是有偏的。这种偏差会通过式 (3.26) 传导给 actor，导致策略更新偏离真实梯度方向。这是 Actor-Critic 方法的固有代价：我们用 critic 的估计误差换取了方差的降低。理论上，当 critic 收敛到真实 $`V^{\pi_\theta}`$ 时偏差消失；实践中，critic 永远只是近似，因此需要在偏差和方差之间找到平衡点。

<a id="s3-8"></a>

## 3.8 n-step 方法与 A2C/A3C

<a id="s3-8-1"></a>

### 3.8.1 A3C 的异步多 Worker 动机

策略梯度方法的一个根本问题是**样本相关性**：同一条轨迹中的连续样本高度相关，导致梯度估计的方差增大、训练不稳定。在 DQN 等 off-policy 方法中，这个问题通过经验回放（experience replay）打破相关性。但在 on-policy 策略梯度方法中，经验回放不可直接使用——因为回放的数据是由旧策略采集的，而策略梯度要求数据来自当前策略（off-policy 修正需要重要性采样，见第 5 章）。

**A3C（Asynchronous Advantage Actor-Critic）** 提出了一种巧妙的替代方案：启动多个并行的 worker，每个 worker 独立与环境交互并计算梯度，异步地将梯度累加到一个共享的全局参数上。并行采样的效果类似于经验回放——不同 worker 处于轨迹的不同阶段、不同状态，汇总后的梯度近似于从混合分布中采样，有效打破了样本相关性。

<a id="s3-8-2"></a>

### 3.8.2 A2C：同步版本

**A2C（Advantage Actor-Critic）** 是 A3C 的同步变体：所有 worker 并行采集固定长度的轨迹片段，然后同步地汇总梯度并更新全局参数，再广播新参数给所有 worker。

实践中 A2C 往往优于 A3C，原因是：(1) 同步更新避免了过时梯度（stale gradient）问题；(2) 可以使用更大的批量（batch size），提高 GPU 利用率；(3) 实现更简单，调试更容易。A3C 的异步优势在现代硬件上并不明显，因为 GPU 并行已经取代了 CPU 多核并行的角色。

<a id="s3-8-3"></a>

### 3.8.3 两者的等价性讨论

从理论上看，A2C 和 A3C 都是对同一个策略梯度目标 (3.21) 的随机近似。区别仅在于梯度估计的聚合方式：A3C 是异步随机梯度下降（asynchronous SGD），A2C 是小批量同步 SGD。在凸优化中两者收敛性有经典结果；在非凸的策略优化中，同步版本通常更稳定。

<a id="s3-9"></a>

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

<a id="s3-10"></a>

## 3.10 策略梯度方法的固有缺陷

尽管策略梯度方法在理论上优雅且通用，但它有三个严重的实践缺陷，这些缺陷直接催生了后续的 TRPO（第 5 章）和 PPO（第 6 章）。

<a id="s3-10-1"></a>

### 3.10.1 样本效率极低

策略梯度是严格的 **on-policy** 方法：每条轨迹只能用于产生它的那个策略的梯度估计。一旦 $`\theta`$ 更新，旧轨迹就不再是当前策略下的无偏样本，不能复用。这意味着每次梯度更新都需要重新采样，样本效率远低于 off-policy 方法（如 DQN 的经验回放可以反复使用同一批数据）。

<a id="s3-10-2"></a>

### 3.10.2 对步长极其敏感

这是策略梯度最致命的问题。在监督学习中，即使学习率过大导致一步走偏，下一轮仍然可以用同样的训练数据纠正回来。但在策略梯度中，**数据分布本身随策略改变**。如果一步更新太大，新策略可能变得很差（例如在某些状态下给灾难性动作分配了高概率），而从这个差策略采集的数据只会进一步强化错误行为，形成恶性循环。这种崩溃往往是**不可恢复的**。

反过来，如果步长太小，收敛速度极慢，在高维连续控制问题中可能需要数百万步才能学到有意义的策略。

这就引出了一个核心矛盾：**我们需要一种机制，既能保证每步更新的改善幅度足够大以加速收敛，又能防止更新过大导致策略崩溃**。TRPO 通过约束 KL 散度来解决这个问题（第 5 章），PPO 通过裁剪（clip）目标函数来实现类似效果（第 6 章）。

<a id="s3-10-3"></a>

### 3.10.3 高方差

即使使用了 reward-to-go 和基线，策略梯度估计量的方差仍然远高于监督学习中的梯度。这是因为梯度估计依赖于随机采样的轨迹，而轨迹本身的随机性（来自策略和环境）会放大估计的不确定性。高方差意味着需要更多样本才能获得可靠的梯度方向，进一步加剧了样本效率问题。

这三个缺陷共同构成了 PPO 存在的直接理由：PPO 通过裁剪机制限制了每步更新的幅度（解决步长敏感性），通过多 epoch 重用数据提高了样本效率（配合裁剪保证安全性），并通过优势函数的标准化等手段降低了方差。这些解决方案的理论基础将在第 5、6 章详细展开。

<a id="s3-11"></a>

## 3.11 本章小结与自测

<a id="h119"></a>

### 要点回顾

1. **策略梯度定理**给出了 $`\nabla_\theta J(\theta)`$ 的精确表达式，推导有两条等价路径：直接展开轨迹分布梯度和似然比技巧。核心结果是 $`\nabla J \propto \mathbb{E}[\nabla \log \pi_\theta(a|s) \cdot Q^\pi(s,a)]`$。

2. **因果性**允许将全轨迹回报 $`R(\tau)`$ 替换为 reward-to-go $`G_t`$，去除过去奖励对当前梯度的噪声贡献，且不引入偏差。

3. **基线**$`b(s)`$ 减去后不改变梯度期望，最优基线被 $`\|\nabla \log \pi\|^2`$ 加权，实践中用 $`V^\pi(s)`$ 近似。减去 $`V^\pi`$ 后自然引出优势函数 $`A^\pi = Q^\pi - V^\pi`$。

4. **Actor-Critic** 用价值网络估计优势函数，critic 和 actor 必须独立优化；critic 的估计误差会给 actor 引入偏差，这是偏差-方差权衡的体现。

5. **策略梯度的三大缺陷**（样本效率低、步长敏感、高方差）是 TRPO 和 PPO 被提出的直接动机。

<a id="h120"></a>

### 自测题

**题目 1**：在策略梯度推导中，为什么 $`\nabla_\theta \log P(s_{t+1}|s_t, a_t) = 0`$？这对 model-free 方法意味着什么？

参考答案：$`P(s_{t+1}|s_t, a_t)`$ 是环境动力学，不含策略参数 $`\theta`$，因此其对 $`\theta`$ 的梯度为零。这意味着策略梯度方法不需要知道环境的转移概率模型，只需通过与环境交互采样轨迹即可计算梯度，这是 model-free 方法的核心特征。

**题目 2**：证明 $`\mathbb{E}_{a \sim \pi_\theta(\cdot|s)}[\nabla_\theta \log \pi_\theta(a|s)] = 0`$，并说明这个性质在基线理论和因果性证明中分别扮演什么角色。

参考答案：$`\mathbb{E}_a[\nabla \log \pi(a|s)] = \sum_a \pi(a|s) \cdot \nabla \log \pi(a|s) = \nabla \sum_a \pi(a|s) = \nabla 1 = 0`$。在基线理论中，它保证了 $`b(s) \cdot \mathbb{E}[\nabla \log \pi] = 0`$，即基线不引入偏差。在因果性证明中，它保证了 $`\mathbb{E}[\nabla \log \pi(a_t|s_t) \cdot r_{t'}] = 0`$（$`t'<t`$），因为内层期望对 $`a_t`$ 取时 $`r_{t'}`$ 是常数，提出来后乘以零向量得零。

**题目 3**：最优基线 $`b^*(s)`$ 与 $`V^\pi(s)`$ 何时相等？为什么实践中仍使用 $`V^\pi(s)`$？

参考答案：当 $`\|\nabla_\theta \log \pi_\theta(a|s)\|^2`$ 对所有 $`a`$ 为常数时，$`b^*(s) = V^\pi(s)`$。实践中使用 $`V^\pi(s)`$ 是因为：最优基线需要额外估计 $`\|\nabla \log \pi\|^2`$ 的期望，计算成本高且自身有估计噪声；$`V^\pi(s)`$ 可通过 critic 网络自然获得；经验表明 $`V^\pi(s)`$ 已能有效降低方差。

**题目 4**：解释为什么策略梯度方法不能像监督学习那样对同一批数据跑多个 epoch。这与 PPO 的设计有什么关系？

参考答案：策略梯度是 on-policy 的，梯度估计要求数据来自当前策略。更新 $`\theta`$ 后数据分布改变，旧数据不再提供当前策略的无偏梯度，强行复用会导致偏差甚至策略崩溃。PPO 通过裁剪目标函数限制了新旧策略之间的比率偏离程度，使得在一定范围内安全地重用数据成为可能，从而提高了样本效率。详见第 6 章。

**题目 5**：在 Actor-Critic 中，如果用 $`r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)`$ 作为优势估计，这个估计量是有偏还是无偏的？偏差来源是什么？

参考答案：有偏。偏差来源于两个方面：(1) $`V_\phi`$ 是 $`V^{\pi_\theta}`$ 的近似，$`V_\phi \neq V^{\pi_\theta}`$ 导致 bootstrap 偏差；(2) TD 目标 $`r_t + \gamma V_\phi(s_{t+1})`$ 本身是 $`Q^{\pi_\theta}(s_t, a_t)`$ 的有偏估计（除非 $`V_\phi = V^{\pi_\theta}`$）。但当 $`V_\phi`$ 足够准确时，偏差很小，而方差的降低使得总体均方误差反而更小。

---

<a id="chapter-4"></a>

# 第 4 章 优势估计与广义优势估计（GAE）

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

令 $`k \to \infty`$，假设 episode 无限长或适当处理终止条件，由于 $`\gamma < 1`$，$`\gamma^k V(s_{t+k}) \to 0`$（在 $`V`$ 有界的条件下），我们得到：

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
| $`k=1`$ | 最大（$`\propto \gamma \|V - V^\pi\|`$） | 最小 | 最高（每步都可更新） | 最强 | 纯 TD，完全信任 critic |
| $`k=2`$ | 较大（$`\propto \gamma^2 \|V - V^\pi\|`$） | 较小 | 高 | 强 | 两步真实奖励 + bootstrap |
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

现在需要交换两个求和的顺序。注意内层求和的条件是 $`0 \leq l \leq k-1`$，即 $`l < k`$，等价于 $`k > l`$，即 $`k \geq l+1`$。因此交换后外层对 $`l`$ 从 $`0`$ 到 $`\infty`$ 求和，内层对 $`k`$ 从 $`l+1`$ 到 $`\infty`$ 求和：

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

在实际训练中，rollout 总是有限长度的。当 rollout 长度为 $`T`$ 时，递推形式计算的实际上是截断级数 $`\sum_{l=0}^{T-t-1} (\gamma\lambda)^l \delta_{t+l}^V`$，而非无穷级数。但由于 $`\gamma\lambda < 1`$，被截断的尾部贡献以 $`(\gamma\lambda)^{T-t}`$ 的速率衰减，在 $`T`$ 足够大时可以忽略。这也是为什么 PPO 等算法通常使用固定长度的 rollout 片段（如 $`T=2048`$）而非完整 episode。

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

参考答案：$`\gamma V(s_{t+1})`$ 与 $`-\gamma V(s_{t+1})`$ 抵消。在一般情形下，这对应于 telescoping sum（裂项求和）结构：相邻项中的 $`V(s_{t+l})`$ 正负配对消去，最终只剩首尾两项 $`-V(s_t)`$ 和 $`\gamma^k V(s_{t+k})`$。

**题目 2：** 如果 critic 网络完美学习了真实价值函数（$`V = V^\pi`$），那么对于任意 $`k`$，$`\hat{A}_t^{(k)}`$ 的偏差是多少？此时选择不同 $`k`$ 还有意义吗？

参考答案：偏差为零（对所有 $`k`$ 都是无偏估计）。但仍有意义，因为不同 $`k`$ 的方差不同。$`k=1`$ 方差最小，是最优选择。只有在 $`V \neq V^\pi`$ 时才需要通过增大 $`k`$ 来降低偏差。

**题目 3：** 在 GAE 的推导中，交换求和顺序后内层级数 $`\sum_{k=l+1}^{\infty} \lambda^{k-1}`$ 的结果是什么？它与外层的 $`(1-\lambda)`$ 因子有什么关系？

参考答案：结果为 $`\lambda^l / (1-\lambda)`$。它与外层的 $`(1-\lambda)`$ 恰好约去，使得最终的 GAE 表达式中不再显式出现 $`(1-\lambda)`$ 因子，只剩下 $`(\gamma\lambda)^l`$ 的简洁形式。

**题目 4：** 在一个最大步数为 200 的环境中，如果在第 200 步发生了截断（truncation），计算该步的 $`\delta_t^V`$ 时是否应该包含 $`\gamma V(s_{t+1})`$？为什么？

参考答案：应该包含。截断不是真正的终止，MDP 在该状态下仍有后续价值。不包含 bootstrap 会导致系统性地低估该状态的价值，引入不必要的偏差。只有真正终止（done=True 且非截断）才应省略 bootstrap 项。

**题目 5：** PPO 原论文使用 $`\gamma=0.99, \lambda=0.95`$。如果将 $`\lambda`$ 改为 0，GAE 退化为什么？这对训练的偏差和方差分别有什么影响？如果改为 1 呢？

参考答案：$`\lambda=0`$ 退化为 $`\hat{A}_t^{(1)} = \delta_t^V`$（纯 TD 优势），偏差最大、方差最小，完全依赖 critic 质量。$`\lambda=1`$ 退化为蒙特卡洛优势 $`\sum \gamma^l r_{t+l} - V(s_t)`$，偏差为零、方差最大，critic 仅作基线。前者在 critic 不准确时可能导致策略朝错误方向更新，后者在稀疏奖励环境下可能导致训练极慢。

---

<a id="chapter-5"></a>

# 第 5 章 信任域方法与重要性采样

<a id="s5-1"></a>

## 5.1 引言：为什么需要限制策略更新幅度

在第 3 章中，我们推导了策略梯度定理并给出了 REINFORCE 算法。然而该算法存在一个致命缺陷：对步长极度敏感。步长过大时，新策略可能跌入低回报区域且无法恢复；步长过小时，收敛速度不可接受。这一困境的根源在于：策略梯度只提供了目标函数 $`J(\theta)`$ 在 $`\theta`$ 处的**一阶局部信息**，而我们对 $`J(\theta)`$ 的全局几何一无所知。参数空间中的欧氏距离 $`\|\Delta\theta\|`$ 并不能反映新旧策略在**分布意义**上的真实差异——同一个 $`\Delta\theta`$ 在不同参数化下可能导致截然不同的策略变化。

本章的核心任务是建立一套数学框架，使得我们能够在保证「策略性能单调不降」的前提下尽可能大地更新策略。这条路线最终导向 TRPO（Trust Region Policy Optimization）算法，并为第 6 章 PPO 的设计提供理论基础。为此，我们需要依次解决以下问题：如何用旧策略的样本评估新策略（重要性采样）、新旧策略的性能差距如何精确表达（性能差异引理）、如何将该差距转化为可计算的优化约束（单调改进界）、以及如何在高维参数空间中高效求解该约束优化（自然梯度与共轭梯度）。

<a id="s5-2"></a>

## 5.2 重要性采样的严格推导

<a id="s5-2-1"></a>

### 5.2.1 基本恒等式与无偏性

设我们希望计算期望 $`\mathbb{E}_{x \sim q}[f(x)]`$，但只能从另一个分布 $`p`$ 中获取样本。假设 $`p`$ 的支撑覆盖 $`q`$ 的支撑，即对所有满足 $`q(x) > 0`$ 的 $`x`$ 都有 $`p(x) > 0`$。定义**重要性比（importance ratio）** 为：

```math
r(x) = \frac{q(x)}{p(x)}.
```

则有如下恒等式：

```math
\mathbb{E}_{x \sim p}\left[r(x) f(x)\right] = \int p(x) \cdot \frac{q(x)}{p(x)} \cdot f(x) \, dx = \int q(x) f(x) \, dx = \mathbb{E}_{x \sim q}[f(x)].
```

上述推导的关键步骤是在积分中乘除 $`q(x)/p(x)`$，然后交换测度。这要求 $`p(x) > 0`$ 在 $`q(x) > 0`$ 的所有点上成立；若存在 $`q(x) > 0`$ 但 $`p(x) = 0`$ 的点，则该点贡献被遗漏，估计量产生系统性偏差。因此在支撑条件满足时，IS 估计量是**严格无偏**的。

给定 $`N`$ 个独立同分布样本 $`x_i \sim p`$，IS 估计量为：

```math
\hat{\mu}_N = \frac{1}{N} \sum_{i=1}^{N} r(x_i) f(x_i).
```

由上述恒等式，$`\mathbb{E}[\hat{\mu}_N] = \mathbb{E}_{x \sim q}[f(x)]`$，无偏性得证。

<a id="s5-2-2"></a>

### 5.2.2 方差爆炸与有效样本量

虽然 IS 估计量无偏，其方差却可能极大甚至无穷。展开方差表达式：

```math
\text{Var}_p\left[r(X)f(X)\right] = \mathbb{E}_p\left[r(X)^2 f(X)^2\right] - \left(\mathbb{E}_q[f(X)]\right)^2 = \int \frac{q(x)^2}{p(x)} f(x)^2 \, dx - \mu_q^2.
```

当 $`q/p`$ 在某些区域取值很大且 $`p`$ 在这些区域的概率密度很小时，被积函数 $`q^2/p`$ 可以发散，导致方差无界。直观地说，如果行为策略 $`p`$ 很少访问目标策略 $`q`$ 认为重要的区域，那么偶尔落入这些区域的样本将获得极大的权重 $`r(x)`$，使估计量剧烈波动。这就是**方差爆炸**问题，也是后续所有「限制策略更新幅度」方法的根本动机。

实践中常用**有效样本量（Effective Sample Size, ESS）** 作为诊断指标：

```math
\text{ESS} = \frac{\left(\sum_{i=1}^{N} w_i\right)^2}{\sum_{i=1}^{N} w_i^2},
```

其中 $`w_i = r(x_i)`$ 为非归一化权重。ESS 的取值范围为 $`[1, N]`$：当所有权重相等时 $`\text{ESS} = N`$（理想情况），当权重集中于单个样本时 $`\text{ESS} \approx 1`$（退化情况）。ESS 远小于 $`N`$ 表明 IS 估计不可靠。

<a id="s5-3"></a>

## 5.3 策略层面的重要性采样

<a id="s5-3-1"></a>

### 5.3.1 轨迹比率与单步比率

在强化学习中，我们希望用旧策略 $`\pi_{\theta_\text{old}}`$ 采集的轨迹来估计新策略 $`\pi_\theta`$ 的目标函数或梯度。最直接的想法是使用完整轨迹的重要性比：

```math
\rho(\tau) = \prod_{t=0}^{T-1} \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}.
```

然而这个乘积的方差随时间步 $`T`$ **指数级增长**。粗略分析如下：若每步比率 $`r_t`$ 独立且 $`\text{Var}(r_t) = \sigma^2`$，则 $`\text{Var}(\prod_t r_t) \approx (1+\sigma^2)^T - 1`$，呈指数增长。即使各步比率不完全独立，在实际 MDP 中这种指数趋势依然显著。因此轨迹级别的 IS 在 horizon 稍大时完全不可用。

<a id="s5-3-2"></a>

### 5.3.2 为什么单步比率就够了

幸运的是，策略梯度定理（见第 3 章）告诉我们：

```math
\nabla J(\theta) = \mathbb{E}_{s \sim d^\pi, a \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^\pi(s,a)\right].
```

注意这里的状态分布是 $`d^\pi`$，即当前策略 $`\pi_\theta`$ 自身的稳态（或折扣加权）状态分布。当我们用 $`\pi_{\theta_\text{old}}`$ 的样本来估计上式时，只需对动作采样做修正：

```math
\mathbb{E}_{s \sim d^{\pi_\text{old}}, a \sim \pi_{\theta_\text{old}}}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_\text{old}}(a|s)} \nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\text{old}}(s,a)\right].
```

这里只出现了**单步比率** $`r_t(\theta) = \pi_\theta(a_t|s_t)/\pi_{\theta_\text{old}}(a_t|s_t)`$，避免了轨迹比率的指数方差。

但这引入了一个关键近似：我们用 $`d^{\pi_\text{old}}`$ 替代了 $`d^{\pi_\theta}`$。当 $`\theta`$ 偏离 $`\theta_\text{old}`$ 较远时，两个状态分布的差异不可忽视。这正是**性能差异引理**要解决的问题——它精确刻画了状态分布失配带来的误差。

<a id="s5-4"></a>

## 5.4 性能差异引理

<a id="s5-4-1"></a>

### 5.4.1 引理陈述

**性能差异引理**（Kakade & Langford, 2002）给出了任意两个策略 $`\pi`$ 和 $`\pi'`$ 之间期望回报差的精确表达式：

```math
J(\pi') - J(\pi) = \mathbb{E}_{s \sim d^{\pi'}}\left[\sum_{a} \pi'(a|s) A^\pi(s,a)\right],
```

其中 $`d^{\pi'}`$ 是新策略 $`\pi'`$ 的折扣加权状态访问频率，$`A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)`$ 是旧策略 $`\pi`$ 的优势函数。

<a id="s5-4-2"></a>

### 5.4.2 完整推导

我们从回报的定义出发。记 $`G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}`$ 为从时刻 $`t`$ 开始的折扣回报。对于策略 $`\pi'`$ 产生的轨迹，有：

```math
V^{\pi'}(s_0) = \mathbb{E}_{\tau \sim \pi'}[G_0 | s_0].
```

考虑沿 $`\pi'`$ 轨迹的 telescoping 分解。注意到：

```math
V^{\pi'}(s_t) = \mathbb{E}_{\tau \sim \pi'}[G_t | s_t],
```

因此 $`G_t - V^{\pi'}(s_t)`$ 是一个均值为零的鞅差序列。但更有用的是将 $`V^\pi`$ 而非 $`V^{\pi'}`$ 插入 telescoping：

```math
V^{\pi'}(s_0) - V^\pi(s_0) = \mathbb{E}_{\tau \sim \pi'}\left[\sum_{t=0}^{\infty} \gamma^t \left(r_t + \gamma V^\pi(s_{t+1}) - V^\pi(s_t)\right)\right].
```

**验证此式**：展开右边求和，得到

```math
\mathbb{E}_{\tau \sim \pi'}\left[\sum_{t=0}^{\infty} \gamma^t r_t + \sum_{t=0}^{\infty} \gamma^{t+1} V^\pi(s_{t+1}) - \sum_{t=0}^{\infty} \gamma^t V^\pi(s_t)\right].
```

后两项构成 telescoping 级数：$`\sum_{t=0}^{\infty} \gamma^{t+1} V^\pi(s_{t+1}) - \sum_{t=0}^{\infty} \gamma^t V^\pi(s_t) = -V^\pi(s_0)`$（因为 $`\lim_{T\to\infty} \gamma^T V^\pi(s_T) = 0`$，假定 $`\gamma < 1`$ 且 $`V^\pi`$ 有界）。因此右边等于 $`\mathbb{E}_{\tau \sim \pi'}[G_0] - V^\pi(s_0) = V^{\pi'}(s_0) - V^\pi(s_0)`$，恒等式成立。

接下来，注意到括号内的表达式正是 TD 误差 $`\delta_t = r_t + \gamma V^\pi(s_{t+1}) - V^\pi(s_t)`$，而由第 2 章结论，$`\mathbb{E}[\delta_t | s_t, a_t] = A^\pi(s_t, a_t)`$。因此：

```math
V^{\pi'}(s_0) - V^\pi(s_0) = \mathbb{E}_{\tau \sim \pi'}\left[\sum_{t=0}^{\infty} \gamma^t A^\pi(s_t, a_t)\right].
```

对初始状态 $`s_0 \sim \rho_0`$ 取期望，并将轨迹期望改写为状态-动作边际分布的形式：

```math
J(\pi') - J(\pi) = \sum_{t=0}^{\infty} \gamma^t \mathbb{E}_{s_t \sim d_t^{\pi'}, a_t \sim \pi'}\left[A^\pi(s_t, a_t)\right] = \mathbb{E}_{s \sim d^{\pi'}}\left[\sum_a \pi'(a|s) A^\pi(s,a)\right],
```

其中 $`d^{\pi'}(s) = (1-\gamma)\sum_{t=0}^{\infty} \gamma^t d_t^{\pi'}(s)`$ 为折扣加权状态分布（归一化为概率分布）。引理得证。

<a id="s5-4-3"></a>

### 5.4.3 关键观察

引理右侧使用的是 **$`d^{\pi'}`$（新策略的状态分布）** 而非 $`d^\pi`$。这意味着要精确计算性能差异，我们需要知道新策略会访问哪些状态——而这恰恰是我们尚未执行的策略。这是 off-policy 评估的根本困难，也是下一节推导单调改进界的出发点。

<a id="s5-5"></a>

## 5.5 TRPO 的单调改进界

本节是本章核心，遵循 Schulman et al. (2015) 的推导路径。

<a id="s5-5-1"></a>

### 5.5.1 第一步：将 $`d^{\pi'}`$ 替换为 $`d^\pi`$ 并量化误差

由性能差异引理：

```math
J(\pi') - J(\pi) = \mathbb{E}_{s \sim d^{\pi'}}\left[\sum_a \pi'(a|s) A^\pi(s,a)\right].
```

我们希望将其中的 $`d^{\pi'}`$ 替换为可计算的 $`d^\pi`$。为此引入替代目标：

```math
\eta(\pi') = \mathbb{E}_{s \sim d^{\pi}}\left[\sum_a \pi'(a|s) A^\pi(s,a)\right].
```

注意 $`\eta(\pi')`$ 只涉及旧策略的状态分布 $`d^\pi`$，可以用旧策略采集的样本通过重要性采样估计。定义近似目标函数：

```math
L_\pi(\pi') = J(\pi) + \eta(\pi').
```

现在量化 $`J(\pi') - L_\pi(\pi')`$ 的误差。由性能差异引理和 $`\eta`$ 的定义：

```math
J(\pi') - L_\pi(\pi') = \mathbb{E}_{s \sim d^{\pi'}}\left[\sum_a \pi'(a|s) A^\pi(s,a)\right] - \mathbb{E}_{s \sim d^{\pi}}\left[\sum_a \pi'(a|s) A^\pi(s,a)\right].
```

记 $`f(s) = \sum_a \pi'(a|s) A^\pi(s,a)`$，则误差为 $`(d^{\pi'} - d^\pi)^T f`$。Schulman et al. (2015) 证明了以下关键不等式：

```math
\left|J(\pi') - L_\pi(\pi')\right| \leq C \cdot \max_s D_\text{KL}\left(\pi'(\cdot|s) \| \pi(\cdot|s)\right),
```

其中 $`C = \frac{4\varepsilon\gamma}{(1-\gamma)^2}`$，$`\varepsilon = \max_{s,a} |A^\pi(s,a)|`$。

**$`C`$ 的来历**：该常数来自将状态分布差异 $`\|d^{\pi'} - d^\pi\|_1`$ 用逐状态 KL 散度控制的推导链。具体地，(i) 由 Pinsker 不等式，$`\|d^{\pi'} - d^\pi\|_1 \leq \sqrt{2 D_\text{KL}(d^{\pi'} \| d^\pi)}`$；(ii) 折扣 MDP 中状态分布的 KL 散度可以被逐状态策略 KL 的加权和所界定，权重涉及 $`\gamma/(1-\gamma)`$；(iii) 结合 $`|f(s)| \leq \max_a |A^\pi(s,a)| \leq \varepsilon`$ 以及额外的 $`\gamma/(1-\gamma)`$ 因子（来自双重求和中折扣因子的累积），最终得到 $`C = 4\varepsilon\gamma/(1-\gamma)^2`$。完整的常数推导较为冗长，核心直觉是：$`\gamma`$ 接近 1 时，策略微小差异经过长时间累积会被放大，$`(1-\gamma)^{-2}`$ 反映了这种累积效应的二阶性质。

<a id="s5-5-2"></a>

### 5.5.2 第二步：单调改进下界

由上述误差界，直接得到：

```math
J(\pi') \geq L_\pi(\pi') - C \cdot \max_s D_\text{KL}\left(\pi'(\cdot|s) \| \pi(\cdot|s)\right).
```

记右端为 $`\underline{J}_\pi(\pi')`$，它是 $`J(\pi')`$ 的一个**可计算下界**。

<a id="s5-5-3"></a>

### 5.5.3 第三步：下界的关键性质与信任域的含义

该下界具有两个至关重要的性质：

**性质 1（紧性）**：当 $`\pi' = \pi`$ 时，$`\eta(\pi) = \mathbb{E}_{s \sim d^\pi}[\sum_a \pi(a|s) A^\pi(s,a)] = \mathbb{E}_{s \sim d^\pi}[0] = 0`$（因为 $`\mathbb{E}_{a \sim \pi}[A^\pi(s,a)] = 0`$，见第 2 章），故 $`L_\pi(\pi) = J(\pi)`$，且 $`\max_s D_\text{KL}(\pi\|\pi) = 0`$。因此 $`\underline{J}_\pi(\pi) = J(\pi)`$，下界在 $`\pi'=\pi`$ 处**取等号**。

**性质 2（梯度一致性）**：在 $`\pi' = \pi`$ 处，$`\nabla_{\pi'} L_\pi(\pi')|_{\pi'=\pi} = \nabla_{\pi'} \eta(\pi')|_{\pi'=\pi} = \nabla J(\pi)`$（因为 $`\nabla J(\pi)`$ 的策略梯度形式恰好等于 $`\nabla \eta(\pi')|_{\pi'=\pi}`$，见第 3 章策略梯度定理）。因此下界函数与真实目标在当前策略处不仅值相同，梯度也相同。

这两个性质的推论是：**只要我们能保证 $`\max_s D_\text{KL}(\pi'\|\pi)`$ 足够小，使得 $`\underline{J}_\pi(\pi') > J(\pi)`$，就必然有 $`J(\pi') > J(\pi)`$**。换言之，在 KL 球约束内优化替代目标 $`L_\pi`$ 就能保证真实性能单调不降。这个 KL 球就是所谓的**信任域（trust region）**——在该区域内，我们对替代目标的优化是「可信的」。

<a id="s5-6"></a>

## 5.6 TRPO 的实际求解形式

<a id="s5-6-1"></a>

### 5.6.1 约束优化问题

将上述理论转化为实际算法，TRPO 求解如下约束优化：

```math
\max_\theta \quad \mathbb{E}_{s \sim d^{\pi_{\theta_\text{old}}}, a \sim \pi_{\theta_\text{old}}}\left[r_t(\theta) \cdot \hat{A}_t\right] \quad \text{s.t.} \quad \mathbb{E}_{s \sim d^{\pi_{\theta_\text{old}}}}\left[D_\text{KL}\left(\pi_{\theta_\text{old}}(\cdot|s) \| \pi_\theta(\cdot|s)\right)\right] \leq \delta,
```

其中 $`r_t(\theta) = \pi_\theta(a_t|s_t)/\pi_{\theta_\text{old}}(a_t|s_t)`$ 为单步重要性比，$`\hat{A}_t`$ 为 GAE 优势估计（见第 4 章），$`\delta`$ 为预设的 KL 阈值（通常取 0.01）。

<a id="s5-6-2"></a>

### 5.6.2 从 max KL 到平均 KL 的松弛

理论下界使用的是 $`\max_s D_\text{KL}`$，而实践中使用的是关于 $`d^{\pi_{\theta_\text{old}}}`$ 的**平均 KL**。这是一个重要的松弛：平均 KL 约束不再保证 $`\max_s`$ KL 有界，因此**单调改进的下界在松弛后不再严格成立**。这是 TRPO 理论与实践之间的一个已知缺口。实践中，平均 KL 约束仍然有效，因为极端大的逐状态 KL 通常也会推高平均 KL；但在 pathological 情况下，理论上不能排除违反单调性的可能。

<a id="s5-7"></a>

## 5.7 自然梯度与 Fisher 信息矩阵

<a id="s5-7-1"></a>

### 5.7.1 参数空间 vs. 分布空间

标准梯度下降在参数空间中以欧氏度量 $`\|\Delta\theta\|^2`$ 衡量步长。但同一个 $`\Delta\theta`$ 在不同参数化下引起的分布变化完全不同。例如，对于 Bernoulli 分布 $`\text{Ber}(p)`$，参数 $`p=0.5`$ 附近 $`\Delta p = 0.01`$ 引起的分布变化很小，但在 $`p=0.01`$ 附近同样的 $`\Delta p = 0.01`$ 意味着分布的剧烈改变（从几乎确定输出 0 变为有一定概率输出 1）。我们需要一种**参数化无关**的距离度量，而 KL 散度恰好提供了这样的度量。

<a id="s5-7-2"></a>

### 5.7.2 KL 散度的二阶展开与 Fisher 矩阵

考虑 $`D_\text{KL}(\pi_\theta \| \pi_{\theta+\Delta\theta})`$ 关于 $`\Delta\theta`$ 的泰勒展开。由第 1 章结论：

- **零阶项**：$`D_\text{KL}(\pi_\theta \| \pi_\theta) = 0`$。
- **一阶项**：$`\nabla_{\Delta\theta} D_\text{KL}(\pi_\theta \| \pi_{\theta+\Delta\theta})|_{\Delta\theta=0} = -\mathbb{E}_{a \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s)] = 0`$，因为 score function 的期望为零（见第 1 章似然比技巧部分）。
- **二阶项**：Hessian 为 $`\nabla^2_{\Delta\theta} D_\text{KL}(\pi_\theta \| \pi_{\theta+\Delta\theta})|_{\Delta\theta=0} = \mathbb{E}_{a \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \nabla_\theta \log \pi_\theta(a|s)^\top\right] = F(\theta)`$，

其中 $`F(\theta)`$ 即为 **Fisher 信息矩阵**。因此：

```math
D_\text{KL}(\pi_\theta \| \pi_{\theta+\Delta\theta}) \approx \frac{1}{2} \Delta\theta^\top F(\theta) \Delta\theta.
```

这说明 Fisher 矩阵定义了参数空间中 KL 散度的**局部二次近似**，即以 Fisher 度量的二次型来衡量分布间的距离。

<a id="s5-7-3"></a>

### 5.7.3 自然梯度的变分推导

现在我们求解如下约束优化：在 KL 球内找到使目标增长最快的方向：

```math
\max_{\Delta\theta} \quad g^\top \Delta\theta \quad \text{s.t.} \quad \frac{1}{2} \Delta\theta^\top F \Delta\theta \leq \delta,
```

其中 $`g = \nabla_\theta J(\theta)`$ 为标准策略梯度。构造拉格朗日函数：

```math
\mathcal{L}(\Delta\theta, \lambda) = g^\top \Delta\theta - \lambda \left(\frac{1}{2} \Delta\theta^\top F \Delta\theta - \delta\right).
```

对 $`\Delta\theta`$ 求导令其为零：$`g - \lambda F \Delta\theta = 0`$，解得 $`\Delta\theta = \frac{1}{\lambda} F^{-1} g`$。代入约束的等式边界 $`\frac{1}{2}\Delta\theta^\top F \Delta\theta = \delta`$ 可确定 $`\lambda`$，但方向已经确定：

```math
\Delta\theta \propto F^{-1} g.
```

这就是**自然梯度（natural gradient）**：$`\tilde{\nabla} J(\theta) = F(\theta)^{-1} \nabla J(\theta)`$。它是在 KL 约束下使目标增长最快的方向，且与参数化选择无关（Fisher 矩阵在重参数化下协变变换，使得 $`F^{-1}g`$ 的方向不变）。

<a id="s5-7-4"></a>

### 5.7.4 常用策略的解析 KL 公式

以下两个公式在第 6、7 章的实现中直接使用。

**对角 Gaussian 策略**：设 $`\pi = \mathcal{N}(\mu, \text{diag}(\sigma^2))`$，$`\pi' = \mathcal{N}(\mu', \text{diag}(\sigma'^2))`$，维度为 $`d`$，则：

```math
D_\text{KL}(\pi \| \pi') = \sum_{i=1}^{d} \left[\log\frac{\sigma'_i}{\sigma_i} + \frac{\sigma_i^2 + (\mu_i - \mu'_i)^2}{2\sigma'^2_i} - \frac{1}{2}\right].
```

**Categorical 策略**：设 $`\pi(a) = p_a`$，$`\pi'(a) = q_a`$，则：

```math
D_\text{KL}(\pi \| \pi') = \sum_a p_a \log \frac{p_a}{q_a}.
```

这两个公式均可直接微分以计算 KL 约束的梯度与 Hessian-vector product。

<a id="s5-8"></a>

## 5.8 共轭梯度法求解自然梯度

<a id="s5-8-1"></a>

### 5.8.1 为什么不能直接求逆

Fisher 矩阵 $`F`$ 的大小为 $`n \times n`$，其中 $`n`$ 是策略网络的参数量（通常 $`10^5 \sim 10^7`$）。显式构造和存储 $`F`$ 需要 $`O(n^2)`$ 内存，求逆需要 $`O(n^3)`$ 计算，完全不可行。

<a id="s5-8-2"></a>

### 5.8.2 共轭梯度法的适用性

共轭梯度法（Conjugate Gradient, CG）是求解线性系统 $`Fx = g`$ 的迭代方法，其关键优势在于**只需要矩阵-向量乘积 $`Fv`$**，而不需要显式构造 $`F`$。对于 $`n`$ 维问题，CG 最多迭代 $`n`$ 次即可得到精确解；实践中由于 $`F`$ 的有效秩远小于 $`n`$，通常 $`10 \sim 20`$ 次迭代即可获得足够精度的近似解。

<a id="s5-8-3"></a>

### 5.8.3 Fisher-向量乘积的高效计算

$`Fv`$ 的计算利用以下恒等式。记 $`g(\theta) = \nabla_\theta J(\theta)`$，则：

```math
F(\theta) v = \nabla_\theta \left(g(\theta)^\top v\right),
```

这是因为 $`F = \mathbb{E}[\nabla \log \pi \nabla \log \pi^\top]`$ 且 $`\nabla(g^\top v) = \nabla(\mathbb{E}[\cdots]^\top v) = \mathbb{E}[\nabla(\nabla \log \pi^\top v)] = Fv`$（在 $`v`$ 与 $`\theta`$ 无关时）。实际操作中：先计算标量 $`g^\top v`$（一次前向+反向传播），再对该标量关于 $`\theta`$ 求梯度（又一次反向传播），总代价约为两次梯度计算，远低于构造 $`F`$。

另一种方法是 Hutchinson 估计：$`Fv = \mathbb{E}_z[(z^\top F v) z]`$ 其中 $`z`$ 为 Rademacher 随机向量，但实践中上述确定性方法更常用。

<a id="s5-8-4"></a>

### 5.8.4 线搜索保证 KL 约束

CG 给出自然梯度方向 $`\Delta\theta = F^{-1}g`$ 后，还需确定步长 $`\alpha`$。TRPO 使用回溯线搜索：从 $`\alpha = 1`$ 开始，逐步减半直到满足 KL 约束 $`\bar{D}_\text{KL}(\pi_{\theta+\alpha\Delta\theta} \| \pi_\theta) \leq \delta`$ 且目标函数有改善。这保证了每次更新都在信任域内。

<a id="s5-9"></a>

## 5.9 TRPO 的实践代价与 PPO 的动机

TRPO 虽然在理论上优美，但实践中存在显著代价：

1. **实现复杂**：需要手动实现 CG 求解器、线搜索、Fisher-向量乘积，代码量大且容易出错。
2. **计算开销**：每步更新需要多次梯度计算（CG 迭代 × 2 + 线搜索评估），吞吐量低于纯一阶方法。
3. **架构兼容性差**：CG 要求目标函数对参数二阶可微且 Fisher 矩阵正定，这与 dropout、batch normalization、参数共享、辅助损失等现代深度学习技术难以兼容。

PPO 的设计目标正是用**一阶优化方法**（SGD/Adam）达到「近似信任域」的效果，同时避免 TRPO 的上述缺点。PPO 提出了两条路线：

- **PPO-Penalty**：将 KL 约束转化为自适应惩罚项加入目标函数，通过调节惩罚系数间接控制 KL。
- **PPO-Clip**：直接截断重要性比 $`r_t(\theta)`$，使其落在 $`[1-\epsilon, 1+\epsilon]`$ 区间内，从根本上阻止过大的策略更新。

这两条路线的具体公式、损失函数设计与超参数机理将在第 6 章详细展开。本章建立的数学框架——重要性采样、性能差异引理、单调改进界、自然梯度的几何意义——构成了理解 PPO 设计选择的理论基础。

<a id="s5-10"></a>

## 5.10 本章小结与自测

<a id="h184"></a>

### 要点回顾

1. **重要性采样**提供了用旧分布样本估计新分布期望的无偏方法，但方差可能爆炸；有效样本量 ESS 是关键诊断指标。
2. **策略层面只用单步比率**而非轨迹比率，因为策略梯度定理已对状态分布取期望；代价是引入了状态分布失配误差。
3. **性能差异引理**精确表达了 $`J(\pi') - J(\pi)`$ 为新策略状态下优势函数的期望，揭示了 off-policy 评估的核心困难在于 $`d^{\pi'}`$ 未知。
4. **TRPO 单调改进界**通过将 $`d^{\pi'}`$ 替换为 $`d^\pi`$ 并用 KL 散度控制误差，建立了可计算的替代目标与真实性能之间的下界关系；在 $`\pi'=\pi`$ 处下界紧且梯度一致，保证了信任域内优化的可靠性。
5. **自然梯度** $`F^{-1}\nabla J`$ 是 KL 约束下的最优更新方向，通过共轭梯度法仅需 Fisher-向量乘积即可高效求解；但 TRPO 的实现复杂性催生了 PPO。

<a id="h185"></a>

### 自测题

**题目 1**：为什么 IS 估计量要求行为分布 $`p`$ 的支撑覆盖目标分布 $`q`$ 的支撑？若不满足，估计量是有偏还是无偏？

参考答案：若存在 $`q(x)>0`$ 但 $`p(x)=0`$ 的区域，该区域对 $`\mathbb{E}_q[f]`$ 的贡献永远无法被采样到，IS 估计量系统性低估真实期望，产生偏差。

**题目 2**：在策略梯度中使用单步重要性比而非轨迹比的根本原因是什么？单步比率忽略了什么误差来源？

参考答案：轨迹比的方差随 horizon 指数增长。单步比率之所以成立是因为策略梯度定理已对状态分布取期望；忽略的误差来源是新旧策略状态分布 $`d^{\pi_\theta}`$ 与 $`d^{\pi_{\theta_\text{old}}}`$ 的差异。

**题目 3**：性能差异引理 $`J(\pi')-J(\pi) = \mathbb{E}_{s\sim d^{\pi'}}[\sum_a \pi'(a|s)A^\pi(s,a)]`$ 中为什么出现的是 $`d^{\pi'}`$ 而非 $`d^\pi`$？这对算法设计有什么影响？

参考答案：推导中 telescoping 分解沿 $`\pi'`$ 轨迹进行，状态自然服从 $`d^{\pi'}`$。这使得精确计算性能差需要知道新策略的状态分布，导致无法直接用旧数据评估；TRPO 通过 KL 约束控制 $`d^{\pi'}`$ 与 $`d^\pi`$ 的差异来解决此问题。

**题目 4**：证明自然梯度方向 $`F^{-1}g`$ 是约束优化 $`\max_{\Delta\theta} g^\top\Delta\theta`$ s.t. $`\frac{1}{2}\Delta\theta^\top F\Delta\theta \leq \delta`$ 的解。

参考答案：拉格朗日函数 $`\mathcal{L}=g^\top\Delta\theta-\lambda(\frac{1}{2}\Delta\theta^\top F\Delta\theta-\delta)`$，对 $`\Delta\theta`$ 求导得 $`g-\lambda F\Delta\theta=0`$，故 $`\Delta\theta=\lambda^{-1}F^{-1}g`$，方向为 $`F^{-1}g`$。

**题目 5**：TRPO 实践中将 $`\max_s D_\text{KL}`$ 松弛为平均 KL 后，单调改进保证是否仍然严格成立？为什么？

参考答案：不再严格成立。平均 KL 有界不能推出每个状态的 KL 都有界，可能存在某些状态的 KL 远超平均值，使理论下界失效。实践中该松弛通常有效，但理论上存在缺口。

---

<a id="chapter-6"></a>

# 第 6 章 PPO 算法本体

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

其中 $`\text{clip}(x, l, u) = \max(l, \min(x, u))`$ 将 $`x`$ 截断到区间 $`[l, u]`$ 内，$`\varepsilon > 0`$ 是裁剪半径（典型取值 0.1～0.3，常用 0.2）。$`\mathbb{E}_t`$ 表示对时间步 $`t`$ 在 rollout 数据上的经验平均。

这个目标函数包含三个关键组件：
- **未裁剪项** $`r_t(\theta) \hat{A}_t`$：即第 5 章中的标准替代目标，是真实性能差异 $`J(\pi_\theta) - J(\pi_{\theta_{\text{old}}})`$ 的一阶近似。
- **裁剪项** $`\text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_t`$：将比率限制在 $`[1-\varepsilon, 1+\varepsilon]`$ 内后再乘以优势。
- **$`\min`$ 操作**：取两者中较小的一个，构造出一个悲观下界（pessimistic lower bound）。

我们的目标是**最大化** $`L^{\text{CLIP}}(\theta)`$。在实际实现中，通常对 $`-L^{\text{CLIP}}(\theta)`$ 做梯度下降最小化。

<a id="s6-2-2"></a>

### 6.2.2 六种情形的完整分析

为了理解 clip 与 min 的联合作用，我们需要分 $`\hat{A}_t > 0`$ 和 $`\hat{A}_t < 0`$ 两大情形，每种情形下再按 $`r_t(\theta)`$ 落在三个区间讨论。记 $`r = r_t(\theta)`$，$`A = \hat{A}_t`$，$`c = \text{clip}(r, 1-\varepsilon, 1+\varepsilon)`$。

**情形一：$`A > 0`$（该动作比平均水平好，我们希望增大其概率）**

| $`r`$ 所在区间 | $`c`$ 的值 | $`rA`$ vs $`cA`$ | $`\min`$ 取值 | 梯度 | 行为解释 |
|:---:|:---:|:---:|:---:|:---:|:---|
| $`r < 1-\varepsilon`$ | $`1-\varepsilon`$ | $`rA < cA`$（因 $`r < 1-\varepsilon`$） | $`rA`$（未裁剪项） | $`\nabla_\theta r \cdot A \neq 0`$ | 当前概率远低于旧策略，未裁剪项更小被选中，保留完整梯度鼓励增大概率 |
| $`1-\varepsilon \leq r \leq 1+\varepsilon`$ | $`r`$ | $`rA = cA`$ | $`rA = cA`$ | $`\nabla_\theta r \cdot A \neq 0`$ | 比率在安全区间内，两项相等，等价于未裁剪的策略梯度 |
| $`r > 1+\varepsilon`$ | $`1+\varepsilon`$ | $`rA > cA`$（因 $`r > 1+\varepsilon`$） | $`cA`$（裁剪项） | $`0`$（$`c`$ 对 $`\theta`$ 为常数） | 概率已大幅增加超出安全范围，裁剪项更小被选中，梯度为零，阻止继续增大 |

**情形二：$`A < 0`$（该动作比平均水平差，我们希望减小其概率）**

| $`r`$ 所在区间 | $`c`$ 的值 | $`rA`$ vs $`cA`$ | $`\min`$ 取值 | 梯度 | 行为解释 |
|:---:|:---:|:---:|:---:|:---:|:---|
| $`r < 1-\varepsilon`$ | $`1-\varepsilon`$ | $`rA > cA`$（因 $`A<0`$，$`r<1-\varepsilon`$ 使 $`rA`$ 负得少） | $`cA`$（裁剪项） | $`0`$ | 概率已大幅减小超出安全范围，裁剪项更小被选中，梯度为零，阻止继续减小 |
| $`1-\varepsilon \leq r \leq 1+\varepsilon`$ | $`r`$ | $`rA = cA`$ | $`rA = cA`$ | $`\nabla_\theta r \cdot A \neq 0`$ | 比率在安全区间内，等价于未裁剪的策略梯度 |
| $`r > 1+\varepsilon`$ | $`1+\varepsilon`$ | $`rA < cA`$（因 $`A<0`$，$`r>1+\varepsilon`$ 使 $`rA`$ 负得多） | $`rA`$（未裁剪项） | $`\nabla_\theta r \cdot A \neq 0`$ | 当前概率远高于旧策略但动作实际很差，未裁剪项更小被选中，保留完整梯度鼓励减小概率 |

以上六行可以浓缩为一条核心规则：**$`\min`$ 操作总是选择对当前参数更新"更保守"的那一项**。当策略更新方向与优势信号一致（$`A>0`$ 时增大 $`r`$，$`A<0`$ 时减小 $`r`$）且幅度超过 $`\varepsilon`$ 时，裁剪项生效、梯度归零，阻止过度更新；当策略更新方向与优势信号相反（例如 $`A>0`$ 但 $`r`$ 反而变小了），说明当前策略在该状态-动作对上犯了错，此时未裁剪项更小被选中，梯度不受阻碍，允许纠错。

<a id="s6-2-3"></a>

### 6.2.3 悲观下界的设计哲学与不对称性

PPO-Clip 最容易被误解的点在于其**不对称性**：它只阻止朝有利方向的过度更新，不阻止朝不利方向的回退。这不是 bug，而是刻意的设计。

从优化角度看，$`L^{\text{CLIP}}(\theta)`$ 是真实替代目标 $`L^{\text{SURR}}(\theta) = \mathbb{E}_t[r_t(\theta)\hat{A}_t]`$ 的一个**逐点下界**：对于每个样本 $`t`$，都有

```math
\min\big(r_t(\theta)\hat{A}_t, \; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}_t\big) \leq r_t(\theta)\hat{A}_t
```

因此 $`L^{\text{CLIP}}(\theta) \leq L^{\text{SURR}}(\theta)`$。最大化一个下界是安全的：如果下界提升了，真实目标至少不会更差（在局部近似成立的范围内）。这就是"悲观"的含义——我们宁可低估收益，也不愿高估后过度乐观地迈出太大一步。

对比 TRPO 的理论保障：TRPO 通过约束 $`\max_s D_{\text{KL}}`$ 得到了 $`J(\pi') \geq L_\pi(\pi') - C \cdot \max_s D_{\text{KL}}`$ 这一全局下界，其中 $`C`$ 依赖于优势函数的最大绝对值和折扣因子。PPO-Clip 没有这样的全局保证，因为 clip 是对单个样本比率的逐点操作，不等价于对状态级 KL 散度的约束。但在实践中，这种启发式的悲观下界足够有效，且计算成本远低于 TRPO 的二阶优化。

关于不对称性的直觉：当 $`A > 0`$ 且 $`r > 1+\varepsilon`$ 时，策略已经大幅增加了该好动作的概率，继续增加的边际收益递减而风险递增（可能导致其他好动作的概率被挤压），所以截断是合理的。但当 $`A > 0`$ 且 $`r < 1-\varepsilon`$ 时，策略反而降低了好动作的概率，这是一个明确的错误信号，不应该被截断——让梯度自由流动以尽快纠正。同理适用于 $`A < 0`$ 的对称情形。

<a id="s6-2-4"></a>

### 6.2.4 梯度一致性验证

一个自然的疑问是：在 $`\theta = \theta_{\text{old}}`$ 处，$`L^{\text{CLIP}}`$ 的梯度是否等于标准策略梯度？答案是肯定的。

当 $`\theta = \theta_{\text{old}}`$ 时，$`r_t(\theta) = 1`$，对所有 $`t`$ 成立。由于 $`1 \in [1-\varepsilon, 1+\varepsilon]`$（只要 $`\varepsilon > 0`$），clip 不生效，即 $`\text{clip}(1, 1-\varepsilon, 1+\varepsilon) = 1`$。此时两项相等：

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

其中 $`\beta > 0`$ 是 KL 惩罚系数，$`D_{\text{KL}}`$ 的定义见第 1 章。与 TRPO 的区别在于：TRPO 将 KL 作为硬约束（$`\leq \delta`$），而 PPO-Penalty 将其作为软惩罚项，并通过自适应机制调节 $`\beta`$。

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

其中 $`\phi`$ 是价值网络（critic）的参数，$`c_1, c_2 > 0`$ 是权重系数。下面逐项解释。

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

当 $`r_t(\theta)`$ 被 clip 时（即 $`r_t < 1-\varepsilon`$ 或 $`r_t > 1+\varepsilon`$ 且对应情形下裁剪项被 $`\min`$ 选中），该项对 $`\theta`$ 的梯度为零。这意味着这部分样本的信息被完全丢弃了。从统计估计的角度看，这等价于对梯度估计施加了一个与数据相关的截断，引入了系统性偏差。在理论上，这个偏差的大小与 $`\varepsilon`$ 和策略更新步数 $`K`$ 有关：$`\varepsilon`$ 越小或 $`K`$ 越大，被截断的样本比例越高，偏差越大。但在实践中，适度的偏差换取了方差的显著降低，总体效果是正面的。

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

其中 $`\pi_\theta(y_t | x, y_{<t})`$ 是语言模型在给定前缀条件下生成 token $`y_t`$ 的条件概率。优势 $`\hat{A}_t`$ 同样逐 token 计算，通常使用 GAE 或直接使用奖励模型的输出作为终端奖励进行反向传播。

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

**题目 1**：当 $`\hat{A}_t > 0`$ 且 $`r_t(\theta) > 1+\varepsilon`$ 时，$`L^{\text{CLIP}}`$ 中哪一项被 $`\min`$ 选中？该项对 $`\theta`$ 的梯度是多少？物理含义是什么？

参考答案：裁剪项 $`\text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\hat{A}_t = (1+\varepsilon)\hat{A}_t`$ 被选中（因为它小于 $`r_t\hat{A}_t`$）。梯度为零，因为 $`1+\varepsilon`$ 是常数。物理含义是策略已经大幅增加了好动作的概率，clip 阻止继续增加以防止过度更新。

**题目 2**：当 $`\hat{A}_t < 0`$ 且 $`r_t(\theta) < 1-\varepsilon`$ 时，$`\min`$ 选中的是哪一项？为什么这种情况下不阻止更新？

参考答案：裁剪项 $`(1-\varepsilon)\hat{A}_t`$ 被选中（因为 $`A<0`$ 时 $`(1-\varepsilon)A < rA`$）。梯度为零。等等——重新审视：当 $`A<0`$ 且 $`r<1-\varepsilon`$ 时，$`rA > (1-\varepsilon)A`$（两个负数，$`r`$ 更小使 $`rA`$ 负得少即更大），所以 $`\min`$ 取 $`(1-\varepsilon)A`$，梯度确实为零。这实际上是阻止了"概率已经降得太低还要继续降"的过度更新。题目问"为什么不阻止"的前提有误——PPO 在这种情况下确实阻止了更新。正确理解是：PPO 阻止的是朝任一方向的过度更新，而非仅阻止有利方向。

**题目 3**：证明在 $`\theta = \theta_{\text{old}}`$ 处，$`L^{\text{CLIP}}`$ 的梯度等于标准策略梯度 $`\mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot \hat{A}]`$。

参考答案：$`\theta = \theta_{\text{old}}`$ 时 $`r_t = 1 \in [1-\varepsilon, 1+\varepsilon]`$，clip 不生效，两项相等均为 $`\hat{A}_t`$。$`\min`$ 取任意一项均可，$`L^{\text{CLIP}} = \mathbb{E}[\hat{A}_t]`$。对其求梯度得 $`\mathbb{E}[\nabla_\theta r_t \cdot \hat{A}_t] = \mathbb{E}[\nabla_\theta \log \pi_\theta \cdot \hat{A}_t]`$，即标准策略梯度。

**题目 4**：RLHF 中奖励塑形的 KL 惩罚与 PPO-Penalty 中的 KL 项有何本质区别？如果只用 PPO-Clip 做 RLHF，是否可以省略所有 KL 相关项？

参考答案：奖励塑形的 KL 是 $`\pi_\theta`$ 与固定参考模型 $`\pi_{\text{ref}}`$ 的距离，进入奖励计算防止 reward hacking；PPO-Penalty 的 KL 是 $`\pi_\theta`$ 与 $`\pi_{\theta_{\text{old}}}`$ 的距离，进入损失函数约束更新步长。即使用 PPO-Clip，奖励塑形的 KL 惩罚也不可省略，否则策略会钻奖励模型漏洞。PPO-Penalty 的 KL 项可以用 clip 替代。

**题目 5**：为什么 PPO 内层可以做多个 epoch 的数据复用，而普通策略梯度不行？clip 在其中扮演什么角色？

参考答案：普通策略梯度是 on-policy 的，数据必须来自当前策略，复用旧数据会导致重要性采样比率 $`r_t`$ 偏离 1，梯度估计方差爆炸。PPO 通过 clip 将 $`r_t`$ 的有效影响限制在 $`[1-\varepsilon, 1+\varepsilon]`$ 内，即使 $`r_t`$ 实际值远离 1，目标函数的行为和梯度都被控制在合理范围内，使得多 epoch 复用数据成为可能。

**题目 6**：Andrychowicz et al. (2021) 的发现对"PPO 之所以有效是因为 clip 机制"这一观点构成了怎样的挑战？我们应该如何重新理解 PPO 的成功？

参考答案：该研究发现工程细节（网络结构、初始化、归一化、学习率调度）对性能的影响与 clip 同等重要，且在某些设置下去掉 clip 也能工作。这说明 PPO 的成功是 clip 机制与一系列工程实践共同作用的结果，不应过度归因于单一组件。clip 提供了必要的稳定化，但良好的工程实践同样是不可或缺的。理解 PPO 应将其视为一个系统性的方法论，而非仅仅是一个目标函数。

---

<a id="chapter-7"></a>

# 第 7 章 工程实现与实践

前六章完成了 PPO 从数学基础到算法本体的全部理论推导。本章的目标是将这些公式逐行落地为可运行的 PyTorch 代码，并明确指出每一处实现与理论之间的差异、近似和数值稳定性处理。读者在阅读本章时，应随时对照前文公式，理解"代码为什么这样写"而非仅仅记住模板。

<a id="s7-1"></a>

## 7.1 环境接口与 Gymnasium

<a id="s7-1-1"></a>

### 7.1.1 Gymnasium API 概述

Gymnasium 是 OpenAI Gym 的官方继任者，提供了标准化的强化学习环境接口。核心 API 有两个：

- `env.reset(seed=None)` → `(obs, info)`：重置环境并返回初始观测与辅助信息字典。
- `env.step(action)` → `(obs, reward, terminated, truncated, info)`：执行动作后返回五元组。

其中 `terminated` 与 `truncated` 的语义区分是 PPO 正确实现的关键，也是旧版 Gym（仅返回单一 `done`）最常见的 bug 来源。

<a id="s7-1-2"></a>

### 7.1.2 terminated 与 truncated 的语义区别

**`terminated=True`** 表示环境因内在逻辑真正结束了一个 episode（例如 CartPole 中杆子倒下、LunarLander 中着陆器坠毁）。此时状态 $`s_{t+1}`$ 是吸收态（absorbing state），其价值 $`V(s_{t+1}) = 0`$，在 GAE 计算中**不应 bootstrap**。

**`truncated=True`** 表示环境仅因外部限制（通常是达到最大步数上限）而中断，episode 在 MDP 意义上并未结束。此时 $`s_{t+1}`$ 仍然是一个有效的非终止状态，**必须用 $`V(s_{t+1})`$ 进行 bootstrap**，否则会将"时间到"错误地当作"失败"，导致价值函数系统性低估。

对应第 4 章的 TD 误差 $`\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)`$，正确的写法应为：

```math
\delta_t = r_t + \gamma \cdot (1 - \text{terminated}_t) \cdot V(s_{t+1}) - V(s_t)
```

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

<a id="s7-2"></a>

## 7.2 向量环境（Vectorized Environments）

<a id="s7-2-1"></a>

### 7.2.1 为什么 PPO 必须用向量环境

第 6 章指出 PPO 每次更新需要 $`N \times T`$ 个样本（$`N`$ 个并行环境各采集 $`T`$ 步）。单环境采集效率极低，且批量过小会导致：(1) 优势估计方差过大；(2) mini-batch 更新不稳定。向量环境通过并行化解决这两个问题。

<a id="s7-2-2"></a>

### 7.2.2 SyncVectorEnv 与 AsyncVectorEnv

Gymnasium 提供两种向量环境封装：

- **`SyncVectorEnv`**：在同一进程中顺序执行 $`N`$ 个环境。优点是共享内存、无序列化开销；缺点是总耗时为 $`N`$ 倍单环境耗时。适合轻量级环境（如 CartPole）。
- **`AsyncVectorEnv`**：每个环境运行在独立子进程中，通过管道通信。真正的并行加速，适合计算密集型环境。代价是进程间数据序列化开销和更复杂的调试体验。

<a id="s7-2-3"></a>

### 7.2.3 Rollout Buffer 的形状约定

推荐约定：buffer 形状为 $`(T, N, \ldots)`$，即第一维是时间步、第二维是环境索引。填充完毕后 flatten 前两维得到 $`(T \times N, \ldots)`$ 用于 mini-batch 采样：

```python
# 采集阶段：shape = (num_steps, num_envs, ...)
obs_buffer = np.zeros((num_steps, num_envs) + obs_shape, dtype=np.float32)
# ... 填充循环 ...

# 更新阶段：flatten 为 (num_steps * num_envs, ...)
obs_flat = obs_buffer.reshape(-1, *obs_shape)
actions_flat = actions_buffer.reshape(-1, *action_shape)
# 后续按 mini_batch_size 随机切分
```

<a id="s7-2-4"></a>

### 7.2.4 异步环境的自动 reset 语义

向量环境中，当某个子环境 terminated 或 truncated 时，会**自动调用 `reset()` 并将新 episode 的首个观测写入返回值**。这意味着 rollout buffer 中相邻时间步可能属于不同 episode。GAE 计算时必须依靠 `terminated` 标志来正确断开 episode 边界，而不能假设连续的时间步属于同一 episode。这也是为什么 7.1.2 节中 `nextnonterminal` 的处理如此重要。

<a id="s7-3"></a>

## 7.3 网络结构设计的数学考量

<a id="s7-3-1"></a>

### 7.3.1 MLP 主干配置

对于 CartPole 等低维任务，两层 $`64 \times 64`$ + `tanh` 激活即可；对于 LunarLander 及以上复杂度的任务，推荐 $`256 \times 256`$ + `ReLU`。激活函数的选择影响梯度传播：`tanh` 输出有界 $`[-1,1]`$，适合小网络；`ReLU` 避免饱和但可能产生死神经元，配合正交初始化可缓解。

<a id="s7-3-2"></a>

### 7.3.2 初始化对策略梯度的特殊影响

PPO 对网络初始化极其敏感。**最后一层输出必须用小权重初始化**，使初始策略接近均匀分布（离散）或窄高斯（连续）。数学解释如下：

- **离散动作空间**：初始 logits 接近 $`0`$ → softmax 输出接近均匀分布 → 熵 $`H[\pi]`$ 最大 → 探索最充分。若初始 logits 绝对值过大，策略过早偏向某些动作，熵崩塌，难以恢复。
- **连续动作空间**：初始均值接近 $`0`$、标准差适中（如 $`\sigma \approx 1`$）→ 避免一开始就过度自信地集中在某个动作附近。

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

<a id="s7-3-3"></a>

### 7.3.3 连续动作空间的对数标准差参数化

对于对角 Gaussian 策略（对应第 1 章和第 5 章），标准差 $`\sigma`$ 的参数化有两种选择：

1. **独立参数**：`log_std = nn.Parameter(torch.zeros(action_dim))`，与状态无关。简单、参数少，适合大多数连续控制任务。
2. **网络输出**：由主干网络的额外头输出，与状态相关。表达力更强但训练更不稳定。

无论哪种方式，**必须约束 `log_std` 的范围**，通常 clip 到 $`[-20, 2]`$（对应 $`\sigma \in [2 \times 10^{-9}, 7.39]`$）或更保守的 $`[-5, 2]`$。原因：

- `log_std` 过小 → $`\sigma \to 0`$ → 策略退化为确定性策略 → 熵崩塌 → 无法再探索，且 KL 散度计算可能出现数值异常。
- `log_std` 过大 → $`\sigma`$ 爆炸 → 动作采样范围不合理 → 奖励信号被噪声淹没。

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

<a id="s7-3-4"></a>

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

**分离主干**各自独立前向，互不干扰，但计算量翻倍。实践中两者效果差异不大，初学者建议先用分离主干以避免调参复杂度。若使用共享主干，建议将 $`c_1`$ 设为较小值（如 0.5），并在监控 `explained_variance` 确认 critic 收敛后再考虑增大。

<a id="s7-4"></a>

## 7.4 PPO 完整参考实现

以下是一份可直接运行于 CartPole-v1 的 PPO 实现，每段代码均标注对应的理论公式出处。

<a id="s7-4-1"></a>

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

<a id="s7-4-2"></a>

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

<a id="s7-4-3"></a>

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

<a id="s7-4-4"></a>

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

<a id="s7-4-5"></a>

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

<a id="s7-4-6"></a>

### 7.4.6 梯度裁剪澄清

`torch.nn.utils.clip_grad_norm_(params, max_grad_norm=0.5)` 裁剪的是所有参数梯度的**全局 L2 范数**：

```math
\|g\|_2 = \sqrt{\sum_i g_i^2}
```

若 $`\|g\|_2 > \text{max\_grad\_norm}`$，则将所有梯度分量等比例缩放至 $`\text{max\_grad\_norm}`$。这与 PPO-Clip 中 $`\text{clip}(r_t, 1-\varepsilon, 1+\varepsilon)`$ 的逐元素裁剪机制完全不同。前者是优化层面的稳定性手段（对应第 1 章），后者是算法层面的策略更新约束（对应第 6 章）。**这两个 "clip" 是 PPO 实现中最容易混淆的概念，务必区分。**

<a id="s7-5"></a>

## 7.5 数值稳定性与实现细节的数学解释

<a id="s7-5-1"></a>

### 7.5.1 对数域计算比率

代码中始终使用 `logratio = new_logprob - old_logprob` 然后 `ratio = logratio.exp()`，而非直接 `ratio = new_prob / old_prob`。原因：概率值可能极小（如 $`10^{-30}`$），直接相除会导致浮点下溢或上溢。对数域减法等价于原始域除法，且 `log_prob` 本身由分布对象精确计算，数值范围可控。仅在最终需要 `ratio` 时才做一次 `exp`，此时 `logratio` 通常在 $`[-10, 10]`$ 范围内，不会溢出。

<a id="s7-5-2"></a>

### 7.5.2 近似 KL 的选择

两种常见近似：

1. **一阶近似**：$`\hat{D}_{KL}^{(1)} = \mathbb{E}[\log \pi_{\text{old}} - \log \pi_{\text{new}}]`$。简单但可能为负（违反 KL 非负性），不适合做 early stopping 判据。
2. **Schulman 二阶近似**：$`\hat{D}_{KL}^{(2)} = \mathbb{E}[r - 1 - \log r]`$，其中 $`r = \pi_{\text{new}} / \pi_{\text{old}}`$。由不等式 $`x - 1 \geq \log x`$ 保证恒非负，且在 $`r \approx 1`$ 时二阶逼近真实 KL。

推荐使用第二种作为 early stopping 判据，因为它不会出现虚假的负值触发。

<a id="s7-5-3"></a>

### 7.5.3 优势归一化的位置

- **整个 rollout 归一化**：统计量基于全部 $`T \times N`$ 个样本，方差估计更准，但不同 mini-batch 的优势尺度相同，丧失了 batch 内的自适应调节能力。
- **Mini-batch 内归一化**：每个 mini-batch 独立标准化。实践中更常用，因为：(1) 减少了不同 mini-batch 之间优势尺度的差异对梯度的影响；(2) 等价于对每个 mini-batch 做了局部的基线校正，进一步降低方差。

<a id="s7-5-4"></a>

### 7.5.4 价值裁剪

价值裁剪将 $`V_\theta(s)`$ 限制在 $`[V_{\text{old}}(s) - \epsilon_v, V_{\text{old}}(s) + \epsilon_v]`$ 内，意图防止价值函数单次更新过大。但如第 6 章所述，其收益存在争议：价值函数的优化目标是 MSE，不像策略目标那样有单调改进保证，裁剪反而可能阻碍 critic 快速收敛。多数现代实现默认关闭价值裁剪，仅在发现 `value_loss` 剧烈震荡时才考虑启用。

<a id="s7-6"></a>

## 7.6 训练诊断指标与调参方法论

<a id="s7-6-1"></a>

### 7.6.1 必须监控的量及其健康范围

| 指标 | 数学含义 | 健康范围 | 异常排查方向 |
|------|---------|---------|-------------|
| Episode Return | $`\mathbb{E}[\sum \gamma^t r_t]`$，主性能指标 | 持续上升 | 若不升：检查奖励归一化、网络容量、lr |
| approx_kl | $`\mathbb{E}[r-1-\log r]`$，策略变化幅度 | < 0.015~0.02 | 过大：降 lr、减 K、增大 ε |
| clip_fraction | 被 clip 的样本占比 | 0.05~0.2 | ≈0：ε 太松或更新太保守；≈1：策略剧变 |
| Entropy | $`H[\pi]`$，探索程度 | 缓慢下降 | 突然崩塌：log_std 塌缩、lr 过高 |
| explained_variance | $`1 - \frac{\text{Var}(R-V)}{\text{Var}(R)}`$，critic 拟合质量 | > 0.5 | < 0：critic 比常数预测还差，检查网络/lr/归一化 |
| value_loss | $`\mathbb{E}[(V_\theta(s) - \hat{R})^2]`$ | 持续下降 | 不降或震荡：critic 容量不足、lr 不当 |
| policy_loss | $`-\mathbb{E}[\min(r\hat{A}, \text{clip}(r)\hat{A})]`$ | 无固定范围 | 突增：结合 approx_kl 判断 |
| grad_norm | $`\|\nabla L\|_2`$（裁剪前） | 平稳 | 突增：不稳定前兆，检查奖励尺度、观测归一化 |

<a id="s7-6-2"></a>

### 7.6.2 调参优先级

基于 Andrychowicz et al. (2021) 的大规模实证结论：

1. **先调学习率与网络结构**：这两个因素对性能影响最大。lr 过高导致 KL 爆炸，过低导致收敛极慢。
2. **再调 $`\varepsilon`$ 与 $`K`$**：$`\varepsilon`$ 控制单次更新的信任域大小，$`K`$ 控制数据复用程度。
3. **$`\gamma`$ 和 $`\lambda`$ 一般用默认值**（0.99 和 0.95），除非任务有特殊的时间尺度需求。

<a id="s7-6-3"></a>

### 7.6.3 训练崩溃排查清单

当训练出现异常时，按以下顺序排查：

1. **看 `explained_variance`**：若 < 0 或不升，critic 有问题 → 检查 critic 网络结构、lr、观测归一化。
2. **看 `entropy`**：若突然崩塌 → 检查 `log_std` 是否有 clamp、lr 是否过高。
3. **看 `approx_kl`**：若持续 > 0.02 → 降 lr、减少 K、增大 ε。
4. **检查观测/奖励归一化**：缺失归一化是 PPO 崩溃的最常见原因之一。

<a id="s7-6-4"></a>

### 7.6.4 观测归一化与奖励归一化

**观测归一化（Observation Normalization）**：使用 Welford 在线算法维护 running mean/std：

```math
\bar{x}_n = \bar{x}_{n-1} + \frac{x_n - \bar{x}_{n-1}}{n}, \quad M_n = M_{n-1} + (x_n - \bar{x}_{n-1})(x_n - \bar{x}_n)
```

其中 $`\sigma_n = \sqrt{M_n / n}`$。PPO 对观测归一化极其敏感，尤其是多环境并行时，统计量必须用**全体环境的数据**更新，而非单个环境。

**奖励缩放（Reward Scaling）**：将奖励除以 running std，使回报的尺度大致在 $`[-1, 1]`$ 范围内。**注意**：奖励缩放会改变 $`\gamma`$ 的有效含义——缩放后的折扣因子等效于 $`\gamma' = \gamma \cdot \sigma_{\text{old}} / \sigma_{\text{new}}`$，在 running std 尚未收敛的训练初期，这种效应尤为明显。因此建议在 running statistics 稳定后再评估最终性能。

<a id="s7-7"></a>

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

<a id="s7-8"></a>

## 7.8 推荐学习资源与进阶路径

<a id="s7-8-1"></a>

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

<a id="s7-8-2"></a>

### 7.8.2 代码库

| 代码库 | 定位 | 适用场景 |
|--------|------|---------|
| **CleanRL** | 单文件实现，代码与论文公式逐行对应 | **强烈推荐入门**，最适合逐行对照公式学习 |
| **Stable-Baselines3** | 工业级封装，API 简洁 | 快速跑实验、生产部署 |
| **OpenAI Spinning Up** | 配套教程质量最高 | 理论学习与代码实践并行 |
| **TRL** (Hugging Face) | RLHF/PPO 专用 | LLM 对齐、RLHF 研究 |

<a id="s7-8-3"></a>

### 7.8.3 务实的学习路径建议

1. **跑通基线**：用 CleanRL 的 `ppo.py` 在 CartPole-v1 上跑通，逐行对照第 6 章伪代码与本章实现，确认每个公式的代码位置。
2. **主动破坏实验**：故意去掉 truncated 处理、去掉优势归一化、将 `log_std` 的 clamp 移除、把学习率调高 10 倍……观察每种破坏对应的训练曲线变化。这是理解"为什么需要这个设计"最有效的方式。
3. **升级任务**：在 LunarLander-v3 上复现，体会网络容量、归一化、超参调整的必要性。
4. **阅读 RLHF 实现**：对照 TRL 的 PPO trainer，理解 token 级别的 PPO 与经典 PPO 的差异（KL 奖励塑形、reference model 等）。
5. **尝试改进**：在理解标准 PPO 后，尝试实现 PPO-Penalty、GRPO 等变体，加深对算法设计空间的理解。

<a id="s7-9"></a>

## 7.9 本章小结与下一步

**要点回顾：**

1. **terminated 与 truncated 的正确处理**是 PPO 实现中最关键的细节之一，直接影响 GAE 和价值函数的正确性。
2. **向量环境**是 PPO 的工程基础设施，决定了数据采集效率和 batch 质量。
3. **网络初始化**（特别是输出层的小权重正交初始化）对策略梯度的初始探索行为有决定性影响。
4. **数值稳定性**贯穿始终：对数域计算比率、`log_std` 约束、mini-batch 级优势归一化、梯度范数裁剪——每一处都有明确的数学动机。
5. **诊断指标**是调参的指南针：`explained_variance`、`approx_kl`、`entropy`、`clip_fraction` 四个量足以定位绝大多数训练问题。

**学完这份文档之后该做什么：**

不要停留在"读懂了"的阶段。打开 CleanRL 的 `ppo.py`，在一个你熟悉的环境上从头跑一遍，然后按照 7.8.3 节的路径主动破坏、修复、升级。当你能不看任何参考独立写出一个能稳定训练 LunarLander 的 PPO，并且能解释每一行代码背后的数学理由时，才算真正掌握了这份文档的内容。之后可以进入 RLHF 领域，将 PPO 应用于 LLM 对齐——那里的挑战不再是算法本身，而是如何在巨大的模型规模和有限的反馈信号下保持 PPO 的稳定性。
