# PPO — All you need to know

> 从概率论一步步推到 PPO：讲清公式的来由、近似与实现，拆开 GAE 与 clip 的细节，把 PyTorch 代码接到训练诊断与排坑。中文系统教程，含 31 道章末练习与参考答案。

中文学习笔记，以数学推导、算法理解与工程实践为主。需要微积分、线性代数、概率论与 Python 基础；从第 1 章开始顺序阅读。

**[开始阅读](docs/chapter-01.md)** · **[完整单文件版](PPO系统学习文档.md)** · **[阅读路线、依赖关系与符号表](docs/README.md)** · [配套 GRPO 文档](https://github.com/zwdx/grpo-all-you-need-to-know)

## 你会沿着怎样的路线学习

- **从基础推到算法**：概率论 → MDP → 策略梯度 → GAE → TRPO → PPO，每一章接着前一章的结论往下走。
- **把关键步骤展开**：对照 [GAE 的推导](docs/chapter-04.md#s4-5) 与递推实现，拆开 [clip 与 min 的六种情形](docs/chapter-06.md#s6-2)，理解公式如何影响梯度。
- **把理论接到代码与调试**：阅读 [PyTorch 参考实现](docs/chapter-07.md#s7-4)，再用 [训练诊断](docs/chapter-07.md#s7-6) 理解指标和排查问题。
- **31 道章末练习，附参考答案**：先独立思考或手推，再展开答案检查理解；分章版默认折叠答案。

## 如何使用

| 你的目标 | 阅读建议 |
|---|---|
| 系统学习 | 按下方章节顺序阅读，结合卷首的依赖关系与阅读路线 |
| 先看算法 | 从 [第 6 章](docs/chapter-06.md) 开始，遇到前置概念再回查 |
| 先看实现 | 从 [第 7 章](docs/chapter-07.md) 开始，对照公式理解实现细节 |
| 复习与查找 | 用完整文档的可点击目录、GitHub 的 Outline 或浏览器查找定位关键词 |
| 检查掌握程度 | 先独立完成章末自测，再展开分章版中的参考答案 |

## 章节导航

<!-- chapters:start -->
| 章节 | 阅读入口 |
|---|---|
| 第 1 章 数学基础 | [分章阅读](docs/chapter-01.md) · [全文位置](PPO系统学习文档.md#chapter-1) |
| 第 2 章 强化学习核心概念 | [分章阅读](docs/chapter-02.md) · [全文位置](PPO系统学习文档.md#chapter-2) |
| 第 3 章 策略梯度方法 | [分章阅读](docs/chapter-03.md) · [全文位置](PPO系统学习文档.md#chapter-3) |
| 第 4 章 优势估计与广义优势估计（GAE） | [分章阅读](docs/chapter-04.md) · [全文位置](PPO系统学习文档.md#chapter-4) |
| 第 5 章 信任域方法与重要性采样 | [分章阅读](docs/chapter-05.md) · [全文位置](PPO系统学习文档.md#chapter-5) |
| 第 6 章 PPO 算法本体 | [分章阅读](docs/chapter-06.md) · [全文位置](PPO系统学习文档.md#chapter-6) |
| 第 7 章 工程实现与实践 | [分章阅读](docs/chapter-07.md) · [全文位置](PPO系统学习文档.md#chapter-7) |
<!-- chapters:end -->

## 阅读体验

- 分章版提供本章目录、上一章、下一章与返回总目录；完整单文件版提供章与小节目录。
- 公式直接使用 GitHub 的 LaTeX 渲染，代码块保留语法高亮；学习时建议手推关键公式。
- 卷首保留原文的逻辑主线、章节依赖、阅读路线与通用符号约定。

## 勘误与贡献

发现问题时，请在 [Issues](https://github.com/zwdx/ppo-all-you-need-to-know/issues) 中写明章节编号、原文、建议修改与依据；也欢迎提交 Pull Request。

维护时修改根目录的完整文档，再执行以下命令更新分章版与目录。Python 3.10 及以上即可，无需安装额外依赖。

```bash
python scripts/build_docs.py
python scripts/build_docs.py --check
```

## 许可证

本仓库采用 [Creative Commons Attribution 4.0 International（CC BY 4.0）](LICENSE)。转载或改编时请注明文档名称、作者 zwdx 与原仓库链接，并说明修改。
