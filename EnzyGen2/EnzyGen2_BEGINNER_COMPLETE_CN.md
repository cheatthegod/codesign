# EnzyGen2 完全初学者指南（完整扩展版）

> **适合人群：** 对蛋白质设计、酶工程、等变图神经网络（EGNN）、蛋白质语言模型（PLM）接触不多，想把 EnzyGen2 的输入、输出、架构、训练、推理、源码主路径一次看透的读者。
>
> **阅读建议：** 不要跳读。这份指南的逻辑是"先建立直觉，再学原理，再看架构，再看 shape"。每一节都依赖上一节建立的概念。
>
> **读完之后，你应该能清楚回答以下问题：**
> 1. 蛋白质序列-结构协同设计是什么，为什么重要，为什么难
> 2. 酶与配体结合的生物学背景
> 3. ESM2 蛋白质语言模型在 EnzyGen2 中如何工作
> 4. 等变图神经网络（EGNN）如何处理三维坐标
> 5. NCBI 物种分类学嵌入为什么有用
> 6. 三阶段预训练（MLM → Motif → Full）为什么这样设计
> 7. 配体结合预测模块（SubstrateEGNN + 结合预测网络）如何工作
> 8. 前向传播的每一步操作了什么张量，shape 如何变化
> 9. 三种损失函数（序列恢复、结构预测、结合预测）的数学定义
> 10. 推理时的三种解码策略（greedy、top-k、top-p）

---

## 目录

### 第一部分：背景与直觉建立
- [1. 蛋白质设计：问题背景](#1-蛋白质设计问题背景)
- [2. 酶与配体结合：生化基础](#2-酶与配体结合生化基础)
- [3. 序列-结构协同设计：为什么比单独设计更好](#3-序列-结构协同设计为什么比单独设计更好)
- [4. EnzyGen2 解决什么任务](#4-enzygen2-解决什么任务)
- [5. 系统整体架构一览](#5-系统整体架构一览)

### 第二部分：ESM2 蛋白质语言模型（编码器）
- [6. 蛋白质语言模型的直觉](#6-蛋白质语言模型的直觉)
- [7. ESM2 的具体结构与参数](#7-esm2-的具体结构与参数)
- [8. Alphabet：氨基酸词表设计](#8-alphabet氨基酸词表设计)
- [9. Token Dropout 机制](#9-token-dropout-机制)
- [10. Rotary Position Embedding](#10-rotary-position-embedding)

### 第三部分：等变图神经网络（EGNN，解码器）
- [11. 为什么需要几何深度学习](#11-为什么需要几何深度学习)
- [12. E(n) 等变性：直觉解释](#12-en-等变性直觉解释)
- [13. EGNN 单层的三步操作：边模型、坐标模型、节点模型](#13-egnn-单层的三步操作边模型坐标模型节点模型)
- [14. K-近邻图构建](#14-k-近邻图构建)
- [15. 四种 EGNN 变体详解](#15-四种-egnn-变体详解)
- [16. SubstrateEGNN：配体原子处理](#16-substrateegnn配体原子处理)

### 第四部分：核心模型架构
- [17. GeometricProteinNCBIModel：基础模型](#17-geometricproteinncbimodel基础模型)
- [18. NCBI 物种分类学嵌入](#18-ncbi-物种分类学嵌入)
- [19. 编码器-解码器交错：每 11 层 Transformer 接一层 EGNN](#19-编码器-解码器交错每-11-层-transformer-接一层-egnn)
- [20. 被遮蔽残基的坐标初始化](#20-被遮蔽残基的坐标初始化)
- [21. GeometricProteinNCBISubstrateModel：配体结合扩展](#21-geometricproteinncbisubstratemodel配体结合扩展)
- [22. 完整前向传播追踪（含每步 shape）](#22-完整前向传播追踪含每步-shape)

### 第五部分：损失函数与训练目标
- [23. 序列恢复损失（Sequence Recovery Loss）](#23-序列恢复损失sequence-recovery-loss)
- [24. 结构预测损失（Structure Prediction Loss）](#24-结构预测损失structure-prediction-loss)
- [25. 配体结合预测损失（Binding Prediction Loss）](#25-配体结合预测损失binding-prediction-loss)
- [26. 总损失函数与权重平衡](#26-总损失函数与权重平衡)

### 第六部分：数据处理流水线
- [27. JSON 数据格式详解](#27-json-数据格式详解)
- [28. IndexedRawTextDataset：序列加载](#28-indexedrawtextdataset序列加载)
- [29. CoordinateDataset：坐标加载与中心化](#29-coordinatedataset坐标加载与中心化)
- [30. ProteinMotifDataset：遮蔽策略](#30-proteinmotifdataset遮蔽策略)
- [31. NCBITaxonomyDataset：物种分类](#31-ncbitaxonomydataset物种分类)
- [32. 配体数据集三件套：原子、坐标、结合标签](#32-配体数据集三件套原子坐标结合标签)
- [33. Batch Collation：从样本到批次](#33-batch-collation从样本到批次)

### 第七部分：三阶段训练流程
- [34. Stage 1：掩码语言模型预训练（MLM）](#34-stage-1掩码语言模型预训练mlm)
- [35. Stage 2：Motif 约束预训练](#35-stage-2motif-约束预训练)
- [36. Stage 3：完整训练（含配体结合）](#36-stage-3完整训练含配体结合)
- [37. 微调（Finetuning）特定酶家族](#37-微调finetuning特定酶家族)
- [38. 超参数对比速查表](#38-超参数对比速查表)

### 第八部分：推理与生成
- [39. 三种解码策略：Greedy、Top-K、Top-P](#39-三种解码策略greedytop-ktop-p)
- [40. 生成流水线完整追踪](#40-生成流水线完整追踪)
- [41. PDB 文件生成](#41-pdb-文件生成)
- [42. ESP 评估指标](#42-esp-评估指标)

### 第九部分：参考与速查
- [43. 关键张量形状速查表](#43-关键张量形状速查表)
- [44. 关键数字速查](#44-关键数字速查)
- [45. 常见误区（初学者必读）](#45-常见误区初学者必读)
- [46. 源码阅读顺序建议](#46-源码阅读顺序建议)
- [47. 总结：EnzyGen2 的价值与局限](#47-总结enzygen2-的价值与局限)

---

## 1. 蛋白质设计：问题背景

### 1.1 蛋白质是什么——最短回顾

蛋白质是由氨基酸串联而成的长链分子。自然界存在 20 种标准氨基酸，它们用单字母缩写表示：

```
A(丙氨酸)  G(甘氨酸)  V(缬氨酸)  L(亮氨酸)  I(异亮氨酸)
P(脯氨酸)  F(苯丙氨酸) W(色氨酸)  M(蛋氨酸)  S(丝氨酸)
T(苏氨酸)  C(半胱氨酸) Y(酪氨酸)  H(组氨酸)  D(天冬氨酸)
E(谷氨酸)  N(天冬酰胺) Q(谷氨酰胺) K(赖氨酸)  R(精氨酸)
```

一条蛋白质的**氨基酸序列**（一级结构）是 20 个字母的字符串，例如：

```
MKVAVLGAAGGIGQALALLLKSL...   ← 某酶蛋白，数百个氨基酸
```

这条链在细胞中自动折叠成特定的三维形状。**序列决定结构，结构决定功能。**

### 1.2 蛋白质设计的含义

**蛋白质设计**是指根据目标功能需求，**从头设计**或**改造**蛋白质的序列和结构：

| 设计层面 | 具体问题 |
|---------|----------|
| 序列设计 | 设计什么氨基酸序列可以折叠成目标结构？ |
| 结构设计 | 设计什么三维结构可以实现目标功能？ |
| 功能设计 | 设计出的蛋白质能否结合目标配体、催化目标反应？ |

EnzyGen2 同时解决前两个层面，并通过配体结合约束间接解决第三个层面。

### 1.3 为什么蛋白质设计如此重要

- **药物开发**：设计能精确结合疾病靶标的蛋白质药物（如抗体、纳米抗体）
- **工业酶工程**：设计在极端条件下工作的工业催化酶（洗涤剂、生物燃料）
- **合成生物学**：设计人工代谢通路中的酶组件
- **疫苗设计**：设计稳定的抗原蛋白以触发免疫反应（如 COVID-19 疫苗中的刺突蛋白工程）

### 1.4 为什么蛋白质设计很难

1. **序列空间指数爆炸**

   一个 300 残基的蛋白质，理论上有 20^300 ≈ 10^390 种可能序列。这个数字远超宇宙中原子的总数（约 10^80）。暴力搜索完全不可能。

2. **序列-结构映射高度非线性**

   一个氨基酸的突变可能导致蛋白质完全无法折叠（misfolding），也可能几乎不影响结构。哪些位点敏感、哪些可以耐受突变，需要深度理解。

3. **功能约束是隐性的**

   蛋白质能否结合特定配体，取决于活性位点的精确三维几何。这种约束很难用简单规则描述。

4. **实验验证成本极高**

   合成一个蛋白质并测试其功能需要数周到数月。每轮设计-合成-测试周期都很昂贵。

### 1.5 当前主要计算方法与其局限

| 方法 | 原理 | 局限 |
|------|------|------|
| 物理模拟（Rosetta） | 分子力场能量最小化 | 计算极慢，受力场精度限制 |
| 定向进化（实验） | 随机突变 + 高通量筛选 | 成本高，搜索空间有限 |
| 反向折叠（ProteinMPNN） | 给定结构→预测序列 | 不生成结构，需要预先知道目标结构 |
| 结构预测（AlphaFold2） | 给定序列→预测结构 | 只做预测，不做设计 |
| 扩散模型（RFDiffusion） | 噪声→结构生成 | 不直接生成序列 |

**EnzyGen2 的不同之处**：**同时**生成序列和结构（协同设计），并且通过配体结合约束确保生成的酶蛋白具有功能活性。这种"序列+结构+功能"三位一体的设计方式是 EnzyGen2 的核心创新。

---

## 2. 酶与配体结合：生化基础

### 2.1 什么是酶

**酶（enzyme）** 是一类具有催化活性的蛋白质，能加速特定化学反应而自身不被消耗：

```
底物（substrate） + 酶 → 产物（product） + 酶
                              ↑
                    反应速度提高 10^6 ~ 10^17 倍
```

酶的催化靠其**活性位点（active site）**——一个三维口袋状结构，底物恰好能嵌入其中。

### 2.2 什么是配体

**配体（ligand）** 是与蛋白质结合的小分子。对于酶来说，配体通常就是底物或辅因子：

```
例子：
  酶：氯霉素乙酰转移酶（ChlR）
  配体/底物：氯霉素（chloramphenicol）
  反应：在氯霉素上加一个乙酰基，使其失去抗菌活性
```

### 2.3 酶的关键结构要素

```
蛋白质整体（数百残基）：
  ┌─────────────────────────────────────────────┐
  │                                             │
  │   [ 框架残基 ]    [ 活性位点残基 ]           │
  │   提供整体折叠    催化反应、结合配体          │
  │   占 ~80%         占 ~20%                    │
  │                                             │
  │              ┌──────────┐                   │
  │              │ 活性口袋 │ ← 配体在此结合     │
  │              │ ●●●●●●● │                    │
  │              └──────────┘                   │
  └─────────────────────────────────────────────┘
```

在 EnzyGen2 中：
- **Motif（功能基序）** = 活性位点残基的位置索引
- **设计目标** = 在保留 Motif 不变的前提下，设计框架残基的序列和结构

### 2.4 配体的表示方式

EnzyGen2 用 **5 维特征向量** 表示每个配体原子：

```
每个配体原子 → [f1, f2, f3, f4, f5]  （5维向量）

其中可能包含：
  - 原子类型（C, N, O, S, P, ...）的独热编码或索引
  - 原子的化学性质（电负性、半径等）
  - 杂化状态信息

配体整体 → [N_atoms, 5] 的矩阵
配体坐标 → [N_atoms, 3] 的矩阵（x, y, z 坐标）
```

### 2.5 NCBI 物种分类学

每个蛋白质来自特定的生物体，不同物种的酶有不同的进化偏好：

```
NCBI Taxonomy ID 示例：
  9606    → Homo sapiens（人）
  562     → Escherichia coli（大肠杆菌）
  83332   → Mycobacterium tuberculosis（结核分枝杆菌）
  83333   → Escherichia coli K-12
  
用途：同一类酶在不同物种中有不同的序列偏好和折叠特征
```

EnzyGen2 将 NCBI Taxonomy ID 编码为可学习的 embedding，帮助模型理解物种间的序列差异。

---

## 3. 序列-结构协同设计：为什么比单独设计更好

### 3.1 单独设计的问题

```
方案 A：只设计序列
  序列 → [AlphaFold2] → 结构（可能不是你想要的）
  缺点：无法控制结构，可能生成不可折叠的序列

方案 B：只设计结构
  结构 → [ProteinMPNN] → 序列
  缺点：需要预先知道目标结构，结构从何而来？

方案 C（EnzyGen2）：同时设计序列和结构
  Motif + 配体 → [EnzyGen2] → 序列 + 结构
  优势：序列和结构互相约束，确保一致性
```

### 3.2 协同设计的直觉

想象你在同时画一幅画的轮廓和颜色：

- 如果先画完轮廓再填色，颜色可能与轮廓不协调
- 如果先填色再画轮廓，轮廓可能与颜色不匹配
- 如果**同时**画轮廓和颜色，每一笔都考虑两者的协调性

EnzyGen2 的做法就是第三种：在 Transformer 层（处理序列）和 EGNN 层（处理结构）之间**交替运行**，让信息双向流动。

---

## 4. EnzyGen2 解决什么任务

### 4.1 任务定义

```
输入：
  1. 蛋白质序列（带遮蔽的氨基酸序列）
  2. 蛋白质 Cα 坐标（带遮蔽的三维坐标）
  3. Motif 索引（哪些残基是功能基序，不能改变）
  4. NCBI Taxonomy ID（物种标识）
  5. [可选] 配体原子特征 + 配体坐标 + 结合标签

输出：
  1. 完整蛋白质序列（每个位置 20 种氨基酸的概率分布）
  2. 完整蛋白质 Cα 骨架坐标 [L, 3]
  3. [可选] 蛋白质-配体结合概率 [2]（结合/不结合）
```

### 4.2 设计范式：Motif-Scaffolding

```
输入示例（长度 L=200 的蛋白质）：

序列：  M K V A _ _ _ _ _ _ _ L L K S _ _ _ _ A
Motif：  ✓ ✓ ✓ ✓             ✓ ✓ ✓ ✓         ✓
遮蔽：              ✗ ✗ ✗ ✗ ✗ ✗ ✗         ✗ ✗ ✗ ✗

模型任务：
  1. 预测所有遮蔽位置 (✗) 的氨基酸类型
  2. 预测所有遮蔽位置 (✗) 的 Cα 三维坐标
  3. 保持 Motif 位置 (✓) 的序列和坐标不变
```

### 4.3 三个下游应用

EnzyGen2 论文中展示了三个具体的酶设计应用：

| 酶名称 | 缩写 | Rhea反应ID | 功能 |
|-------|------|----------|------|
| 氯霉素乙酰转移酶 | ChlR | 18421 | 催化氯霉素的乙酰化，使其失活 |
| 氨基糖苷腺苷酰转移酶 | AadA | 20245 | 催化氨基糖苷类抗生素的腺苷酰化 |
| 硫嘌呤甲基转移酶 | TPMT | - | 催化硫嘌呤药物的甲基化 |

这三种酶都与**抗生素抗性**或**药物代谢**相关，设计新的变体有重要的药学应用价值。

### 4.4 与先前方法的对比

| 特性 | ProteinMPNN | RFDiffusion | EnzyGen (v1) | EnzyGen2 |
|------|------------|-------------|-------------|----------|
| 输入 | 固定结构 | 噪声/条件 | 序列+结构+Motif | 序列+结构+Motif+配体 |
| 输出 | 仅序列 | 仅结构 | 序列+结构 | 序列+结构+结合预测 |
| 配体感知 | 否 | 否 | 否 | **是** |
| 物种信息 | 否 | 否 | 否 | **是（NCBI embedding）** |
| 训练范式 | 反向折叠 | 扩散去噪 | MLM+结构 | **三阶段渐进** |

### 4.5 Motif-Scaffolding 的数学形式化

上一节用示意图展示了 Motif-Scaffolding 的概念，但要真正理解 EnzyGen2 的训练和推理逻辑，我们需要把它严格地写成数学语言。

#### 4.5.1 问题定义

设蛋白质总长度为 L，残基集合为 R = {1, 2, ..., L}。我们把 R 拆成两个不重叠的子集：

```
Motif 残基集合  M ⊂ R   （已知的功能残基，不可修改）
Scaffold 残基集合 S = R \ M   （需要设计的骨架残基）
```

对于每个残基 i ∈ R，我们关心两个量：

```
s_i ∈ {A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y}   ← 氨基酸类型（20种之一）
x_i ∈ ℝ³   ← Cα 原子的三维坐标
```

**已知信息**：Motif 残基的序列 s_M = {s_i | i ∈ M} 和坐标 x_M = {x_i | i ∈ M}。

**设计目标**：找到 Scaffold 残基的序列 s_S 和坐标 x_S，使得：

```
(1) 序列目标：   s_S = argmax P(s_S | s_M, x_M)
    → Scaffold 位置的氨基酸序列应当最大化条件概率
    → 直觉：给定活性位点的已知残基，什么样的骨架序列最"自然"

(2) 结构目标：   x_S = argmin RMSD(x_S, x_S^true)
    → Scaffold 位置的预测坐标应尽量接近真实坐标
    → 直觉：骨架的三维形状要跟天然结构吻合

(3) 物理可行性：蛋白质 (M ∪ S) 必须满足物理约束
    → 相邻残基的 Cα 距离 ≈ 3.8 Å（肽键长度约束）
    → 无原子碰撞（范德华排斥）
    → 可折叠（自由能最低）
```

#### 4.5.2 motif_mask 的编码方式（反直觉！）

在代码中，motif 信息通过一个二值向量 `motif_mask` 编码，**但它的含义和你直觉想的是反的**：

```
motif_mask[i] = 0  →  位置 i 是 Motif（已知，保留不变）
motif_mask[i] = 1  →  位置 i 是 Scaffold（未知，需要设计）
```

为什么是反的？因为在代码中 `motif_mask` 实际上表示"需要遮蔽（mask）的位置"，即"需要预测的位置"。Motif 位置不需要预测，所以 mask=0（不遮蔽）；Scaffold 位置需要预测，所以 mask=1（遮蔽）。

**具体数值示例**：

```
蛋白质长度 L = 10（简化示例）
序列（真值）：  M  K  V  A  L  G  A  A  G  G
Motif 位置：     1  2  3  4           8  9  10   ← 已知的功能残基
Scaffold 位置：              5  6  7              ← 需要设计的骨架残基

motif_mask:    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
                ↑Motif      ↑Scaffold   ↑Motif

src_tokens（输入）:  [20, 15, 7, 5, 32, 32, 32, 5, 13, 13]
                     ↑M   ↑K  ↑V ↑A  ↑MASK       ↑A ↑G  ↑G

src_tokens（目标）:  [20, 15, 7, 5, 10, 13, 5,  5, 13, 13]
                                    ↑L   ↑G  ↑A
```

注意看：Scaffold 位置（5、6、7）在输入中被替换为 `<mask>`（索引 32），但在目标中保留真实氨基酸。

#### 4.5.3 坐标处理

```
Motif 位置的坐标：保留原始 PDB 中的真实 Cα 坐标
  x_motif = [[12.5, -3.2, 8.1],     ← 位置 1 的 Cα 坐标（Å）
             [15.8, -1.1, 7.3],     ← 位置 2
             ...]

Scaffold 位置的坐标：随机初始化为均匀噪声
  x_scaffold = uniform(-1, 1) × scale   ← 每个坐标分量在 [-1, 1] 范围
  
  例如：x_scaffold[5] = [0.32, -0.78, 0.15]   ← 完全随机的初始坐标
```

模型的任务就是：从这些随机坐标出发，经过 3 层 EGNN 的逐步修正，把 Scaffold 位置的坐标"推"到正确的三维位置。

#### 4.5.4 为什么这种形式化是优雅的

1. **统一框架**：序列设计和结构预测用同一个模型完成，共享 Transformer 表示
2. **条件生成**：Motif 信息作为条件（已知的序列 + 坐标）自然地流入注意力机制
3. **端到端可微**：从 mask 输入到序列概率 + 坐标预测，整个计算图是可微的，可以用标准梯度下降训练
4. **灵活的 Motif 比例**：通过调整 motif_mask 中 0 和 1 的比例，可以控制"保留多少已知信息"

---

## 5. 系统整体架构一览

### 5.1 完整架构图

```
┌───────────────────────────────────────────────────────────────────────┐
│                        EnzyGen2 整体架构                              │
└───────────────────────────────────────────────────────────────────────┘

输入数据:
  src_tokens [B, L]   ← 蛋白质序列（部分被 MASK）
  coors [B, L, 3]     ← Cα 坐标（部分被随机初始化）
  motif [B, L]        ← 二值遮蔽掩码
  ncbi [B]            ← 物种 ID
  [ligand_atom] [B, N, 5]    ← 配体原子特征（可选）
  [ligand_coor] [B, N, 3]    ← 配体坐标（可选）
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│                    Token Embedding                       │
│                                                         │
│  embed_tokens(tokens) × embed_scale + ncbi_embedding    │
│  → [B, L, 1280]                                        │
└─────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│              ESM2 Transformer (33 层)                    │
│                                                         │
│  ┌─────────────────────────┐                            │
│  │ Transformer Layer 1-11  │ → Self-Attention + FFN     │
│  └────────────┬────────────┘                            │
│               ▼                                         │
│  ┌─────────────────────────┐                            │
│  │ EGNN Layer 0            │ → 边模型+坐标更新+节点更新  │
│  │ (等变图神经网络)         │   重建 KNN 图              │
│  └────────────┬────────────┘                            │
│               ▼                                         │
│  ┌─────────────────────────┐                            │
│  │ Transformer Layer 12-22 │ → Self-Attention + FFN     │
│  └────────────┬────────────┘                            │
│               ▼                                         │
│  ┌─────────────────────────┐                            │
│  │ EGNN Layer 1            │ → 边模型+坐标更新+节点更新  │
│  │ (等变图神经网络)         │   重建 KNN 图              │
│  └────────────┬────────────┘                            │
│               ▼                                         │
│  ┌─────────────────────────┐                            │
│  │ Transformer Layer 23-33 │ → Self-Attention + FFN     │
│  └────────────┬────────────┘                            │
│               ▼                                         │
│  ┌─────────────────────────┐                            │
│  │ EGNN Layer 2            │ → 边模型+坐标更新+节点更新  │
│  └────────────┬────────────┘                            │
│               ▼                                         │
│  LayerNorm + LM Head → Softmax                          │
└─────────────────────────────────────────────────────────┘
       │                          │
       ▼                          ▼
  encoder_prob [B, L, 33]    coords [B, L, 3]
  (序列概率分布)              (预测的 Cα 坐标)
       │                          │
       │    ┌─────────────────────┘
       │    │  (如果有配体)
       ▼    ▼
┌─────────────────────────────────────────────────────────┐
│            SubstrateEGNN（配体处理）                      │
│                                                         │
│  ligand_atom [B,N,5] → Linear(5→1280) → EGNN(3层)      │
│  → sub_feats [B, 1280]（池化后）                         │
│                                                         │
│  protein_rep [B, 1280] ← 序列表示求和池化                │
│  concat(protein_rep, sub_feats) → Linear(2560→2)        │
│  → scores [B, 2]（结合/不结合概率）                      │
└─────────────────────────────────────────────────────────┘
       │
       ▼
  输出: encoder_prob, coords, [scores]
```

### 5.2 关键文件对应

```
fairseq/models/
├── geometric_protein_model.py  ← 核心模型（GeometricProteinNCBIModel + SubstrateModel）
├── egnn.py                     ← 等变图神经网络（E_GCL, EGNN, SubstrateEGNN）
├── esm.py                      ← ESM2 蛋白质语言模型
└── esm_modules.py              ← ESM2 子模块（Alphabet, TransformerLayer, LMHead）

fairseq/criterions/
├── geometric_protein_ncbi_loss.py         ← 序列+结构损失
└── geometric_protein_ncbi_ligand_loss.py  ← 序列+结构+结合损失

fairseq/tasks/
└── geometric_protein_design.py  ← 任务定义（数据加载+推理逻辑）

fairseq/data/
├── indexed_dataset.py           ← 所有数据集实现（序列、坐标、Motif、NCBI、配体）
├── ncbi_protein_dataset.py      ← 预训练批次整合（含配体）
└── ncbi_protein_finetune_dataset.py  ← 微调批次整合（不含配体）

fairseq_cli/
├── train.py                     ← 预训练脚本
├── finetune.py                  ← 微调脚本
├── generate.py                  ← 推理/生成脚本
└── generate_pdb_file.py         ← PDB 文件生成工具
```


### 5.3 数据流的维度追踪

理解 EnzyGen2 最重要的一步是搞清楚**张量形状在每一步怎么变化**。下面以一个具体的批次为例，逐步追踪整个数据流。

#### 设定

```
B = 2      （批次大小：同时处理 2 条蛋白质）
L = 152    （序列长度：包括 <cls> 和 <eos> 两个特殊 token，实际蛋白质长度为 150）
D = 1280   （隐藏维度：ESM2-650M 的 embedding 维度）
V = 33     （词表大小）
K = 30     （KNN 图的邻居数）
```

#### 逐步追踪

```
步骤 1：Token Embedding
  输入：  src_tokens [2, 152]       整数序列，值域 0 到 32
  操作：  embed_tokens(tokens) * embed_scale，其中 embed_tokens = nn.Embedding(33, 1280)
  输出：  x [2, 152, 1280]         浮点张量

  具体数值：假设第一个蛋白质的前 5 个 token 是 [0, 20, 15, 7, 5]
  （即 <cls>, M, K, V, A），embed_tokens 查表后得到 5 个 1280 维向量。

步骤 2：NCBI Taxonomy Embedding 加法
  输入：  ncbi_ids [2]              两个物种的 taxonomy ID，例如 [9606, 562]
                                    9606 = 人类（Homo sapiens），562 = 大肠杆菌（E. coli）
  操作：  x = x + ncbi_embed(ncbi_ids).unsqueeze(1)
  说明：  ncbi_embed(ncbi_ids) 的形状是 [2, 1280]
         unsqueeze(1) 后变成 [2, 1, 1280]
         加到 x [2, 152, 1280] 时，第 1 维广播到 152
  输出：  x [2, 152, 1280]         每个位置都加上了相同的物种向量

  直觉：物种向量就像给整条蛋白质打上一个"物种标签"。人类蛋白和大肠杆菌蛋白
  即使序列相似，它们的密码子偏好、折叠环境（温度、pH）都不同，
  物种 embedding 帮助模型捕捉这种差异。

步骤 3：Token Dropout（仅训练时）
  操作：  把 <mask> 位置的 embedding 置零，然后缩放
  输出：  x [2, 152, 1280]         形状不变，但值被修改
  详见第 9 节的完整解释。

步骤 4：坐标初始化
  输入：  coors [2, 152, 3]
  说明：  3 代表 (x, y, z) 三个空间维度，单位 Angstrom（埃）
         Motif 位置保留真实 PDB 坐标，例如 [12.5, -3.2, 8.1]
         Scaffold 位置是随机初始化的，例如 [0.32, -0.78, 0.15]

步骤 5：维度转置（为 Transformer 准备）
  操作：  x = x.transpose(0, 1)    Transformer 期望 (L, B, D) 格式
  输出：  x [152, 2, 1280]

  为什么要转置？PyTorch 的 nn.MultiheadAttention 默认期望输入格式为
  (seq_len, batch, embed_dim)。这是 PyTorch 的历史设计选择，与 TensorFlow 不同。

步骤 6：Transformer 第 1-11 层
  每一层的操作（Pre-LN 风格）：
    residual = x
    x = self_attn_layer_norm(x)                    LayerNorm(1280)
    x = residual + self_attn(x, x, x)              20 头自注意力，含 RoPE
    residual = x
    x = final_layer_norm(x)                        LayerNorm(1280)
    x = residual + fc2(gelu(fc1(x)))               前馈网络 1280 -> 5120 -> 1280

  注意力计算细节（单个头）：
    Q = x @ W_Q    [152, 2, 64]    （head_dim = 1280/20 = 64）
    K = x @ W_K    [152, 2, 64]
    V = x @ W_V    [152, 2, 64]
    Q_rot = RoPE(Q, positions)      对 Q 施加旋转位置编码
    K_rot = RoPE(K, positions)      对 K 施加旋转位置编码
    attn = softmax(Q_rot @ K_rot^T / sqrt(64))    [2, 152, 152]
    out = attn @ V    [152, 2, 64]

  20 个头的输出拼接后：[152, 2, 1280]，再经过输出投影 W_O。

  输出：  x [152, 2, 1280]         经过 11 层 Transformer 精炼后的表示

步骤 7：转置回 (B, L, D)，传入 EGNN 0
  操作：  x = x.transpose(0, 1)    EGNN 期望 (B, L, D) 格式
  输入到 EGNN：
    h [2, 152, 1280]     节点特征（即 x）
    coors [2, 152, 3]    当前坐标
  EGNN 内部：
    1. 构建 KNN 图：对每个残基，计算它与所有其他残基的欧氏距离，
       选最近的 K=30 个作为邻居。共 152 * 30 = 4560 条有向边。
    2. 计算边特征：两端节点特征拼接 [2560] + 距离 [1] + 距离差 [1] = [2562]
    3. 边 MLP：Linear(2562, 1280) -> SiLU -> Linear(1280, 1280) -> SiLU
    4. 坐标更新：对每个节点 i，计算
       delta_x_i = (1/30) * sum_j (x_i - x_j) * phi_coord(m_ij)
       其中 phi_coord 是一个标量函数，输出每条边的坐标更新权重
       coors_new = coors + delta_x
    5. 节点更新：h_i = h_i + (1/30) * sum_j m_ij
       将邻居的消息聚合后加回节点特征
  输出：
    h [2, 152, 1280]     更新后的节点特征
    coors [2, 152, 3]    更新后的坐标（Scaffold 位置被修正了一步）

步骤 8：转置回 (L, B, D)，继续 Transformer 第 12-22 层
  操作：  x = h.transpose(0, 1)    回到 (L, B, D) 格式
  经过 11 层 Transformer（与步骤 6 相同的结构，但权重不同）
  输出：  x [152, 2, 1280]

步骤 9：传入 EGNN 1（重复步骤 7 的过程）
  关键区别：KNN 图用新坐标重新构建！因为经过 EGNN 0 后，Scaffold 位置的坐标
  已经被推向了更合理的位置，所以邻居关系可能发生变化。
  输出：h [2, 152, 1280], coors [2, 152, 3]

步骤 10：Transformer 第 23-33 层
  输出：  x [152, 2, 1280]

步骤 11：传入 EGNN 2（第三次也是最后一次几何修正）
  输出：h [2, 152, 1280], coors [2, 152, 3]

步骤 12：LayerNorm + LM Head
  操作：  x = emb_layer_norm_after(h)     LayerNorm(1280)
         logits = lm_head(x)             包含 Linear(1280, 1280) + GELU + LayerNorm + Linear(1280, 33)
  输出：  logits [2, 152, 33]            每个位置对 33 个 token 的未归一化分数

步骤 13：Softmax（推理时）/ CrossEntropy（训练时）
  推理：  probs = softmax(logits, dim=-1)
         probs [2, 152, 33]              概率分布，每行加和为 1
         例如位置 5（一个 Scaffold 位置）的概率分布可能是：
         [0.0, 0.0, 0.0, 0.0,           <- 特殊 token（概率接近 0）
          0.35, 0.02, 0.01, 0.15, ...    <- A=0.35, C=0.02, D=0.01, E=0.15, ...
          0.0, 0.0, ...]                 <- 非标准 token
         argmax = 4（对应 A，丙氨酸），模型预测该位置最可能是 A

  训练：  loss = CrossEntropyLoss(logits, target_tokens)
         仅在 Scaffold 位置计算损失（Motif 位置是已知的，不需要预测）
```

#### 关键数字汇总

```
总参数量的分布：
  Token Embedding：     33 * 1280 = 42,240（约 0.04M）
  Transformer 33 层：   33 * (4*1280^2 + 2*1280*5120 + 4*1280) = 约 649M
  EGNN 3 层：           3 * (2*1280*1280 + 1280 + ...) = 约 15M
  NCBI embedding：      约 1M
  LM Head：             1280*1280 + 1280*33 = 约 1.7M
  SubstrateEGNN（可选）：约 8M
  总计：                约 674M
```

### 5.4 为什么是 33 层 Transformer + 3 层 EGNN

这个架构设计的核心数字关系是：33 层 Transformer 除以 3 层 EGNN = 每 11 层 Transformer 后插入 1 层 EGNN。

#### 为什么是这个比例？

**原因 1：ESM2 的固有结构决定了总层数**

ESM2-650M 恰好有 33 层 Transformer。这不是 EnzyGen2 选择的——这是 Meta AI 在设计 ESM2 时通过 scaling law 实验确定的最佳层数（对于 650M 参数量而言）。EnzyGen2 的设计约束是：在这 33 层中均匀插入 EGNN 层。

33 的因数有：1, 3, 11, 33。每种因数对应一种插入方案：

| EGNN 层数 | 每隔几层 Transformer 插入 | 问题 |
|-----------|-------------------------|------|
| 33 | 1 | EGNN 参数量爆炸（33*5M=165M），训练极慢 |
| 11 | 3 | 计算量大，训练慢 3 倍以上，且 3 层 Transformer 信息不足 |
| **3** | **11** | **最佳平衡：足够的序列理解 + 适度的几何修正** |
| 1 | 33 | 几何修正太少，只在最后修正一次，RMSD 显著增大 |

**原因 2：计算效率 vs 几何精度的权衡**

EGNN 层的作用是修正三维坐标。每次 EGNN 调用需要：
- 构建 KNN 图：O(L^2) 的距离计算（对 L=150 的蛋白质，约 22500 次距离计算）
- 消息传递：O(L * K * D) 的矩阵运算（150 * 30 * 1280 = 约 576 万次浮点运算）

如果太频繁地插入 EGNN（比如每 3 层），EGNN 的总计算量会超过 Transformer 本身，形成瓶颈。而且，坐标改善呈边际递减——前几次 EGNN 调用的改善最大，后续的改善越来越小。

**原因 3：信息积累的物理直觉**

Transformer 层的作用是从序列角度"理解"蛋白质。考虑一个 150 残基的蛋白质：
- 经过 1 层 Transformer：每个位置只能"看到"直接邻居的信息（感受野约 3-5 个残基）
- 经过 3 层 Transformer：信息传播范围约 10-15 个残基
- 经过 11 层 Transformer：信息已在整条序列中充分传播，每个位置都包含全局上下文

只有在"理解了整条序列"之后进行坐标更新，EGNN 才能做出高质量的决策。比如，一个远端的 loop 区域的坐标更新，需要知道活性位点（可能在序列另一端）的信息——这需要足够多的 Transformer 层来传播。

**直觉类比**：

想象你在拼一个 3D 拼图（蛋白质折叠）：
- 方案 A：每看 3 块拼图就尝试放一块到 3D 模型上（太仓促，容易放错）
- 方案 B：每看 11 块拼图就尝试放一块（有足够信息做出正确判断）
- 方案 C：看完所有拼图再一次性放好（信息充分，但没有中间纠错的机会）

EnzyGen2 选择了方案 B——在信息积累与几何反馈之间取得最佳平衡。3 次 EGNN 调用就像 3 轮"校准"：第一轮粗调（EGNN 0），第二轮微调（EGNN 1），第三轮精修（EGNN 2）。


---

## 6. 蛋白质语言模型的直觉

### 6.1 类比自然语言

蛋白质语言模型（PLM）是将 NLP 中 BERT/GPT 的思路应用到蛋白质序列上：

```
自然语言：
  词表大小：50,000+ 个英文单词/子词
  训练任务：遮蔽语言模型（MLM）—— "The cat sat on the [MASK]" → "mat"
  训练数据：数十亿句子
  学到的知识：语法规则、语义关系、世界知识

蛋白质"语言"：
  词表大小：20 种氨基酸 + 特殊 token ≈ 33 个 token
  训练任务：遮蔽语言模型（MLM）—— "MKVAVL[MASK]AAGG..." → "G"
  训练数据：UniRef50 中约 2.5 亿条蛋白质序列
  学到的知识：进化共变关系、结构约束、功能位点模式
```

### 6.2 为什么 PLM 的 embedding 包含结构和功能信息

经过大规模预训练后，PLM 学到的不只是"哪个氨基酸出现在哪"：

- **进化共变**：如果位置 i 是 V，位置 j 往往是 L（因为它们在三维空间中接触，需要互补）
- **结构约束**：PLM 的注意力头自然学会了关注空间上相近的残基
- **功能标记**：活性位点残基有独特的 embedding 模式（高度保守、与特定模式共现）

这就是为什么 EnzyGen2 选择 ESM2 作为编码器——它提供了丰富的序列-结构-功能先验知识。


### 6.3 为什么 ESM2 的 embedding 包含三维信息——实验证据

上一节提到 PLM 学到了结构信息，这听起来可能不可思议——模型从未见过任何三维坐标，怎么可能学到空间信息？这里给出具体的实验证据。

#### 证据 1：注意力图与接触图的相关性

2021 年 Rao 等人在论文 "Transformer protein language models are unsupervised structure learners" 中展示了一个惊人的发现：

```
实验设置：
  1. 取一条蛋白质序列，输入 ESM-1b（ESM2 的前身）
  2. 提取所有注意力头的注意力权重 A[h, i, j]
     A[h, i, j] = 注意力头 h 中，位置 i 对位置 j 的注意力分数
     形状：[n_heads * n_layers, L, L] = [20 * 33, 150, 150]
  3. 用一个简单的逻辑回归模型，从注意力权重预测残基接触图
     接触图定义：如果残基 i 和 j 的 Cb 原子距离 < 8 Angstrom，则 contact[i,j] = 1

结果：
  仅用注意力权重，不用任何坐标信息：
  - 接触预测的 Top-L 精度 > 70%（L 为序列长度）
  - 某些注意力头几乎完美地对应了三维接触关系
  - 浅层（1-10 层）的注意力头主要捕捉局部接触（序列距离 < 12）
  - 深层（20-33 层）的注意力头主要捕捉远程接触（序列距离 > 24）
```

这意味着什么？Transformer 通过学习"氨基酸 A 出现时，位置 j 处的氨基酸 B 往往出现"这种共变模式，**隐式地学到了三维空间邻近关系**。因为在进化中，空间上接触的残基会共同进化（一个突变时另一个也需要补偿性突变），这种共进化信号被 PLM 的注意力机制捕获。

#### 证据 2：ESM2 embedding 的主成分分析

如果把一条蛋白质所有残基的 ESM2 embedding（每个残基一个 1280 维向量）进行主成分分析（PCA），取前 3 个主成分：

```
embedding PCA 结果：
  PC1, PC2, PC3 与蛋白质三维结构的 x, y, z 坐标高度相关

  例如：一个 alpha 螺旋区域的残基，它们的 embedding 在 PCA 空间中
  会形成螺旋状的排列，与真实 3D 结构中的螺旋几何吻合

  一个 beta 折叠区域的残基，它们的 embedding 在 PCA 空间中
  会形成平面状的分布，与真实的 beta 折叠几何一致
```

#### 对 EnzyGen2 的意义

EnzyGen2 利用了 ESM2 embedding 中的隐式 3D 信息。当 EGNN 接收到 Transformer 的输出时，它不是从零开始推断坐标——而是从已经包含大量 3D 线索的 embedding 出发。这就像给 EGNN 一个"粗略的 3D 草图"，EGNN 只需要在此基础上精修，而不是从白纸画起。

### 6.4 EnzyGen2 中 PLM 的双重角色

在 EnzyGen2 中，ESM2 同时扮演两个角色，这是理解整个架构的关键。

#### 角色 1：特征提取器（Feature Extractor）

```
输入：氨基酸序列 tokens [B, L]
输出：每个残基的 embedding 向量 [B, L, 1280]

这些 embedding 编码了什么：
  - 进化信息：该位置在自然界中最常见的氨基酸类型
  - 二级结构倾向：该位置可能是 alpha 螺旋、beta 折叠还是 loop
  - 溶剂暴露度：该位置可能在蛋白质表面还是内部
  - 功能位点标记：该位置是否可能是活性位点的一部分
  - 保守性：该位置在进化中有多保守（保守 = 功能重要）
```

这些 embedding 被传递给 EGNN，作为节点特征参与消息传递。换句话说，ESM2 的输出是 EGNN 的输入——它们共享同一个表示空间。

#### 角色 2：生成模型（Generative Model）

```
输入：最后一层 Transformer + EGNN 2 的输出 [B, L, 1280]
操作：LM Head（线性层 + GELU + LayerNorm + 线性层）
输出：每个位置对 33 个 token 的概率分布 [B, L, 33]

LM Head 的结构：
  x [B, L, 1280]
  -> Linear(1280, 1280)    权重与 embed_tokens 不共享
  -> GELU 激活
  -> LayerNorm(1280)
  -> Linear(1280, 33)      权重与 embed_tokens.weight 共享！（Weight Tying）
  -> logits [B, L, 33]
```

**Weight Tying 是什么？** LM Head 的最后一层线性变换的权重矩阵 [33, 1280] 与 embed_tokens 的权重矩阵 [33, 1280] 是**同一个张量**。这意味着：
- embed_tokens：把 token 索引映射到 1280 维空间（token -> embedding）
- LM Head 最后一层：把 1280 维空间映射回 token 概率（embedding -> token）
- 两者互为"逆过程"，共享权重确保了映射的一致性

#### 关键洞察：没有独立的 Encoder 和 Decoder

传统的序列到序列模型（如机器翻译）有独立的编码器和解码器。但 EnzyGen2 不同：

```
传统 Seq2Seq：
  Encoder（理解输入） -> 中间表示 -> Decoder（生成输出）
  两个独立的网络，参数不共享

EnzyGen2：
  ESM2 的 33 层 Transformer 既是 Encoder 又是 Decoder
  输入：部分遮蔽的序列
  输出：完整的序列概率分布

  这就是 BERT 式的 MLM 范式——
  没有独立的解码器，整个 Transformer 就是一个"填空模型"
```

这种设计的优势是参数高效——不需要额外训练一个解码器。劣势是生成过程不是自回归的（不能逐个 token 生成），而是一次性预测所有遮蔽位置。


---

## 7. ESM2 的具体结构与参数

### 7.1 EnzyGen2 使用的 ESM2 版本

```
esm2_t33_650M_UR50D（Meta AI, 2022）：
  参数量：650M（6.5 亿参数）
  Transformer 层数：33（t33 的含义）
  每层隐藏维度：1280
  前馈网络维度：5120（= 4 × 1280）
  注意力头数：20
  训练数据：UniRef50（2021年3月版，约2.5亿序列）
  位置编码：Rotary Position Embedding（RoPE）
  激活函数：GELU
```

### 7.2 ESM2 在 EnzyGen2 中的使用方式

**重要**：EnzyGen2 **不是直接调用** ESM2 的 forward 方法，而是**拆开使用**其内部组件。

```python
# 来自 geometric_protein_model.py (line 200-202)
@classmethod
def build_encoder(cls, args, src_dict, embed_tokens):
    model, alphabet = load_from_pretrained_models(args.pretrained_esm_model)
    return model  # 返回 ESM2 模型实例

# 在 forward() 中（line 225）：
x = self.encoder.embed_scale * self.encoder.embed_tokens(tokens)
# 然后手动遍历 self.encoder.layers，在每 11 层后插入 EGNN
```

这种"拆开使用"是关键设计：让 Transformer 层和 EGNN 层交替执行。

### 7.3 ESM2 的子模块

```python
ESM2 实例的子模块（来自 esm.py）：
├── embed_tokens        ← nn.Embedding(33, 1280)   词嵌入
├── embed_scale         ← 1（ESM2 不缩放 embedding）
├── layers              ← ModuleList[33 × TransformerLayer]
│   └── TransformerLayer:
│       ├── self_attn_layer_norm  ← LayerNorm(1280)
│       ├── self_attn             ← MultiheadAttention(1280, 20 heads, RoPE)
│       ├── final_layer_norm      ← LayerNorm(1280)
│       ├── fc1                   ← Linear(1280 → 5120)
│       └── fc2                   ← Linear(5120 → 1280)
├── emb_layer_norm_after ← LayerNorm(1280)
├── lm_head              ← RobertaLMHead(1280 → 33, weight tying)
└── contact_head         ← ContactPredictionHead（EnzyGen2 不使用）
```


### 7.4 ESM2-650M 的具体参数细节

让我们精确拆解 ESM2-650M 的每一层参数，帮你建立对 650M 这个数字的直觉。

#### 单个 Transformer 层的参数量

```
Multi-Head Self-Attention（20 个头，head_dim = 64，d_model = 1280）：
  W_Q: [1280, 1280] + b_Q: [1280]     = 1,638,400 个参数
  W_K: [1280, 1280] + b_K: [1280]     = 1,638,400 个参数
  W_V: [1280, 1280] + b_V: [1280]     = 1,638,400 个参数
  W_O: [1280, 1280] + b_O: [1280]     = 1,638,400 个参数
  ──────────────────────────────────────
  注意力总计：4 * 1280 * 1280 + 4 * 1280 = 6,558,720 (约 6.6M)

  为什么是 4 个矩阵？Q/K/V 各一个投影矩阵，加上输出投影 W_O。
  20 个头并不增加参数量——20 个头共享 W_Q/W_K/W_V/W_O，
  只是在计算时把 1280 维切成 20 个 64 维的子空间。

前馈网络 FFN（1280 -> 5120 -> 1280，带 GELU 激活）：
  W_1: [1280, 5120] + b_1: [5120]     = 6,558,720 个参数
  W_2: [5120, 1280] + b_2: [1280]     = 6,554,880 个参数
  ──────────────────────────────────────
  FFN 总计：约 13,113,600 (约 13.1M)

  为什么中间维度是 5120 = 4 * 1280？
  这是 Transformer 的标准设计：FFN 的中间维度通常是 4 倍于 d_model。
  更大的中间维度 = 更强的非线性表达能力。
  GELU（Gaussian Error Linear Unit）比 ReLU 更平滑，梯度更稳定。

LayerNorm（两个）：
  self_attn_layer_norm: gamma [1280] + beta [1280] = 2,560 个参数
  final_layer_norm:     gamma [1280] + beta [1280] = 2,560 个参数
  ──────────────────────────────────────
  LayerNorm 总计：5,120 个参数（可以忽略不计）

单层总计：6,558,720 + 13,113,600 + 5,120 = 19,677,440 (约 19.7M)
```

#### 全模型参数量

```
33 层 Transformer：  33 * 19,677,440 = 649,355,520 (约 649M)

Embedding 层：
  embed_tokens: [33, 1280] = 42,240
  （ESM2 的 embed_scale = 1，没有额外的缩放参数）

最终 LayerNorm：
  emb_layer_norm_after: gamma [1280] + beta [1280] = 2,560

LM Head (RobertaLMHead)：
  Linear(1280, 1280): 1,280 * 1,280 + 1,280 = 1,639,680
  LayerNorm(1280): 2,560
  Linear(1280, 33): 与 embed_tokens 共享权重（Weight Tying），不额外计算
  LM Head 总计：约 1.64M

Contact Head：ESM2 原版有，但 EnzyGen2 不使用

总参数量：
  649M + 0.04M + 0.003M + 1.64M = 约 651M
  Meta AI 四舍五入称之为 "650M"
```

#### 参数量的直觉

```
650M 参数意味着什么？
  - 存储空间：FP32 精度下约 2.5 GB，FP16 精度下约 1.25 GB
  - 单次前向传播（L=150）的浮点运算量：约 200 GFLOPs
  - 在 A100 GPU（312 TFLOPS FP16）上：单次前向约 0.6 毫秒（理论值）
  - 实际训练中（含反向传播 + 梯度更新）：每步约 50-200 毫秒

对比其他 ESM2 版本：
  esm2_t6_8M_UR50D:    6 层，8M 参数     （最小版本，实验用）
  esm2_t12_35M_UR50D:  12 层，35M 参数    （轻量版）
  esm2_t30_150M_UR50D: 30 层，150M 参数   （中等版本）
  esm2_t33_650M_UR50D: 33 层，650M 参数   （EnzyGen2 使用的版本）
  esm2_t36_3B_UR50D:   36 层，3B 参数     （大版本）
  esm2_t48_15B_UR50D:  48 层，15B 参数    （最大版本）

EnzyGen2 选择 650M 版本的原因：
  - 足够大以捕获丰富的序列-结构关系
  - 足够小以在单张 A100 GPU 上训练（3B 和 15B 需要多卡并行）
  - 33 层恰好可以被 3 整除，方便均匀插入 EGNN
```

### 7.5 ESM2 在交错架构中的信息流

EnzyGen2 最巧妙的设计是把 ESM2 的 33 层 Transformer"拆开"，在中间插入 EGNN。这里详细追踪每个"交接点"（handoff point）发生了什么。

#### 交接点的代码逻辑

在 `geometric_protein_model.py` 的 `forward()` 方法中，核心循环大致如下：

```python
# 遍历 33 层 Transformer
for layer_idx, layer in enumerate(self.encoder.layers):
    x = layer(x, ...)              # Transformer 层的正常前向传播

    if (layer_idx + 1) % 11 == 0:  # 每 11 层触发一次 EGNN
        egnn_idx = (layer_idx + 1) // 11 - 1   # 0, 1, 2

        # 步骤 A：维度转置
        x = x.transpose(0, 1)      # [L, B, 1280] -> [B, L, 1280]

        # 步骤 B：EGNN 前向传播
        h, coors = self.egnn_layers[egnn_idx](x, coors)

        # 步骤 C：维度转回
        x = h.transpose(0, 1)      # [B, L, 1280] -> [L, B, 1280]
```

#### 为什么需要转置？

这是因为 ESM2 和 EGNN 对张量维度的约定不同：

```
ESM2 Transformer 层：
  输入/输出格式：[L, B, D] = [seq_len, batch, embed_dim]
  原因：PyTorch 的 nn.MultiheadAttention 历史默认格式
  在这种格式下，同一个位置的不同样本的数据在内存中是连续的

EGNN 层：
  输入/输出格式：[B, L, D] = [batch, seq_len, embed_dim]
  原因：图神经网络通常以 batch-first 格式处理节点特征
  在这种格式下，同一个样本的不同位置的数据在内存中是连续的
  这对 KNN 图构建很重要——需要快速访问同一个样本中所有节点的坐标
```

`transpose(0, 1)` 操作本身几乎不消耗计算（只改变张量的 stride），但对内存访问模式有影响。

#### 三个交接点的语义含义

```
交接点 1：Transformer 层 11 -> EGNN 0（粗调）
  Transformer 输出：11 层后的 embedding 已经包含了局部二级结构信息
    - 位置 i 的 embedding 已经"知道"自己大概在 alpha 螺旋还是 beta 折叠中
    - 邻近位置（i-5 到 i+5）的信息已被充分整合
  EGNN 0 的任务：利用这些初步的结构信息，对坐标进行第一轮粗调
    - Scaffold 位置从随机坐标出发，被推向大致合理的三维位置
    - 修正量通常较大：坐标变化可能在 5-15 Angstrom 量级

交接点 2：Transformer 层 22 -> EGNN 1（中调）
  Transformer 输出：22 层后的 embedding 包含了全局上下文
    - 位置 i 已经"知道"了整条蛋白质的大致结构布局
    - 远程接触信息已被注意力机制传播
  EGNN 1 的任务：利用更精确的全局信息，进行第二轮坐标修正
    - 修正远程相互作用（比如两个 beta 折叠之间的距离）
    - 修正量中等：坐标变化通常在 1-5 Angstrom 量级

交接点 3：Transformer 层 33 -> EGNN 2（精修）
  Transformer 输出：完整 33 层后的 embedding 是最终的序列表示
    - 包含了所有层次的信息：局部、中程、远程、功能
  EGNN 2 的任务：进行最后的精细调整
    - 微调局部几何（键角、键长）
    - 确保物理合理性（Calpha 间距约 3.8 Angstrom）
    - 修正量较小：坐标变化通常在 0.5-2 Angstrom 量级
```

#### 坐标的原地更新

一个重要的细节是：坐标 `coors` 在整个前向传播过程中是**累积更新**的。不是每次 EGNN 从零开始预测坐标，而是在上一次的基础上做增量修正：

```
初始坐标：     coors_0 = [Motif 真实坐标 | Scaffold 随机坐标]
经过 EGNN 0：  coors_1 = coors_0 + delta_0    （粗调，delta_0 较大）
经过 EGNN 1：  coors_2 = coors_1 + delta_1    （中调，delta_1 中等）
经过 EGNN 2：  coors_3 = coors_2 + delta_2    （精修，delta_2 较小）
最终输出：     coors_final = coors_3

注意：Motif 位置的坐标也会被 EGNN 更新！
但在损失函数中，Motif 位置的结构损失权重为 0，
所以即使 EGNN 修改了 Motif 坐标，也不会影响训练梯度。
在推理时，可以选择保留 Motif 的原始坐标，也可以使用 EGNN 的预测坐标。
```


---

## 8. Alphabet：氨基酸词表设计

### 8.1 ESM-1b 词表结构

EnzyGen2 使用 ESM-1b 风格的 Alphabet（来自 `esm_modules.py`）：

```python
# 词表构成（ESM-1b 架构）：
prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")   # 索引 0-3
standard_toks = proteinseq_toks["toks"]                # 20种氨基酸 + B/U/Z/O/X/./-
# 中间还有 null token 用于对齐到 8 的倍数
append_toks = ("<mask>",)                               # 遮蔽 token

# 特殊 token 索引：
cls_idx (BOS) = 0    # <cls> 作为句首标记
padding_idx   = 1    # <pad> 用于批次对齐
eos_idx       = 2    # <eos> 句末标记
unk_idx       = 3    # <unk> 未知 token
mask_idx      = 32   # <mask> 遮蔽标记

# ESM-1b 配置：
prepend_bos = True    # 序列前加 <cls>
append_eos  = True    # 序列后加 <eos>
```

### 8.2 编码示例

```
原始序列：    M  K  V  A  L  G  A  A  G  G
编码后 tokens：[0, 20, 15, 7, 5, 10, 13, 5, 5, 13, 13, 2]
              ↑CLS                                     ↑EOS
总长度 = 序列长度 + 2（BOS + EOS）
```

### 8.3 词表大小的意义

```
ESM2 词表大小 = 33（all_toks 的长度）

这意味着 LM Head 输出 logits 的最后一维是 33：
  encoder_prob: [B, L, 33]

其中：
  索引 0-3：特殊 token（在推理时被设为 -inf，排除在预测之外）
  索引 4-23：20种标准氨基酸
  索引 24-32：非标准 token（B/U/Z/O/X 等，在推理时也被排除）
```


### 8.4 特殊 Token 在训练和推理中的不同用途

理解特殊 token 在不同阶段的行为，是避免常见 bug 的关键。

#### 完整的 ESM-1b Alphabet 映射表

```
索引    Token    含义                       类型
────    ─────    ───────                    ────
 0      <cls>    句首标记 / BOS             特殊
 1      <pad>    填充标记                   特殊
 2      <eos>    句末标记 / EOS             特殊
 3      <unk>    未知标记                   特殊
 4      L        亮氨酸 (Leucine)           标准氨基酸
 5      A        丙氨酸 (Alanine)           标准氨基酸
 6      G        甘氨酸 (Glycine)           标准氨基酸
 7      V        缬氨酸 (Valine)            标准氨基酸
 8      S        丝氨酸 (Serine)            标准氨基酸
 9      E        谷氨酸 (Glutamic acid)     标准氨基酸
10      R        精氨酸 (Arginine)          标准氨基酸
11      T        苏氨酸 (Threonine)         标准氨基酸
12      I        异亮氨酸 (Isoleucine)      标准氨基酸
13      D        天冬氨酸 (Aspartic acid)   标准氨基酸
14      P        脯氨酸 (Proline)           标准氨基酸
15      K        赖氨酸 (Lysine)            标准氨基酸
16      Q        谷氨酰胺 (Glutamine)       标准氨基酸
17      N        天冬酰胺 (Asparagine)      标准氨基酸
18      F        苯丙氨酸 (Phenylalanine)   标准氨基酸
19      Y        酪氨酸 (Tyrosine)          标准氨基酸
20      M        蛋氨酸 (Methionine)        标准氨基酸
21      H        组氨酸 (Histidine)         标准氨基酸
22      W        色氨酸 (Tryptophan)        标准氨基酸
23      C        半胱氨酸 (Cysteine)        标准氨基酸
24      X        任意氨基酸 (Unknown)       非标准
25      B        天冬氨酸或天冬酰胺 (D/N)   非标准
26      U        硒代半胱氨酸               非标准
27      Z        谷氨酸或谷氨酰胺 (E/Q)     非标准
28      O        吡咯赖氨酸                 非标准
29      .        间隙（alignment gap）      非标准
30      -        间隙（alignment gap）      非标准
31      <null_1> 空 token（对齐用）         对齐填充
32      <mask>   遮蔽标记                   特殊
```

**注意索引顺序**：20 种标准氨基酸占据索引 4-23，**不是** 0-19！这是因为前 4 个位置被特殊 token 占据。这是 ESM 系列模型的历史设计，初学者经常在这里犯错。

#### 各 Token 在训练和推理中的行为

```
Token        训练时的行为                              推理时的行为
─────        ───────────                              ───────────
<cls> (0)    自动添加到序列最前面                      自动添加到序列最前面
             参与注意力计算但不计入损失                参与注意力计算
             embedding 不受 Token Dropout 影响         保持正常

<pad> (1)    填充短序列到批次最大长度                  填充短序列到批次最大长度
             通过 padding_mask 排除在注意力之外        通过 padding_mask 排除
             不参与损失计算                            不参与预测

<eos> (2)    自动添加到序列最后面                      自动添加到序列最后面
             参与注意力但不计入损失                    参与注意力计算

<mask> (32)  替换 Scaffold 位置的真实氨基酸            替换需要设计的位置
             Token Dropout 将其 embedding 置零         推理时不置零（正常 embedding）
             模型需要预测这些位置的真实氨基酸          模型预测这些位置应是什么氨基酸

标准 AA      Motif 位置保留真实氨基酸                  Motif 位置保留真实氨基酸
(4-23)       不计入序列损失（已知答案）                LM Head 输出中排名最高的 AA 即为预测

非标准       在 EnzyGen2 的训练数据中极少出现          推理时 logits 中这些索引被设为 -inf
(24-30)      如出现则作为普通 token 参与计算           确保不会被采样到
```

#### 推理时的 Logits 过滤

```
推理时，LM Head 输出 logits [B, L, 33]
在采样之前，需要过滤掉不合法的 token：

logits[:, :, 0] = -inf    # 排除 <cls>
logits[:, :, 1] = -inf    # 排除 <pad>
logits[:, :, 2] = -inf    # 排除 <eos>
logits[:, :, 3] = -inf    # 排除 <unk>
logits[:, :, 24:] = -inf  # 排除所有非标准 token 和 <mask>

只保留索引 4-23（20 种标准氨基酸）的 logits
然后对这 20 个值做 softmax，得到合法的氨基酸概率分布
```

#### 编码/解码的具体数值示例

```
原始蛋白质序列：MKVALGAAGG（10 个残基）
Motif 位置：0, 1, 2, 3, 7, 8, 9（保留）
Scaffold 位置：4, 5, 6（需要设计）

编码过程：
  1. 加 BOS 和 EOS：  <cls> M K V A L G A A G G <eos>
  2. 查表得到索引：   [0,  20, 15, 7, 5, 4, 6, 5, 5, 6, 6, 2]
  3. 遮蔽 Scaffold：  [0,  20, 15, 7, 5, 32, 32, 32, 5, 6, 6, 2]
                        ^                     ^mask区域^
  序列长度 = 10 + 2 = 12

解码过程（推理后）：
  LM Head 输出 logits [1, 12, 33]
  对位置 5, 6, 7（对应原始位置 4, 5, 6）取 softmax：
    位置 5 的概率：A=0.05, C=0.01, ..., L=0.42, ..., V=0.12
    argmax = 4 -> L（亮氨酸）
    位置 6 的概率：A=0.03, ..., G=0.55, ...
    argmax = 6 -> G（甘氨酸）
    位置 7 的概率：A=0.61, ...
    argmax = 5 -> A（丙氨酸）
  恢复后序列：M K V A L G A A G G（与原始序列完全一致 -> 序列恢复率 100%）
```


---

## 9. Token Dropout 机制

### 9.1 什么是 Token Dropout

Token Dropout 是 ESM2 引入的一种正则化技术。在训练时，被 MASK 的位置的 embedding 被**置零**，然后用一个缩放因子补偿：

```python
# 来自 geometric_protein_model.py (line 230-236)
if self.encoder.token_dropout:
    x.masked_fill_((tokens == self.encoder.mask_idx).unsqueeze(-1), 0.0)
    # x: B x T x C
    mask_ratio_train = 0.15 * 0.8    # = 0.12 = 12%
    src_lengths = (~padding_mask).sum(-1)
    mask_ratio_observed = (tokens == self.encoder.mask_idx).sum(-1).to(x.dtype) / src_lengths
    x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]
```

### 9.2 数学原理

```
设：
  mask_ratio_train = 0.12（训练时期望的遮蔽比例）
  mask_ratio_observed = 实际遮蔽比例（可能是 0.8 或其他值）

缩放因子 = (1 - 0.12) / (1 - mask_ratio_observed)

目的：保持 embedding 的期望值不变
  未遮蔽 token 的贡献被放大，补偿被置零的遮蔽 token
```

### 9.3 为什么需要这个机制

在推理时（inference），所有 token 都是可见的（mask_ratio = 0），如果训练时没有做缩放补偿，模型在训练和推理之间会有信息量不匹配的问题。Token Dropout 的缩放确保训练和推理之间的 embedding 分布一致。


### 9.4 Token Dropout 的数值示例

为了彻底理解 Token Dropout，让我们用一个完整的数值示例走一遍整个过程。

#### 场景设定

```
蛋白质 1：长度 150 残基（加 BOS/EOS 后 L=152）
蛋白质 2：长度 150 残基（加 BOS/EOS 后 L=152）
批次大小 B=2

Motif 比例：20%（每条蛋白质有 30 个 Motif 残基，120 个 Scaffold 残基）
Scaffold 位置全部被 <mask> 替换
```

#### 步骤 1：统计遮蔽比例

```
蛋白质 1：
  总 token 数（不含 padding）：L_1 = 152
  被 <mask> 替换的数量：120（Scaffold 位置）
  mask_ratio_observed_1 = 120 / 152 = 0.7895

蛋白质 2：
  假设这条蛋白质有不同的 Motif 比例，30% 是 Motif
  被 <mask> 替换的数量：105
  mask_ratio_observed_2 = 105 / 152 = 0.6908
```

#### 步骤 2：Embedding 置零

```
x 的形状：[2, 152, 1280]

对于蛋白质 1 的第 35 个位置（假设是 Scaffold 位置，token = 32 即 <mask>）：
  x[0, 35, :] = [0.12, -0.85, 0.33, ..., 0.67]   <- embed_tokens(<mask>) 的输出

  Token Dropout 操作：
  x[0, 35, :] = [0.0, 0.0, 0.0, ..., 0.0]          <- 全部置零！

对于蛋白质 1 的第 2 个位置（假设是 Motif 位置，token = 20 即 M）：
  x[0, 2, :] = [0.45, 0.12, -0.67, ..., 0.89]    <- embed_tokens(M) 的输出
  不变，保留原始 embedding
```

#### 步骤 3：缩放补偿

```
mask_ratio_train = 0.15 * 0.8 = 0.12（ESM2 的默认训练遮蔽率）

对于蛋白质 1：
  缩放因子 = (1 - 0.12) / (1 - 0.7895) = 0.88 / 0.2105 = 4.18

  这意味着：蛋白质 1 中所有未遮蔽 token 的 embedding 被放大 4.18 倍！

  为什么这么大？因为 80% 的 token 被遮蔽（置零），只剩 20% 的 token
  贡献了信息。为了保持总信号量不变，这 20% 必须被大幅放大。

对于蛋白质 2：
  缩放因子 = (1 - 0.12) / (1 - 0.6908) = 0.88 / 0.3092 = 2.85

  遮蔽比例较低（69%），所以缩放因子也较小。
```

#### 步骤 4：理解缩放的物理意义

```
没有 Token Dropout 时：
  x 的每个位置的 embedding 的 L2 范数大约在 10-20 之间
  所有位置的 embedding 之和（用于下游任务）大约是 150 * 15 = 2250

有 Token Dropout 但没有缩放时：
  120 个位置被置零，只剩 30 个位置有非零 embedding
  embedding 之和变成 30 * 15 = 450（缩小了 5 倍！）
  这会导致后续层的激活值分布剧变，训练不稳定

有 Token Dropout 且有缩放时：
  30 个非零位置的 embedding 被放大 4.18 倍
  embedding 之和 = 30 * 15 * 4.18 = 1881（接近原始的 2250）
  偏差来自 mask_ratio_train (0.12) 与 mask_ratio_observed (0.79) 的不匹配

  缩放公式的本质：假设训练时只有 12% 的 token 被遮蔽（ESM2 原始训练），
  但现在 79% 被遮蔽，所以需要额外补偿。
```

#### 步骤 5：推理时的行为

```
推理时（inference），所有遮蔽位置也要通过 <mask> token 输入：
  mask_ratio_observed = 遮蔽比例（与训练时相同）

  但在某些实现中，推理时可能关闭 Token Dropout：
  x.masked_fill_((tokens == mask_idx).unsqueeze(-1), 0.0)  <- 仍然置零
  缩放因子仍然生效

  关键点：Token Dropout 不像普通 Dropout 那样在推理时关闭！
  它在训练和推理时的行为是一致的。
```

### 9.5 Token Dropout 与 BERT 的 Mask 策略对比

Token Dropout 和 BERT 的遮蔽策略看起来类似，但有本质区别。

#### BERT 的遮蔽策略（MLM）

```
BERT 选择 15% 的 token 进行处理，对这 15% 中的每个 token：
  80% 概率 -> 替换为 [MASK] token
  10% 概率 -> 替换为词表中随机的一个 token
  10% 概率 -> 保持不变

例如（自然语言）：
  原始：    "The cat sat on the mat"
  选中位置： "The [cat] sat on the [mat]"

  处理后可能是：
  "The [MASK] sat on the mat"     <- cat 被替换为 [MASK]（80%）
  "The cat sat on the [dog]"      <- mat 被替换为随机词 dog（10%）
  "The cat sat on the mat"        <- 保持不变（10%）

为什么 BERT 要这样做？
  1. 80% [MASK]：告诉模型"这个位置需要预测"
  2. 10% 随机替换：防止模型依赖"只有 [MASK] 位置才需要预测"的假设
  3. 10% 不变：让模型学会"即使看到了正确 token，也要做出有意义的表示"
```

#### ESM2/EnzyGen2 的 Token Dropout 策略

```
EnzyGen2 的做法更简单直接：
  1. 确定 Scaffold 位置（需要设计的残基）
  2. 这些位置的 token 被设为 <mask>（索引 32）
  3. <mask> 位置的 embedding 向量被**完全置零**（1280 维全 0）
  4. 所有 embedding 乘以缩放因子

对比 BERT：
  - 没有 10% 随机替换：不会引入噪声 token
  - 没有 10% 保持不变：遮蔽位置一律置零
  - embedding 被置零而不是保留 <mask> 的 embedding
```

#### 为什么 EnzyGen2 选择更简单的策略？

```
原因 1：信息泄露问题
  BERT 的 10% 随机替换在蛋白质上是有害的。蛋白质只有 20 种氨基酸，
  随机替换有 1/20 = 5% 的概率恰好替换为正确的氨基酸。
  更重要的是，某些位置的氨基酸高度保守（比如活性位点的催化残基），
  随机替换为同类氨基酸（比如 D->E，两者都是酸性）可能给模型提供
  不应有的线索。

原因 2：蛋白质 embedding 的连续性
  在自然语言中，词与词之间是离散的——"cat" 和 "dog" 的 embedding 没有
  物理上的相似性。但在蛋白质中，氨基酸之间有化学属性的连续性——
  V（缬氨酸）和 I（异亮氨酸）的 embedding 非常接近，因为它们的
  化学性质类似（都是非极性、脂肪族、大小相近）。

  置零是一种"绝对中立"的做法——不给模型任何关于该位置的信息，
  而不是像随机替换那样给出一个可能有偏的线索。

原因 3：与 EGNN 的配合
  EGNN 层需要对 Scaffold 位置的坐标进行预测。如果 Scaffold 位置的
  embedding 包含了来自随机氨基酸的信息，EGNN 可能会被误导，
  把坐标推向错误的方向。置零确保了 EGNN 只依赖邻居节点的信息。

原因 4：实现简单
  置零 + 缩放只需要 3 行代码，不需要维护复杂的遮蔽策略。
  在大规模训练中，代码简洁性 = 更少的 bug。
```


---

## 10. Rotary Position Embedding

### 10.1 为什么需要位置编码

Transformer 的自注意力机制本身对位置不敏感——交换两个 token 的位置不会改变注意力分数。但蛋白质序列中位置信息至关重要（第 5 个残基和第 100 个残基有不同的结构角色），所以需要位置编码。

### 10.2 RoPE 的优势

ESM2 使用 **Rotary Position Embedding（RoPE）**，相比传统的正弦位置编码：

```
传统方法：position embedding 加到 token embedding 上
  x = x + pos_emb
  缺点：位置信息可能在深层网络中被淡化

RoPE：在注意力计算时旋转 Q 和 K 向量
  Q_rotated = rotate(Q, position)
  K_rotated = rotate(K, position)
  attn = Q_rotated · K_rotated
  
  优点：
  1. 注意力分数自然编码相对位置距离
  2. 位置信息不会在深层网络中丢失
  3. 理论上可以泛化到更长的序列
```


### 10.3 RoPE 的数学原理

Rotary Position Embedding（旋转位置编码）的核心思想是：**不改变 embedding 本身，而是在计算注意力时旋转 Q 和 K 向量**。

#### 旋转矩阵的构造

对于位置 m 处的 token，考虑其 Q 向量中的一对维度 (d_{2i}, d_{2i+1})，RoPE 对这一对维度施加以下旋转：

```
旋转矩阵 R(m, i)：

  [ cos(m * theta_i)    -sin(m * theta_i) ]     [ q_{2i}   ]
  [                                        ]  *  [          ]
  [ sin(m * theta_i)     cos(m * theta_i) ]     [ q_{2i+1} ]

其中 theta_i = 10000^(-2i/d)
  d = head_dim = 64（每个注意力头的维度）
  i = 0, 1, 2, ..., 31（共 32 对维度，覆盖 64 维）
```

#### 为什么旋转能编码相对位置？

这是 RoPE 最优雅的数学性质。考虑位置 m 的 Q 和位置 n 的 K：

```
Q_m = R(m) * Q    （位置 m 处旋转后的 Q）
K_n = R(n) * K    （位置 n 处旋转后的 K）

注意力分数 = Q_m^T * K_n
            = (R(m) * Q)^T * (R(n) * K)
            = Q^T * R(m)^T * R(n) * K
            = Q^T * R(n - m) * K

关键推导：
  R(m)^T * R(n) = R(n - m)

  这是因为旋转矩阵的性质：R(a)^T = R(-a)，且 R(-a) * R(b) = R(b - a)。

  所以注意力分数只依赖于 (n - m)，即两个位置的**相对距离**！
```

这意味着：模型不需要知道"我在位置 42"和"你在位置 50"，它只需要知道"你在我右边 8 个位置"。这种相对位置编码对蛋白质特别自然——因为蛋白质的局部结构主要取决于残基间的相对距离，而不是绝对位置。

#### 频率 theta_i 的具体数值

对于 head_dim = 64（ESM2-650M 的配置），每个注意力头有 32 对维度：

```
i=0:   theta_0  = 10000^(-0/64)  = 10000^0     = 1.0
                  波长 = 2*pi/1.0 = 6.28 个位置

i=1:   theta_1  = 10000^(-2/64)  = 10000^(-0.03125)  = 0.737
                  波长 = 2*pi/0.737 = 8.52 个位置

i=2:   theta_2  = 10000^(-4/64)  = 10000^(-0.0625)   = 0.543
                  波长 = 2*pi/0.543 = 11.57 个位置

...

i=15:  theta_15 = 10000^(-30/64) = 10000^(-0.46875)  = 0.00294
                  波长 = 2*pi/0.00294 = 2137 个位置

i=31:  theta_31 = 10000^(-62/64) = 10000^(-0.96875)  = 0.000107
                  波长 = 2*pi/0.000107 = 58,544 个位置
```

这意味着：
- **低频维度（i 接近 0）**：波长短（约 6 个位置），对**局部**位置差异敏感。相邻残基和间隔 3 个残基的区别很明显。
- **高频维度（i 接近 31）**：波长长（约 58000 个位置），对**全局**位置差异敏感。序列中相距 100 和相距 200 的残基能被区分。

这种多频率设计就像傅里叶变换——同时在多个尺度上编码位置信息。

#### 旋转操作的计算效率

```
在实际实现中，旋转操作不用矩阵乘法实现（太慢），而是用向量操作：

# 假设 q 是 [L, 64] 的 Q 向量
q_even = q[:, 0::2]    # 偶数维度 [L, 32]
q_odd  = q[:, 1::2]    # 奇数维度 [L, 32]

cos_pos = cos(positions * thetas)    # [L, 32]
sin_pos = sin(positions * thetas)    # [L, 32]

q_rotated_even = q_even * cos_pos - q_odd * sin_pos
q_rotated_odd  = q_even * sin_pos + q_odd * cos_pos

q_rotated = interleave(q_rotated_even, q_rotated_odd)    # [L, 64]

计算量：4 次逐元素乘法 + 2 次加法 = 6L*32 = 192L 次浮点运算
相比注意力计算的 O(L^2 * 64)，RoPE 的开销可以忽略不计。
```

### 10.4 RoPE 在蛋白质序列中的具体意义

#### 相邻残基的注意力偏好

在蛋白质中，相邻残基（序列距离 = 1）之间有共价键连接（肽键），它们的空间距离固定为约 3.8 Angstrom。RoPE 对这种近距离关系的编码如何？

```
位置差 = 1 时，每个频率维度 i 的旋转角度为 theta_i：

i=0:   旋转角 = 1 * 1.0 = 1.0 弧度 = 57.3 度
       → 旋转角较大，Q 和 K 的相似度显著变化
       → 这个维度能敏感地区分"相邻"和"间隔 2 个残基"

i=15:  旋转角 = 1 * 0.00294 = 0.00294 弧度 = 0.17 度
       → 旋转角极小，几乎不影响注意力分数
       → 这个维度对局部位置差异不敏感，主要编码远程关系

综合效果：相邻残基的注意力特征非常相似（高频维度的小旋转），
但也有微妙的差异（低频维度的较大旋转），模型可以学会利用这些差异。
```

#### 远程残基的注意力行为

蛋白质中最有趣的是远程接触——序列上相距 50-100 个残基，但在 3D 空间中距离小于 8 Angstrom 的残基对。RoPE 如何帮助模型处理这种情况？

```
位置差 = 100 时：

i=0:   旋转角 = 100 * 1.0 = 100.0 弧度（取模 2*pi 后约 5.75 弧度）
       → 有效旋转角约 329 度，几乎转了一圈回来
       → 低频维度对这个距离的编码"混叠"了
       → 但模型可以通过其他维度来区分

i=5:   旋转角 = 100 * theta_5 = 100 * 0.178 = 17.8 弧度（取模 2*pi 后约 5.23 弧度）
       → 有效旋转角约 300 度
       → 中频维度仍然能提供有用的相对位置信号

i=15:  旋转角 = 100 * 0.00294 = 0.294 弧度 = 16.8 度
       → 旋转角适中，能清晰区分距离 100 和距离 200
       → 高频维度在这个尺度上提供最有用的位置信号
```

#### 最大可区分距离

```
RoPE 的最长波长对应 i=31（最后一对维度）：
  波长 = 2 * pi / theta_31 = 2 * pi / 0.000107 ≈ 58,544 个位置

这远超过已知的最长蛋白质序列（titin 蛋白约 34,350 个残基）。
因此，对于任何自然蛋白质，RoPE 都能在所有频率维度上唯一地编码
每个相对位置——不存在"两个不同距离被编码成相同向量"的混叠问题。

但对于低频维度（i=0），波长只有 6.28 个位置。这意味着
位置差 = 1 和位置差 = 7（= 1 + 2*pi/1.0）的编码在这个维度上非常接近。
这不是问题——因为有 32 个不同频率的维度同时编码，
模型综合所有维度的信息就能唯一确定任何相对位置。
```

#### RoPE vs 绝对位置编码

```
传统绝对位置编码（如原始 Transformer 中的正弦位置编码）：
  pos_embed(i) = [sin(i/10000^0), cos(i/10000^0), sin(i/10000^(2/d)), ...]
  x = x + pos_embed    ← 直接加到 embedding 上

问题 1：位置信息在深层网络中逐渐"稀释"
  经过多层 LayerNorm 和残差连接后，加上去的位置信号可能变得微弱
  模型在深层（比如第 30 层）可能"忘记"了位置信息

问题 2：长度泛化能力差
  如果训练时最长序列是 500，推理时遇到 600 的序列，
  位置 500-600 的位置编码是模型从未见过的

RoPE 如何解决这些问题：
  问题 1：RoPE 在每一层的注意力计算中都重新施加旋转，位置信息不会丢失
  问题 2：RoPE 只编码相对位置。即使序列变长，每一对残基的相对距离
         范围仍然在训练时见过的范围内（比如距离 1-500），所以能泛化
```


---

## 11. 为什么需要几何深度学习

### 11.1 蛋白质的三维本质

蛋白质不是简单的字符串——它是三维空间中的分子结构。两个在序列上很远的残基，可能在三维空间中非常接近（形成结构接触）：

```
序列距离 vs 空间距离：

序列：... A₁₀ ... A₅₀ ... A₁₅₀ ...
              ↑序列上很远

三维空间：
      A₁₀ ●
            ╲
             ╲  ← 只有 3.8Å
              ╲
      A₁₅₀ ●    ← 在空间中很近！
           
      A₅₀ ●     ← 在空间中可能很远
```

标准 Transformer 只能看到序列上的邻近关系，但 EGNN 可以看到三维空间中的邻近关系。

### 11.2 图神经网络在蛋白质中的应用

```
蛋白质 → 图表示：
  节点 = 残基（每个残基一个节点）
  边 = 空间邻近关系（3D 空间中 K 个最近邻）
  节点特征 = Transformer 的 hidden state [1280维]
  边特征 = 距离信息 [1维]
  坐标 = Cα 原子的 (x, y, z) [3维]
```

### 11.3 Transformer vs GNN 在蛋白质中的互补性

Transformer 和 GNN 各有擅长和不足，EnzyGen2 通过交错执行将二者的优势互补：

**Transformer 的优势与局限**

Transformer 的自注意力机制可以让序列中任意两个位置直接交互，时间复杂度为 O(L^2)。这使得它擅长捕捉**长程序列依赖**——例如，一个蛋白质家族中第 10 位和第 150 位的协同进化关系（coevolution），或者一个保守的催化残基三联体（catalytic triad）在序列上相距甚远却功能上紧密协作的模式。ESM2 在数亿条蛋白质序列上预训练，已经学到了这些丰富的进化模式。

然而，Transformer **完全忽略三维几何信息**。即使两个残基在三维空间中只隔 3.8A 形成关键的氢键，如果它们在序列上相距 100 个位置，Transformer 需要通过多层注意力才能间接学习到这种关系。更重要的是，标准 Transformer 没有坐标的概念——它不知道残基在空间中的位置，因此无法直接推理空间构象。

```
Transformer 看到的世界（一维序列）：

  A₁ ─ A₂ ─ A₃ ─ ... ─ A₅₀ ─ ... ─ A₁₅₀
  └───────────────────────────────────────┘
  任意两个位置都可以通过注意力直接交互
  但不知道 A₅₀ 和 A₁₅₀ 在空间中其实只隔 4Å
```

**GNN 的优势与局限**

图神经网络（特别是 EGNN）将蛋白质建模为三维空间中的图，节点是残基，边由 K-近邻关系决定。EGNN 天然擅长捕捉**三维空间关系**——它直接知道哪些残基在空间中相邻，可以推理空间局部的几何结构（如二级结构的氢键网络、疏水核心的残基堆积）。

然而，GNN 的信息传播受限于图的拓扑结构。在 K=30 的 KNN 图中，每个节点只能直接与 30 个最近邻交互。要让两个距离较远的残基交换信息，需要经过多跳（multi-hop）消息传递。如果两个残基在空间中相距 20A，可能需要 3-4 跳才能通信，而每跳都会损失信息（over-squashing 问题）。此外，GNN 通常只有 1-3 层（层数多了容易过平滑），进一步限制了信息传播范围。

```
GNN 看到的世界（三维局部图）：

      A₅₀ ●──● A₅₅
      ╱  ╲
  A₁₅₀ ●  ● A₄₈
      ╲
       ● A₁₄₇

  每个节点只看到 K 个最近邻（局部视野）
  空间远处的节点需要多跳才能通信
```

**EnzyGen2 的交错策略：两全其美**

EnzyGen2 的核心洞察是：让 Transformer 和 EGNN **交替执行**，每 11 层 Transformer 接一层 EGNN，交替 3 次。这样做实现了渐进式的序列-结构协同精化：

```
第 1 轮（Transformer 1-11 → EGNN 0）：
  Transformer 学习初步的序列上下文（哪些位置可能是什么氨基酸）
  → EGNN 利用这些序列信息 + 初始坐标，做第一次几何精化
  → 坐标更新后，重建 KNN 图

第 2 轮（Transformer 12-22 → EGNN 1）：
  Transformer 在更新后的特征上继续学习更精细的序列模式
  → EGNN 利用更好的序列特征 + 更好的坐标，进一步精化几何
  → 坐标再次更新

第 3 轮（Transformer 23-33 → EGNN 2）：
  Transformer 学习最终的序列决策
  → EGNN 做最后一次几何精化，产出最终坐标
```

每一轮中，Transformer 提供**全局序列上下文**（"这个位置在进化上倾向于哪些氨基酸"），EGNN 提供**局部几何约束**（"这个位置在三维空间中的邻居暗示它应该是疏水残基"）。两种信息通过交错不断融合、互相修正，最终达到序列和结构的一致性。

这种设计比"先跑完所有 Transformer 再跑 EGNN"效果更好，因为后者的信息流是单向的——EGNN 无法把几何信息反馈给 Transformer 的后续层。交错设计允许双向信息流，形成闭环。

---

## 12. E(n) 等变性：直觉解释

### 12.1 什么是等变性

**等变性（Equivariance）** 是指：对输入做某种变换，输出也做相同的变换。

```
对于三维坐标处理，我们需要两种性质：

1. 平移等变（Translation Equivariance）：
   如果把蛋白质向右平移 5Å，预测的坐标也应该向右平移 5Å

2. 旋转等变（Rotation Equivariance）：
   如果把蛋白质绕 x 轴旋转 90°，预测的坐标也应该旋转 90°

这两者合起来就是 E(n) 群的等变性，n 是空间维度（这里 n=3）。
```

### 12.2 为什么等变性重要

```
错误的非等变设计：

输入坐标：     旋转后坐标：
(1, 0, 0)     (0, 1, 0)
(2, 1, 0)     (-1, 2, 0)

如果模型不是等变的：
  预测(输入) = (3, 2, 1)
  预测(旋转) = (4, 0, 2)  ← 不等于旋转后的 (3, 2, 1) = (-2, 3, 1)！

等变模型保证：
  预测(旋转后输入) = 旋转(预测(输入))
  这是物理世界的基本对称性——蛋白质的行为不应该因为你选择的坐标系而改变。
```

### 12.3 EGNN 如何实现等变性

EGNN 的关键设计：

```
1. 节点特征 h（标量）是不变的：
   无论怎么旋转/平移蛋白质，h 不变
   → 用于序列预测

2. 坐标 x（向量）是等变的：
   旋转/平移蛋白质，x 做相同的变换
   → 用于结构预测

3. 只使用不变量（距离）来计算消息：
   edge_feature = f(||x_i - x_j||)  ← 距离是旋转不变的
   
4. 坐标更新使用坐标差：
   x_i += Σ (x_i - x_j) × φ(m_ij)   ← 坐标差是等变的
```

### 12.4 等变性的数学定义

我们来给出 E(n) 等变性的严格数学定义。

**定义**：设 f 是一个以坐标 x 和特征 h 为输入的函数，它同时输出新的坐标 f_x(x, h) 和新的特征 f_h(x, h)。若对任意旋转矩阵 R（满足 R^T R = I, det(R) = 1）和任意平移向量 t，都有：

```
f(Rx + t, h) = (R · f_x(x, h) + t,  f_h(x, h))
```

则称 f 是 **E(n)-等变的**。

上式包含两层含义：
1. **坐标输出是等变的**：对输入坐标施加旋转 R 和平移 t 后，输出坐标也相应地旋转和平移。这保证了物理世界中"换个角度看蛋白质，预测的结构也跟着转"。
2. **特征输出是不变的**：无论怎么旋转或平移输入坐标，输出的节点特征 h 不变。这是因为特征描述的是"残基是什么氨基酸"这样的内禀属性，不应该因为观察角度改变而改变。

**具体二维旋转示例**

为了直观理解，考虑一个 2D 简化例子。假设一个只有 2 个节点的"蛋白质"：

```
原始输入：
  节点 A：坐标 x_A = (1, 0)，特征 h_A = [0.5, 0.3]
  节点 B：坐标 x_B = (0, 1)，特征 h_B = [0.8, 0.1]
```

现在对坐标施加 90° 逆时针旋转（无平移）。90° 旋转矩阵为：

```
R = | cos90°  -sin90° |   =   |  0  -1 |
    | sin90°   cos90° |       |  1   0 |
```

旋转后的坐标：

```
R · x_A = |0  -1| · |1| = | 0|     →  x_A' = (0, 1)
          |1   0|   |0|   | 1|

R · x_B = |0  -1| · |0| = |-1|     →  x_B' = (-1, 0)
          |1   0|   |1|   | 0|
```

如果 EGNN 是等变的，会发生什么？

```
用原始坐标跑 EGNN：
  输入：x_A=(1,0), x_B=(0,1), h_A=[0.5,0.3], h_B=[0.8,0.1]
  输出：x_A_out=(1.2, 0.3), x_B_out=(-0.1, 1.4), h_A_out=[0.6,0.4], h_B_out=[0.9,0.2]

用旋转后坐标跑 EGNN：
  输入：x_A'=(0,1), x_B'=(-1,0), h_A=[0.5,0.3], h_B=[0.8,0.1]
  输出应该是：
    坐标 = R · [原始输出坐标]：
      R · (1.2, 0.3) = (-0.3, 1.2)   ✓ 坐标也旋转了 90°
      R · (-0.1, 1.4) = (-1.4, -0.1) ✓ 坐标也旋转了 90°
    特征 = [原始输出特征]（不变）：
      h_A_out = [0.6, 0.4]  ✓ 完全不变
      h_B_out = [0.9, 0.2]  ✓ 完全不变
```

这就是等变性的具体含义：**坐标随坐标系变换而变换，特征保持不变**。

### 12.5 EGNN 如何保证等变性——逐步证明

下面我们逐步证明 EGNN 的单层操作确实满足 E(n) 等变性。

**回顾 EGNN 单层的三步操作**：

```
Step 1（边模型）：m_ij = φ_e(h_i, h_j, ||x_i - x_j||²)
Step 2（坐标更新）：x_i' = x_i + Σ_{j∈N(i)} (x_i - x_j) · φ_x(m_ij)
Step 3（节点更新）：h_i' = h_i + φ_h(h_i, Σ_j m_ij)
```

**证明目标**：若将所有输入坐标替换为 Rx + t（其中 R 是旋转矩阵，t 是平移向量），则输出坐标变为 Rx' + t，输出特征不变。

**第 1 步：证明边模型输出 m_ij 是旋转/平移不变的**

边模型的输入包含三项：h_i（特征）、h_j（特征）和 ||x_i - x_j||^2（距离平方）。

对于距离平方，我们验证它在旋转平移下不变：

```
变换后的坐标：x_i → Rx_i + t,  x_j → Rx_j + t

距离平方：
||(Rx_i + t) - (Rx_j + t)||²
= ||R(x_i - x_j)||²            （平移 t 抵消）
= (R(x_i - x_j))^T · (R(x_i - x_j))
= (x_i - x_j)^T · R^T · R · (x_i - x_j)
= (x_i - x_j)^T · I · (x_i - x_j)     （因为 R^T R = I）
= ||x_i - x_j||²               ✓ 不变！
```

由于 h_i、h_j 是特征（不受坐标变换影响），||x_i - x_j||^2 也是不变的，所以：

```
m_ij = φ_e(h_i, h_j, ||x_i - x_j||²)  →  不变  ✓
```

**第 2 步：证明坐标更新是等变的**

坐标更新公式为：x_i' = x_i + Σ_j (x_i - x_j) · φ_x(m_ij)

将所有坐标替换为 Rx + t：

```
新的坐标差：
(Rx_i + t) - (Rx_j + t) = R(x_i - x_j)

新的标量权重：
φ_x(m_ij) 不变（因为 m_ij 不变，已在第 1 步证明）

新的坐标更新：
(Rx_i + t) + Σ_j R(x_i - x_j) · φ_x(m_ij)
= Rx_i + t + R · Σ_j (x_i - x_j) · φ_x(m_ij)
= R · [x_i + Σ_j (x_i - x_j) · φ_x(m_ij)] + t
= R · x_i' + t                                    ✓ 等变！
```

关键点：φ_x(m_ij) 输出的是**标量**，不是向量。标量乘以等变向量 (x_i - x_j) 仍然是等变的。如果 φ_x 输出的是向量，等变性就会被破坏——这就是 EGNN 设计 coord_mlp 最后一层输出维度为 1 的根本原因。

**第 3 步：证明节点特征更新是不变的**

节点更新公式为：h_i' = h_i + φ_h(h_i, Σ_j m_ij)

```
h_i：特征，不变
m_ij：已证明不变
Σ_j m_ij：不变量的求和，不变
φ_h(h_i, Σ_j m_ij)：MLP 作用于不变量，不变

所以 h_i' = h_i + φ_h(h_i, Σ_j m_ij) 也是不变的  ✓
```

**结论**：EGNN 的每一层都是 E(n)-等变的。由于等变映射的复合仍然是等变的，因此多层 EGNN 堆叠也是等变的。整个 EGNN 解码器保证了：无论用什么坐标系描述蛋白质，预测的坐标都会在同一个坐标系中，预测的序列概率不受坐标系影响。

---

## 13. EGNN 单层的三步操作：边模型、坐标模型、节点模型

### 13.1 EGNN 单层概览

```
EGNN 单层（E_GCL）的三步操作：

Step 1: 边模型（Edge Model）
  计算每对邻居节点之间的"消息" m_ij
  输入：节点特征 h_i, h_j + 距离 ||x_i - x_j||²
  输出：消息向量 m_ij

Step 2: 坐标模型（Coordinate Model）
  利用消息更新三维坐标
  输入：当前坐标 x_i, 坐标差 (x_i - x_j), 消息 m_ij
  输出：更新后的坐标 x_i'

Step 3: 节点模型（Node Model）
  利用聚合的消息更新节点特征
  输入：当前特征 h_i, 聚合的消息 Σ m_ij
  输出：更新后的特征 h_i'（带残差连接）
```

### 13.2 边模型详解

```python
# 来自 egnn.py (E_GCL, line 71-84)
def edge_model(self, source, target, radial, edge_attr, batch_size, k):
    # source: h_i, 来自发送节点 [B*L*K, 1280]
    # target: h_j, 来自接收节点 [B*L*K, 1280]
    # radial: ||x_i - x_j||², 距离平方 [B*L*K, 1]
    
    out = torch.cat([source, target, radial], dim=1)  # [B*L*K, 1280+1280+1]
    out = self.edge_mlp(out)  # MLP: [2561] → [1280] → [1280]
    # edge_mlp = Linear(2561→1280) → SiLU → Linear(1280→1280) → SiLU
    return out  # m_ij: [B*L*K, 1280]
```

**关键点**：边模型只依赖节点特征和**距离**（不依赖坐标方向），保证了旋转不变性。

### 13.3 坐标模型详解

```python
# 来自 egnn.py (E_GCL, line 102-117)
def coord_model(self, coord, edge_index, coord_diff, edge_feat, batch_size, k):
    # coord_diff: (x_i - x_j)，归一化后 [B*L*K, 3]
    # edge_feat: m_ij [B*L*K, 1280]
    
    trans = coord_diff * self.coord_mlp(edge_feat)  # [B*L*K, 3]
    # coord_mlp: Linear(1280→1280) → SiLU → Linear(1280→1, gain=0.001)
    # 输出标量 φ(m_ij) ∈ R，乘以坐标差得到平移向量
    
    # 聚合所有邻居的坐标更新
    agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
    
    coord = coord + agg  # 坐标更新
    return coord
```

**数学公式**：

$$x_i^{l+1} = x_i^l + \sum_{j \in N(i)} (x_i^l - x_j^l) \cdot \phi_x(m_{ij})$$

其中 $\phi_x$ 是坐标 MLP，输出标量权重。

**为什么乘以坐标差**：坐标差 $(x_i - x_j)$ 是等变的——旋转输入坐标，坐标差也会同样旋转。标量 $\phi_x(m_{ij})$ 是不变的。等变向量 × 不变标量 = 等变向量。

### 13.4 节点模型详解

```python
# 来自 egnn.py (E_GCL, line 86-100)
def node_model(self, x, edge_index, edge_attr, node_attr, batch_size, k):
    # 聚合来自邻居的消息
    agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
    # agg: [B*L, 1280]
    
    agg = torch.cat([x, agg], dim=1)  # [B*L, 2560]
    out = self.node_mlp(agg)          # MLP: [2560] → [1280] → [1280]
    
    if self.residual:
        out = x + out  # 残差连接
    return out, agg
```

### 13.5 距离计算

```python
# 来自 egnn.py (line 119-128)
def coord2radial(self, edge_index, coord):
    row, col = edge_index
    coord_diff = coord[row] - coord[col]           # (x_i - x_j) [B*L*K, 3]
    radial = torch.sum(coord_diff**2, 1).unsqueeze(1)  # ||x_i - x_j||² [B*L*K, 1]
    
    if self.normalize:
        norm = torch.sqrt(radial).detach() + 1e-8
        coord_diff = coord_diff / norm   # 归一化为单位方向向量
    
    return radial, coord_diff
```

### 13.6 完整数值示例：3 个节点、K=2 的 EGNN 单步

为了彻底理解 EGNN 单层的运算过程，我们用一个极简的数值示例，追踪每一步的具体数值变化。为了简化，特征维度使用 4 维（实际模型中是 1280 维，但原理完全一样）。

**初始设定**

```
3 个节点（对应 3 个残基），K=2（每个节点连接 2 个最近邻）

坐标：
  节点 0：x_0 = (0, 0, 0)
  节点 1：x_1 = (3, 0, 0)
  节点 2：x_2 = (0, 4, 0)

特征（4维，简化表示）：
  节点 0：h_0 = [1.0, 0.5, -0.3, 0.8]
  节点 1：h_1 = [0.2, 1.0,  0.6, -0.1]
  节点 2：h_2 = [-0.5, 0.3, 1.0, 0.4]
```

**第 1 步：构建 KNN 图（K=2）**

首先计算所有节点对之间的距离：

```
距离矩阵：
  d(0,1) = √((3-0)² + (0-0)² + (0-0)²) = √9 = 3.0 Å
  d(0,2) = √((0-0)² + (4-0)² + (0-0)²) = √16 = 4.0 Å
  d(1,2) = √((0-3)² + (4-0)² + (0-0)²) = √(9+16) = √25 = 5.0 Å

每个节点的 2 个最近邻（排除自身）：
  节点 0 的邻居：节点 1（d=3.0），节点 2（d=4.0）
  节点 1 的邻居：节点 0（d=3.0），节点 2（d=5.0）
  节点 2 的邻居：节点 0（d=4.0），节点 1（d=5.0）

边列表（有向，from → to）：
  edge_index = [[0,0, 1,1, 2,2],    ← row（源节点）
                [1,2, 0,2, 0,1]]    ← col（目标节点）
  共 6 条有向边（每个节点 2 条出边）
```

**第 2 步：计算距离和坐标差（coord2radial）**

对每条边计算坐标差和距离平方：

```
边 (0→1)：coord_diff = x_0 - x_1 = (0-3, 0-0, 0-0) = (-3, 0, 0)
           radial = (-3)² + 0² + 0² = 9.0
           归一化后 coord_diff = (-3, 0, 0) / (√9 + 1e-8) = (-1.0, 0, 0)

边 (0→2)：coord_diff = x_0 - x_2 = (0-0, 0-4, 0-0) = (0, -4, 0)
           radial = 0² + (-4)² + 0² = 16.0
           归一化后 coord_diff = (0, -4, 0) / (√16 + 1e-8) = (0, -1.0, 0)

边 (1→0)：coord_diff = x_1 - x_0 = (3, 0, 0)
           radial = 9.0
           归一化后 coord_diff = (1.0, 0, 0)

边 (1→2)：coord_diff = x_1 - x_2 = (3, -4, 0)
           radial = 9 + 16 = 25.0
           归一化后 coord_diff = (3, -4, 0) / (√25 + 1e-8) = (0.6, -0.8, 0)

边 (2→0)：coord_diff = x_2 - x_0 = (0, 4, 0)
           radial = 16.0
           归一化后 coord_diff = (0, 1.0, 0)

边 (2→1)：coord_diff = x_2 - x_1 = (-3, 4, 0)
           radial = 25.0
           归一化后 coord_diff = (-0.6, 0.8, 0)
```

**第 3 步：边模型（edge_model）**

对每条边，拼接源节点特征、目标节点特征和距离平方，送入 edge_mlp：

```
以边 (0→1) 为例：
  source = h_0 = [1.0, 0.5, -0.3, 0.8]     （4维）
  target = h_1 = [0.2, 1.0,  0.6, -0.1]     （4维）
  radial = [9.0]                              （1维）

  拼接：[1.0, 0.5, -0.3, 0.8, 0.2, 1.0, 0.6, -0.1, 9.0]  （9维，实际中为 2561维）

  送入 edge_mlp：
    Linear(9 → 4) → SiLU → Linear(4 → 4) → SiLU
    假设输出为 m_01 = [0.3, -0.2, 0.5, 0.1]

以边 (0→2) 为例：
  拼接：[1.0, 0.5, -0.3, 0.8, -0.5, 0.3, 1.0, 0.4, 16.0]
  假设输出为 m_02 = [0.1, 0.4, -0.1, 0.3]

类似地，计算所有 6 条边的消息 m_ij。
```

**第 4 步：坐标模型（coord_model）**

对每条边，将消息 m_ij 送入 coord_mlp 得到标量权重，乘以坐标差方向：

```
coord_mlp：Linear(4 → 4) → SiLU → Linear(4 → 1, gain=0.001)

以边 (0→1) 为例：
  coord_mlp(m_01) = coord_mlp([0.3, -0.2, 0.5, 0.1]) = 0.0015（标量，很小）
  方向向量（归一化后）= (-1.0, 0, 0)
  坐标平移贡献 = (-1.0, 0, 0) × 0.0015 = (-0.0015, 0, 0)

以边 (0→2) 为例：
  coord_mlp(m_02) = coord_mlp([0.1, 0.4, -0.1, 0.3]) = -0.0008
  方向向量 = (0, -1.0, 0)
  坐标平移贡献 = (0, -1.0, 0) × (-0.0008) = (0, 0.0008, 0)

节点 0 的坐标聚合更新（来自它的两个邻居）：
  Δx_0 = (-0.0015, 0, 0) + (0, 0.0008, 0) = (-0.0015, 0.0008, 0)

  x_0' = (0, 0, 0) + (-0.0015, 0.0008, 0) = (-0.0015, 0.0008, 0)
```

注意坐标变化非常微小（gain=0.001 的设计意图）——EGNN 是渐进式精化坐标，每一步只做微调，避免坐标剧烈跳动导致结构崩溃。

```
类似地计算节点 1 和节点 2 的坐标更新：

节点 1：
  来自边 (1→0)：coord_mlp(m_10) = 0.0012，方向 (1.0, 0, 0)
    → 贡献 (0.0012, 0, 0)
  来自边 (1→2)：coord_mlp(m_12) = -0.0005，方向 (0.6, -0.8, 0)
    → 贡献 (-0.0003, 0.0004, 0)
  Δx_1 = (0.0009, 0.0004, 0)
  x_1' = (3.0009, 0.0004, 0)

节点 2：
  来自边 (2→0)：coord_mlp(m_20) = 0.0010，方向 (0, 1.0, 0)
    → 贡献 (0, 0.0010, 0)
  来自边 (2→1)：coord_mlp(m_21) = -0.0003，方向 (-0.6, 0.8, 0)
    → 贡献 (0.00018, -0.00024, 0)
  Δx_2 = (0.00018, 0.00076, 0)
  x_2' = (0.00018, 4.00076, 0)
```

**第 5 步：节点模型（RM-Node 变体的门控更新）**

在 RM-Node 变体中，不使用独立的节点 MLP，而是用门控机制。首先对每个节点聚合来自所有邻居的消息：

```
节点 0 的聚合消息：agg_0 = m_01 + m_02 = [0.3+0.1, -0.2+0.4, 0.5-0.1, 0.1+0.3]
                         = [0.4, 0.2, 0.4, 0.4]

门控值计算（node_gate）：
  node_gate：Linear(4→4) → ReLU → Linear(4→4) → Sigmoid
  gate_0 = sigmoid(node_gate_linear(relu(node_gate_linear(agg_0))))
  假设 gate_0 = [0.7, 0.3, 0.8, 0.5]   （每个维度在 [0,1] 之间）

门控残差更新：
  h_0' = h_0 + gate_0 * agg_0
       = [1.0, 0.5, -0.3, 0.8] + [0.7×0.4, 0.3×0.2, 0.8×0.4, 0.5×0.4]
       = [1.0, 0.5, -0.3, 0.8] + [0.28, 0.06, 0.32, 0.20]
       = [1.28, 0.56, 0.02, 1.00]
```

门控值的含义：
- gate ≈ 1.0 的维度：完全接受邻居信息（该维度的信息需要从邻居获取）
- gate ≈ 0.0 的维度：忽略邻居信息（该维度的信息已经足够，不需要更新）
- gate ≈ 0.5 的维度：折中，部分接受邻居信息

```
类似地计算节点 1 和节点 2 的特征更新。

最终 EGNN 单步输出：
  h_0' = [1.28, 0.56, 0.02, 1.00],  x_0' = (-0.0015, 0.0008, 0)
  h_1' = [更新后的特征],              x_1' = (3.0009, 0.0004, 0)
  h_2' = [更新后的特征],              x_2' = (0.00018, 4.00076, 0)

可以观察到：
  1. 坐标变化极小（O(0.001)），符合渐进精化的设计
  2. 特征变化较显著（O(0.1-0.3)），因为特征更新没有 gain 限制
  3. 三角形构型基本保持（3节点的相对位置关系几乎不变）
```

---

## 14. K-近邻图构建

### 14.1 为什么用 KNN 图

蛋白质中每个残基不需要与所有其他残基交互（那样计算量是 O(L²)）。只与空间上最近的 K 个邻居交互就足够了：

```
蛋白质（200个残基）：
  全连接图：200 × 200 = 40,000 条边  ← 太多
  KNN图（K=30）：200 × 30 = 6,000 条边  ← 合理
```

### 14.2 KNN 图构建代码

```python
# 来自 geometric_protein_model.py (line 137-151)
def get_edges_batch(n_nodes, batch_size, coords, k=30):
    rows, cols = [], []
    # 处理无效坐标
    coords = torch.where(torch.isinf(coords), torch.full_like(coords, 0), coords)
    coords = torch.where(torch.isnan(coords), torch.full_like(coords, 0), coords)
    
    for i in range(batch_size):
        # 用 sklearn 的 Ball Tree 算法查找 K 近邻
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords[i])
        distances, indices = nbrs.kneighbors(coords[i])  # [N, K+1]
        
        edges = get_edges(n_nodes, k, indices)  # 构建边列表
        # 偏移：让每个样本的节点 ID 不重叠
        rows.append(edges[0] + n_nodes * i)
        cols.append(edges[1] + n_nodes * i)
    
    edges = [torch.cat(rows).to(device), torch.cat(cols).to(device)]
    return edges  # [2, B*L*K]
```

### 14.3 动态重建

**关键设计**：KNN 图在每次 EGNN 层执行前**重新构建**。因为 EGNN 层会更新坐标，更新后的坐标会改变哪些残基是"最近邻"。

```python
# 来自 geometric_protein_model.py (line 280-286)
if (layer_idx + 1) % 11 == 0:
    # 每 11 层 Transformer 后：
    coords = coords.view(batch_size, -1, 3)
    edges = get_edges_batch(n_nodes, batch_size, coords.detach().cpu(), k)  # 重建 KNN
    coords = coords.reshape(-1, 3)
    x, coords, _ = self.decoder._modules["gcl_%d" % decoder_layer_idx](x, edges, coords, ...)
```

### 14.4 KNN 图的具体数值示例

下面用 5 个节点、K=2 的具体例子，完整演示 KNN 图构建过程。

**设定**

```
5 个残基的 Cα 坐标：
  节点 0：(0, 0, 0)
  节点 1：(3, 0, 0)
  节点 2：(0, 4, 0)
  节点 3：(3, 4, 0)
  节点 4：(1.5, 2, 5)

K = 2（每个节点连接 2 个最近邻）
```

**第 1 步：计算距离矩阵**

```
距离矩阵 D[i][j]（对称矩阵，对角线为 0）：

       节点0   节点1   节点2   节点3   节点4
节点0  0.000   3.000   4.000   5.000   5.590
节点1  3.000   0.000   5.000   4.000   5.590
节点2  4.000   5.000   0.000   3.000   5.590
节点3  5.000   4.000   3.000   0.000   5.590
节点4  5.590   5.590   5.590   5.590   0.000

计算过程（部分）：
  D[0][1] = sqrt(3^2 + 0^2 + 0^2) = 3.0
  D[0][3] = sqrt(3^2 + 4^2 + 0^2) = 5.0
  D[0][4] = sqrt(1.5^2 + 2^2 + 5^2) = sqrt(2.25 + 4 + 25) = sqrt(31.25) ≈ 5.590
  D[2][3] = sqrt(3^2 + 0^2 + 0^2) = 3.0
```

**第 2 步：为每个节点选择 K=2 个最近邻**

```
NearestNeighbors(n_neighbors=K+1=3) 返回包含自身的 3 个最近邻。
去除自身后保留 2 个：

节点 0 的距离排序：自身(0) < 节点1(3.0) < 节点2(4.0) < 节点3(5.0) < 节点4(5.59)
  → 最近邻：{1, 2}

节点 1 的距离排序：自身(0) < 节点0(3.0) < 节点3(4.0) < 节点2(5.0) < 节点4(5.59)
  → 最近邻：{0, 3}

节点 2 的距离排序：自身(0) < 节点3(3.0) < 节点0(4.0) < 节点1(5.0) < 节点4(5.59)
  → 最近邻：{3, 0}

节点 3 的距离排序：自身(0) < 节点2(3.0) < 节点1(4.0) < 节点0(5.0) < 节点4(5.59)
  → 最近邻：{2, 1}

节点 4 的距离排序：自身(0) < 其余四个距离都约 5.59
  → 最近邻：{0, 1}（距离几乎相同，取前两个）
```

**第 3 步：构建边列表**

```
有向边（每个节点发出 K=2 条边）：

  row (源)：[0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
  col (目标)：[1, 2, 0, 3, 3, 0, 2, 1, 0, 1]

  共 10 条有向边（5 个节点 × K=2）

注意：
  边 (0→1) 和 边 (1→0) 都存在 ← 这两条是独立的有向边
  边 (0→3) 不存在 ← 因为节点 3 不是节点 0 的 2 个最近邻
  节点 4 只有出边 (4→0) 和 (4→1)，但节点 0 的邻居中没有节点 4
    → 这意味着 KNN 图不是对称的！
```

**自环处理**

```
代码中使用 n_neighbors=K+1，然后排除自身：
  NearestNeighbors 返回 [自身, 第1近邻, 第2近邻]
  get_edges() 函数跳过索引 0（自身），只取索引 1 和 2

因此，KNN 图中没有自环（self-loop）。
这是合理的——一个残基不需要给自己发消息，残差连接已经保留了自身信息。
```

### 14.5 为什么每次 EGNN 后重建 KNN 图

这是 EnzyGen2 的一个关键设计决策：在每次 EGNN 层执行前，用**当前坐标**（而非初始坐标）重新构建 KNN 图。

**坐标更新改变邻域关系**

EGNN 的核心功能是更新三维坐标。每次坐标更新后，残基之间的距离关系会发生变化，原来的近邻可能不再是近邻，原来较远的残基可能变成近邻。

```
示例：考虑一个 100 残基的蛋白质，初始状态下：

初始坐标下的邻域（K=30）：
  残基 5 的第 30 近邻是残基 50（距离 12.0Å）

经过 EGNN 层 0 的坐标更新后：
  残基 5 和残基 50 之间距离变为 8.5Å
  残基 50 进入了残基 5 的前 5 近邻！

如果不重建 KNN 图：
  残基 5 的邻域仍然基于旧坐标，可能不包含残基 50
  EGNN 层 1 无法利用这个重要的新邻近关系
```

**渐进式邻域发现**

三次 EGNN 层配合三次 KNN 重建，实现了"渐进式邻域发现"：

```
EGNN 层 0 + KNN 重建 0：
  基于初始坐标（motif 坐标 + 随机游走初始化）构建图
  EGNN 做第一次粗调：将随机初始化的坐标向大致合理的位置移动
  → 一些远程接触开始形成

KNN 重建 1（基于更新后的坐标）：
  新的图反映了第一次调整后的空间关系
  更多潜在的接触被发现

EGNN 层 1 + KNN 重建 1：
  在更准确的图上做精调
  → 更多精细接触形成

KNN 重建 2 + EGNN 层 2：
  最终精化，图已经接近真实结构的接触图
```

**如果不重建会怎样**

如果整个过程只用初始坐标构建一次 KNN 图：

1. 被遮蔽残基的初始坐标来自随机游走，其邻域关系基本是随机的
2. 后续 EGNN 层虽然更新了坐标，但仍然在旧的（错误的）图上传递消息
3. 模型相当于在"错误的邻居"之间交换信息，效果大打折扣
4. 这就好比导航软件用了一张过时的地图——路已经修好了，但你还在绕远路

**计算代价**

重建 KNN 图的代价是 O(L * K * log L)（Ball Tree 算法），远小于 EGNN 层本身的计算代价 O(L * K * d)。代码中用了 `coords.detach().cpu()` 将坐标从 GPU 复制到 CPU 上用 sklearn 的 Ball Tree 实现。虽然引入了 CPU-GPU 数据传输开销，但保证了邻域关系的准确性，是值得的权衡。

---

## 15. 四种 EGNN 变体详解

EnzyGen2 实现了四种 EGNN 变体，默认使用 `rm-node`：

### 15.1 Full（E_GCL）

完整版本，包含所有三个子模型：

```
边模型：MLP([h_i, h_j, ||x_i-x_j||²] → m_ij)
坐标模型：x_i += Σ (x_i-x_j) × coord_mlp(m_ij)
节点模型：h_i = h_i + MLP([h_i, Σ m_ij])  ← 有独立的节点 MLP

参数量最大，表达能力最强
```

### 15.2 RM-Node（E_GCL_RM_Node）— **EnzyGen2 默认**

移除节点 MLP，用**门控机制**替代：

```python
# 来自 egnn.py (line 215-223)
def node_model(self, x, edge_index, edge_attr, node_attr, batch_size, k):
    edge_attr = edge_attr.view(batch_size, -1, k, dim)
    agg = torch.sum(edge_attr, dim=2).view(-1, dim)  # 直接求和聚合
    out = x
    if self.residual:
        out = x + self.node_gate(agg) * agg  # 门控残差
    return out, agg

# node_gate 结构：
self.node_gate = nn.Sequential(
    nn.Linear(1280, 1280),
    nn.ReLU(),
    nn.Linear(1280, 1280),
    nn.Sigmoid()  # 输出 [0,1] 的门控值
)
```

**优势**：门控机制让模型自适应地决定从邻居聚合多少信息，同时减少参数量。

### 15.3 RM-Edge（E_GCL_RM_Edge）

移除边 MLP，用**注意力机制**替代：

```python
# 来自 egnn.py (line 294-300)
def edge_model(self, source, target, radial, edge_attr, batch_size, k):
    # 用点积注意力 + 距离加权
    attn = torch.sum(source * target, dim=-1)  # 点积相似度
    dist_score = 1.0 / (radial + 1e-8)        # 距离的倒数
    attn_score = softmax(attn + dist_score, dim=-1)
    out = target * attn_score  # 注意力加权的邻居特征
    return out, attn_score
```

### 15.4 RM-All（E_GCL_RM_All）

移除边 MLP 和节点 MLP，最简化版本：

```
边模型：纯注意力（点积 + 距离）
坐标模型：x_i += Σ (x_i-x_j) × attn_score
节点模型：h_i = h_i + Σ (h_j × attn_score)  ← 直接加权求和

参数量最小，计算最快
```

### 15.5 四种变体对比

| 变体 | 边模型 | 节点模型 | 坐标更新 | 额外参数 |
|------|--------|---------|---------|---------|
| full | MLP | MLP+残差 | coord_mlp | edge_mlp + node_mlp + coord_mlp |
| rm-node | MLP+注意力 | 门控残差 | coord_mlp | edge_mlp + node_gate + coord_mlp |
| rm-edge | 点积注意力 | MLP+残差 | 注意力加权 | node_mlp |
| rm-all | 点积注意力 | 直接求和 | 注意力加权 | 无额外MLP |

### 15.6 为什么 EnzyGen2 选择 RM-Node 变体

在四种 EGNN 变体中，EnzyGen2 默认选择 RM-Node（E_GCL_RM_Node）。这个选择背后有深思熟虑的工程权衡。

**Full 变体的问题：过拟合风险**

Full 变体拥有最多的参数——它同时包含 edge_mlp、coord_mlp 和 node_mlp 三个独立的 MLP 网络。每个 MLP 都有两层 Linear + 激活函数，总参数量约为：

```
edge_mlp：Linear(2561→1280) + Linear(1280→1280) ≈ 3.3M + 1.6M = 4.9M
coord_mlp：Linear(1280→1280) + Linear(1280→1) ≈ 1.6M + 1.3K = 1.6M
node_mlp：Linear(2560→1280) + Linear(1280→1280) ≈ 3.3M + 1.6M = 4.9M
每层总计 ≈ 11.4M，3 层 ≈ 34.2M 参数

对于酶设计数据集（通常只有几万到几十万条样本），这么多参数容易过拟合。
模型可能记住训练集中的特定结构模式，而无法泛化到新的酶序列。
```

**RM-All 变体的问题：表达能力不足**

RM-All 是最极简的变体，完全去掉了 edge_mlp 和 node_mlp，只用点积注意力和距离加权。它的参数量几乎为零（除了坐标更新用的 attention score），但表达能力太弱：

```
问题 1：边模型只用点积相似度 + 距离倒数，无法学习复杂的距离依赖交互
  例如：两个残基距离 5Å 时应该有排斥力（空间碰撞），
       距离 8Å 时应该有吸引力（范德华力），
       这种非单调的距离-力关系，纯注意力无法表达

问题 2：节点模型直接求和，没有选择性地过滤邻居信息
  所有邻居的贡献被无差别地加到节点上，噪声和信号一起混入
```

**RM-Node 变体的平衡：保留关键能力，简化冗余部分**

RM-Node 做了一个精妙的权衡——它保留了 edge_mlp（边模型），但用门控机制替代了 node_mlp（节点模型）：

```
保留 edge_mlp 的原因：
  边模型负责学习"两个残基之间的相互作用应该是什么"
  输入包含距离信息（radial），MLP 可以学习非线性的距离依赖关系：
    - 近距离（<4Å）：强排斥
    - 中距离（4-8Å）：弱吸引（范德华力、氢键）
    - 远距离（>10Å）：几乎无作用
  这种复杂的距离响应模式需要 MLP 的非线性拟合能力

用门控替代 node_mlp 的原因：
  节点模型本质上做的是"将聚合的邻居信息整合到自身特征中"
  这个操作不需要很复杂——简单的门控就够了：
    gate = sigmoid(MLP(agg))   ← 输出 [0,1] 的门控向量
    h_new = h + gate * agg     ← 选择性地接受邻居信息

  门控机制的参数量：
    Linear(1280→1280) + Linear(1280→1280) ≈ 3.3M
  远少于完整 node_mlp 的 4.9M
```

**门控作为"信息阀门"的直觉解释**

门控值（sigmoid 输出，范围 [0,1]）可以理解为一个"信息阀门"：

```
gate ≈ 0（阀门关闭）：
  忽略来自邻居的信息
  适用于：当前节点的特征已经很好了（例如 motif 残基，已有确定的氨基酸类型）
  不需要邻居来修正

gate ≈ 1（阀门全开）：
  完全接受邻居的信息
  适用于：当前节点的特征不确定（例如被遮蔽的 scaffold 残基）
  需要从邻居获取线索

gate ≈ 0.5（阀门半开）：
  部分接受邻居信息
  适用于：当前特征部分确定，需要微调

关键：门控是逐维度的（每个 1280 维特征都有独立的门控值）
  → 模型可以在某些维度上接受邻居信息，在其他维度上保持自身信息
  → 比全有全无的更新（全接受或全拒绝）灵活得多
```

**实验验证**

在 EnzyGen2 的论文中，RM-Node 在序列恢复率（Sequence Recovery Rate）和结构预测精度（RMSD）上都取得了最佳或接近最佳的性能，同时训练速度比 Full 变体快约 15%。这验证了"保留边 MLP + 简化节点模型"是正确的设计选择。

---

## 16. SubstrateEGNN：配体原子处理

### 16.1 配体 vs 蛋白质的处理差异

```
蛋白质：
  节点 = 残基（Cα 原子）
  特征来源 = ESM2 Transformer 的隐藏状态 [1280维]
  已经有丰富的上下文信息

配体：
  节点 = 每个原子
  特征来源 = 原子级别的 5 维描述符
  需要从底层学起
```

### 16.2 SubstrateEGNN 结构

```python
# 来自 egnn.py (line 507-550)
class SubstrateEGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, ...):
        self.embedding_in = nn.Linear(5, hidden_nf)  # 5维 → 1280维
        # 3 层 E_GCL_RM_Node
        for i in range(n_layers):  # n_layers=3
            self.add_module("gcl_%d" % i, E_GCL_RM_Node(...))
    
    def forward(self, h, x, edges, edge_attr=None, k=30):
        h = self.embedding_in(h.float())      # [B*N, 5] → [B*N, 1280]
        h = h.reshape(-1, h.size()[-1])       # [B*N, 1280]
        x = x.reshape(-1, x.size()[-1])       # [B*N, 3]
        for i in range(self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, x, ...)
        return h, x
```

### 16.3 配体处理流程

```
配体原子特征 [B, N_atoms, 5]
        │
        ▼  Linear(5 → 1280)
配体节点特征 [B*N_atoms, 1280]
        │
        ▼  构建 KNN 图（配体原子间）
        │
        ▼  E_GCL_RM_Node × 3 层
        │
配体更新特征 [B*N_atoms, 1280]
        │
        ▼  reshape + sum pooling
        │
配体表示 [B, 1280]  ← 整个配体的固定维度表示
```

### 16.4 配体 5 维原子描述符的含义

SubstrateEGNN 的输入是每个配体原子的 5 维特征向量。这 5 个维度编码了原子的基本化学性质：

```
5 维原子描述符：
  维度 0：原子序数（atomic number）
    → 归一化后的值，区分不同元素
    → C=6, N=7, O=8, S=16, P=15, ...

  维度 1：成键数目（number of bonds / degree）
    → 该原子连接了多少个其他原子
    → sp3 碳=4, sp2 碳=3, 羰基氧=1, ...

  维度 2：形式电荷（formal charge）
    → 原子上的净电荷
    → 中性=0, 质子化胺=+1, 去质子化羧基=-1

  维度 3：是否芳香（is_aromatic）
    → 二值特征：0=非芳香, 1=芳香
    → 苯环上的碳=1, 脂肪族碳=0

  维度 4：氢原子数目（number of hydrogens）
    → 连接的氢原子个数
    → 甲基碳=3, 亚甲基碳=2, 叔碳=1, 季碳=0
```

**为什么选择这 5 个特征**

这 5 个特征的组合可以区分绝大多数药物和底物中的原子类型。例如：

```
以华法林（Warfarin，一种常见药物分子）为例：

芳香环上的碳原子：
  [6, 3, 0, 1, 0]
  原子序数=6(碳), 成键数=3(sp2), 电荷=0, 芳香=1, 氢数=0

羰基氧原子：
  [8, 1, 0, 0, 0]
  原子序数=8(氧), 成键数=1(双键连碳), 电荷=0, 非芳香, 氢数=0

羟基氧原子：
  [8, 2, 0, 0, 1]
  原子序数=8(氧), 成键数=2(连碳+连氢), 电荷=0, 非芳香, 氢数=1

脂肪族 CH₃ 碳原子：
  [6, 4, 0, 0, 3]
  原子序数=6(碳), 成键数=4(sp3), 电荷=0, 非芳香, 氢数=3
```

这些化学特征捕捉了原子参与非共价相互作用的能力：
- 原子序数决定了原子半径和电负性
- 成键数反映了杂化状态和空间位阻
- 形式电荷影响静电相互作用
- 芳香性决定了 π-π 堆积能力
- 氢原子数影响氢键供体能力

### 16.5 为什么配体和蛋白质用不同的 EGNN

EnzyGen2 使用两套独立的 EGNN 来分别处理蛋白质和配体，而不是将它们放在同一个图中。这个设计有三个重要原因。

**原因 1：特征维度不同**

```
蛋白质节点特征：1280 维
  来源：ESM2 Transformer 的隐藏状态
  经过 33 层 Transformer 处理，包含丰富的序列上下文信息
  已经"知道"每个残基在进化上的偏好

配体节点特征：5 维 → 线性投影到 1280 维
  来源：原子级别的化学描述符
  需要从零开始学习原子间的相互作用模式

两者的"起点"完全不同：蛋白质特征已经是高层语义表示，
配体特征还是底层的物理化学属性。如果强行放在同一个图中，
共享的 MLP 参数需要同时适应两种完全不同的输入分布，学习效率低。
```

**原因 2：图密度不同**

```
蛋白质图（150 个残基，K=30）：
  每个节点连接 30 个邻居
  30/150 = 20% 的节点是邻居
  图是稀疏的——每个残基只看到附近的局部结构

配体图（20 个原子，K=min(30, 19)=19）：
  每个节点连接 19 个邻居（几乎全连接！）
  19/20 = 95% 的节点是邻居
  图是密集的——每个原子几乎能看到整个分子

这两种图有完全不同的信息传播特性：
  蛋白质图需要多跳传播才能覆盖全局 → 需要更深的网络
  配体图一跳就能传遍全分子 → 浅层网络就够了
```

**原因 3：物理性质不同**

```
蛋白质残基之间的相互作用：
  - 残基间距 3.8Å（序列相邻）到 >30Å（远端折叠接触）
  - 相互作用类型：氢键、疏水作用、盐桥、范德华力
  - 约束：主链连接的序列拓扑、二级结构（α-helix、β-sheet）
  - 关键：残基间的相对方向很重要（如 β-sheet 的平行/反平行排列）

配体原子之间的相互作用：
  - 原子间距 1.2-1.6Å（共价键）到 3-4Å（跨环非键接触）
  - 相互作用类型：共价键、共轭效应、空间排斥
  - 约束：键角、键长、二面角、手性中心
  - 关键：小分子构象通常较为刚性，自由度有限

两种系统的物理本质不同，使用独立的 EGNN 可以让每套参数
专门适应各自的物理规律，而不必做妥协。
```

**最终融合**

虽然蛋白质和配体用不同的 EGNN 处理，但它们最终通过 sum-pooling 和拼接融合在一起：

```
蛋白质表示 [B, 1280] ← 蛋白质 EGNN 输出的 sum-pooling
配体表示   [B, 1280] ← 配体 EGNN 输出的 sum-pooling
拼接       [B, 2560] ← 送入结合预测头
```

这种"分别处理、最后融合"的设计，既尊重了两个子系统各自的特性，又实现了跨系统的信息交互。

---

## 17. GeometricProteinNCBIModel：基础模型

### 17.1 模型注册

```python
# 来自 geometric_protein_model.py (line 154)
@register_model("geometric_protein_model_ncbi")
class GeometricProteinNCBIModel(TransformerModel):
    # 在 fairseq 框架中注册为 "geometric_protein_model_ncbi"
    # 继承自 TransformerModel，但完全重写了 forward 方法
```

### 17.2 构造函数

```python
# line 181-187
def __init__(self, args, encoder, decoder):
    super().__init__(args, encoder, decoder)
    self.encoder_layers = args.encoder_layers  # 33（ESM2层数）或 6（小型配置）
    self.mask_index = self.encoder.alphabet.mask_idx   # 32
    self.k = args.knn                                  # 30
    self.ncbi_embeddings = nn.Embedding(10000, args.encoder_embed_dim)  # [10000, 1280]
    nn.init.normal_(self.ncbi_embeddings.weight, mean=0, std=args.encoder_embed_dim ** -0.5)
```

### 17.3 编码器和解码器的构建

```python
# line 200-208
@classmethod
def build_encoder(cls, args, src_dict, embed_tokens):
    model, alphabet = load_from_pretrained_models(args.pretrained_esm_model)
    return model  # ESM2 实例

@classmethod
def build_decoder(cls, args, tgt_dict, embed_tokens):
    decoder = EGNN(
        in_node_nf=args.encoder_embed_dim,   # 1280
        hidden_nf=args.encoder_embed_dim,     # 1280
        out_node_nf=3,                        # 输出3维坐标
        in_edge_nf=0,                         # 无边特征输入
        n_layers=args.decoder_layers,         # 3
        attention=True,
        mode=args.egnn_mode                   # "rm-node"
    )
    return decoder
```

### 17.4 模型初始化的细节

在 `__init__` 方法中，模型的各个组件按以下顺序构建和初始化。理解这个过程有助于调试和修改模型。

**编码器初始化（ESM2 预训练权重）**

```
build_encoder() 调用 load_from_pretrained_models()：
  1. 从 esm 库加载预训练的 ESM2 模型
  2. ESM2 配置（以 esm2_t33_650M_UR50D 为例）：
     - encoder_layers = 33        （Transformer 层数）
     - encoder_embed_dim = 1280   （隐藏维度）
     - encoder_attention_heads = 20（注意力头数）
     - 每个头的维度 = 1280 / 20 = 64
     - FFN 中间维度 = 1280 × 4 = 5120
  3. 总参数量 ≈ 650M（其中嵌入层 ~42K, Transformer层 ~648M, LM头 ~42K）
  4. 所有权重从预训练 checkpoint 加载（不是随机初始化）

编码器的参数在训练时可以被微调（fine-tuned），也可以冻结（frozen）。
EnzyGen2 默认是全参数微调——这让 ESM2 的表示适应酶设计任务。
```

**解码器初始化（EGNN 随机初始化）**

```
build_decoder() 构建 EGNN：
  1. 创建 EGNN 实例，包含 3 层 E_GCL_RM_Node
  2. 每层 E_GCL_RM_Node 的组件：
     - edge_mlp：
         Linear(2561, 1280) → SiLU → Linear(1280, 1280) → SiLU
         参数：2561×1280 + 1280 + 1280×1280 + 1280 ≈ 4.9M
     - coord_mlp：
         Linear(1280, 1280) → SiLU → Linear(1280, 1, gain=0.001)
         参数：1280×1280 + 1280 + 1280×1 + 1 ≈ 1.6M
     - node_gate：
         Linear(1280, 1280) → ReLU → Linear(1280, 1280) → Sigmoid
         参数：1280×1280 + 1280 + 1280×1280 + 1280 ≈ 3.3M
     - att_mlp（注意力）：
         Linear(1280, 1) → Sigmoid
         参数：1280 + 1 ≈ 1.3K
  3. 每层总参数 ≈ 9.8M
  4. 3 层总参数 ≈ 29.4M

  初始化方式：
     Linear 层使用 PyTorch 默认的 Xavier（Kaiming）均匀初始化
     coord_mlp 最后一层使用 gain=0.001（极小初始权重，确保初始坐标更新量极小）
```

**NCBI 嵌入初始化**

```
self.ncbi_embeddings = nn.Embedding(10000, 1280)
nn.init.normal_(self.ncbi_embeddings.weight, mean=0, std=1280**(-0.5))

  - 10000 个物种槽位
  - 每个嵌入向量 1280 维
  - 用均值 0、标准差 1/sqrt(1280) ≈ 0.028 的正态分布初始化
  - 这个标准差的选择使嵌入向量的 L2 范数约为 1.0，
    与 token embedding 的量级匹配，避免加法时一方压倒另一方
```

**参数量总结**

```
模块                    参数量         初始化方式
─────────────────────────────────────────────────
ESM2 编码器             ~650M          预训练权重
EGNN 解码器 (3层)       ~29.4M         Xavier 随机
NCBI 嵌入              ~12.8M         正态分布随机
LM Head                ~42K           预训练权重
─────────────────────────────────────────────────
总计                    ~692M
```

---

## 18. NCBI 物种分类学嵌入

### 18.1 设计动机

不同物种的蛋白质有不同的序列偏好和折叠特征：

```
同一功能（ATP结合）在不同物种中的序列差异：

人类（9606）：    ...GXGXXG...  ← 经典的 P-loop 序列
大肠杆菌（562）：...GXGXXG...  ← 保守，但周围氨基酸不同
结核杆菌（83332）：...GXGXXA... ← 可能有变异

物种 embedding 让模型学习这些偏好差异
```

### 18.2 实现方式

```python
# 来自 geometric_protein_model.py (line 186-187)
self.ncbi_embeddings = nn.Embedding(10000, args.encoder_embed_dim)
# 10000 个物种槽位，每个 1280 维

# 在 forward 中 (line 227-228)：
ncbi_emb = self.ncbi_embeddings(ncbi.reshape(-1, 1)).reshape(batch_size, 1, -1)
# ncbi: [B] → reshape → [B, 1] → embedding → [B, 1, 1280]
x = x + ncbi_emb
# x: [B, L, 1280]
# ncbi_emb: [B, 1, 1280]  ← 广播到每个位置
```

**关键设计**：物种 embedding 是**全局加到每个位置**的（通过广播），而不是只加到第一个位置。这让每个残基都"知道"自己来自哪个物种。

### 18.3 物种 ID 映射

```python
# 来自 indexed_dataset.py (line 458-459)
self.ncbi2id = json.load(open("data/ncbi2id.json", "r"))
# ncbi2id 示例：
# {"562": 0, "9606": 1, "83332": 2, ...}
# 将原始 NCBI Taxonomy ID 映射到 [0, 9999] 的连续整数索引
```

### 18.4 NCBI 嵌入的数值示例

下面用一个具体的例子追踪 NCBI 物种嵌入从输入到添加的全过程。

**场景设定**

```
假设：
  batch_size = 2
  序列 0 来自大肠杆菌（E. coli），NCBI Taxonomy ID = 562
  序列 1 来自人类（Homo sapiens），NCBI Taxonomy ID = 9606
  seq_len = 150（加 BOS/EOS 后为 152）
```

**第 1 步：NCBI ID 到整数索引的映射**

```
ncbi2id.json 中的映射：
  {"562": 0, "9606": 1, "83332": 2, "10090": 3, ...}

数据加载时：
  序列 0：ncbi_taxonomy_id = 562 → ncbi2id["562"] = 0
  序列 1：ncbi_taxonomy_id = 9606 → ncbi2id["9606"] = 1

batch 中的 ncbi 张量：
  ncbi = tensor([0, 1])    shape: [2]
```

**第 2 步：嵌入查找**

```python
ncbi_emb = self.ncbi_embeddings(ncbi.reshape(-1, 1)).reshape(batch_size, 1, -1)

逐步追踪：
  ncbi.reshape(-1, 1) → tensor([[0], [1]])        shape: [2, 1]

  self.ncbi_embeddings(tensor([[0], [1]]))
    → 查找 Embedding(10000, 1280) 的第 0 行和第 1 行
    → tensor([[[0.03, -0.01, 0.02, ..., -0.04]],    # E. coli 的嵌入向量
              [[-0.02, 0.05, -0.01, ..., 0.03]]])   # 人类的嵌入向量
    shape: [2, 1, 1280]

  .reshape(batch_size, 1, -1)
    → 不变，仍然是 [2, 1, 1280]
```

**第 3 步：广播加到 token embedding 上**

```python
x = x + ncbi_emb

维度分析：
  x:        [2, 152, 1280]    ← 每个 batch、每个位置都有 1280 维特征
  ncbi_emb: [2,   1, 1280]    ← 每个 batch 只有 1 个嵌入向量

广播规则：
  ncbi_emb 的第 1 维从 1 广播到 152
  → 等价于将 ncbi_emb 复制 152 次，加到每个位置上

效果：
  x[0, 0, :] += E_coli_embedding    ← 序列0 的第 1 个位置
  x[0, 1, :] += E_coli_embedding    ← 序列0 的第 2 个位置
  ...
  x[0, 151, :] += E_coli_embedding  ← 序列0 的第 152 个位置
  x[1, 0, :] += Human_embedding     ← 序列1 的第 1 个位置
  ...
  x[1, 151, :] += Human_embedding   ← 序列1 的第 152 个位置
```

**直觉理解**

NCBI 嵌入相当于给整条序列的每个位置都打上一个"物种标签"。经过训练后，这个标签向量会编码该物种的特有偏好：

```
E. coli 嵌入向量（训练后）可能编码了：
  - 偏好较高的 GC 含量对应的密码子（影响氨基酸偏好）
  - 偏好较低的二硫键（E. coli 胞浆是还原性环境）
  - 偏好特定的信号肽序列

人类嵌入向量（训练后）可能编码了：
  - 偏好更复杂的蛋白质折叠
  - 偏好更多的翻译后修饰位点
  - 偏好不同的密码子使用模式

这些偏好通过加法的方式"偏移"了每个位置的表示，
将其推向该物种在嵌入空间中的"区域"。
```

### 18.5 10000 个物种槽位够用吗

EnzyGen2 的 NCBI 嵌入层有 10000 个槽位（`nn.Embedding(10000, 1280)`）。这个数字是如何确定的，以及它是否足够？

**训练数据的物种覆盖**

```
EnzyGen2 的训练数据来自 UniProt/Swiss-Prot 和 PDB 中的酶序列。
这些数据库中的蛋白质来源物种分布大致如下：

  训练集覆盖的独特物种数 ≈ 6,000-8,000 种
  高频物种（>100 条序列）≈ 500 种
  中频物种（10-100 条序列）≈ 2,000 种
  低频物种（<10 条序列）≈ 4,000-6,000 种

10000 个槽位提供了约 2,000-4,000 个空余位置，
足够容纳训练过程中发现的新物种，以及未来可能加入的物种。
```

**未知物种的处理**

```
当遇到训练集中从未出现过的物种时（即 NCBI ID 不在 ncbi2id.json 中）：

方案 1：映射到默认 ID（通常是 0）
  → 使用第 0 个嵌入向量（通常对应最常见的物种）
  → 模型的行为退化为"不考虑物种信息"（使用一个通用的偏置）

方案 2：随机分配一个未使用的 ID
  → 嵌入向量保持随机初始化的值
  → 由于训练时未见过这个 ID，嵌入向量不含有意义的信息
  → 效果类似于不加物种嵌入

实际中，对于酶设计任务，大多数目标物种（如模式生物 E. coli、
酵母、人类等）都已在训练集中充分覆盖。
```

**10000 够用吗？数学分析**

```
NCBI Taxonomy 数据库总共有 ~2,300,000 个物种条目。
但蛋白质数据库中有序列的物种只有 ~300,000 种。
其中有结构信息（PDB）的只有 ~60,000 种。
有酶活性注释的更少，约 ~15,000 种。

10000 个槽位覆盖了酶数据中最常见的约 67% 的物种。
考虑到长尾分布（少数常见物种贡献了绝大多数数据），
10000 个槽位在实际中覆盖了 >95% 的训练样本。

嵌入层内存占用：10000 × 1280 × 4 bytes ≈ 48.8 MB
这相比模型总参数量（~692M × 4 bytes ≈ 2.6 GB）微不足道。
即使增加到 100000 个槽位，额外内存也只有 ~488 MB。
```

**嵌入是端到端学习的**

NCBI 嵌入不是预定义的（如 one-hot 编码），而是通过反向传播端到端学习的。这意味着：

1. 嵌入向量会自动捕捉与序列-结构预测相关的物种特征
2. 同一分类群（如所有大肠杆菌菌株）的嵌入向量在训练后会趋于相似
3. 功能相似的物种（如都能在高温下存活的嗜热菌）的嵌入向量可能在某些维度上聚类
4. 罕见物种的嵌入向量由于训练信号不足，可能与初始值相差不大

---

## 19. 编码器-解码器交错：每 11 层 Transformer 接一层 EGNN

### 19.1 交错设计的动机

```
传统设计：先跑完所有 Transformer 层，再跑 EGNN
  问题：序列处理和结构处理是串行的，信息流是单向的

EnzyGen2 的设计：Transformer 和 EGNN 交替运行
  优势：
  1. 序列信息 → 结构 → 更新后的结构信息 → 回到序列处理
  2. 多次信息交换，序列和结构互相修正
  3. EGNN 的坐标更新反馈到后续 Transformer 层
```

### 19.2 交错时机

```
ESM2 有 33 层 Transformer，EGNN 有 3 层：

Transformer 层 1-11  → EGNN 层 0  (layer_idx=10, 10+1=11, 11%11==0)
Transformer 层 12-22 → EGNN 层 1  (layer_idx=21, 21+1=22, 22%11==0)
Transformer 层 23-33 → EGNN 层 2  (layer_idx=32, 32+1=33, 33%11==0)

每 11 层 Transformer 后执行一次 EGNN，总共 3 次
```

### 19.3 交错执行的代码

```python
# 来自 geometric_protein_model.py (line 268-291)
for layer_idx, layer in enumerate(self.encoder.layers):
    # Transformer 层
    x, attn = layer(x, self_attn_padding_mask=padding_mask, ...)
    
    if (layer_idx + 1) % 11 == 0:  # 每 11 层后
        decoder_layer_idx = int(layer_idx / 11)
        
        # 维度转换：(T,B,E) → (B*L, E)
        x = x.transpose(0, 1).reshape(-1, x.size()[-1])
        
        # 重建 KNN 图
        coords = coords.view(batch_size, -1, 3)
        edges = get_edges_batch(n_nodes, batch_size, coords.detach().cpu(), k)
        coords = coords.reshape(-1, 3)
        
        # EGNN 层
        x, coords, _ = self.decoder._modules["gcl_%d" % decoder_layer_idx](
            x, edges, coords, edge_attr=None, batch_size=batch_size, k=k
        )
        
        # 维度转回：(B*L, E) → (T, B, E)
        x = x.reshape(batch_size, -1, x.size()[-1]).transpose(0, 1)
```

### 19.4 维度变换的完整追踪

```
Transformer 内部：
  x: (T, B, E) = (L, B, 1280)  ← T-first format for attention efficiency

交错到 EGNN 前：
  x: (T, B, E) → transpose → (B, T, E) → reshape → (B*L, E)
  coords: (B*L, 3) → view → (B, L, 3) [for KNN] → reshape → (B*L, 3)

EGNN 内部：
  x: (B*L, E) = (B*L, 1280)
  coords: (B*L, 3)
  edges: [2, B*L*K]

EGNN 结束后回到 Transformer：
  x: (B*L, E) → reshape → (B, L, E) → transpose → (L, B, E) = (T, B, E)
```

### 19.5 交错执行的数值追踪

下面用具体的张量形状和数值追踪一次完整的交错执行过程。

**设定**

```
batch_size B = 2
序列长度 L = 152（含 BOS/EOS）
隐藏维度 E = 1280
K = 30（KNN 近邻数）
```

**第 1 轮：Transformer 1-11 → EGNN 0**

```
═══ Transformer 层 1-11 ═══

输入 x：[L, B, E] = [152, 2, 1280]（T-first 格式）
  为什么用 T-first？因为 PyTorch 的 MultiheadAttention 默认期望这个格式，
  避免了额外的转置操作。

每层 Transformer 的操作：
  Self-Attention：
    Q = W_Q × x → [152, 2, 1280]
    K = W_K × x → [152, 2, 1280]
    V = W_V × x → [152, 2, 1280]
    拆分成 20 个头：[152, 2, 20, 64] → 注意力计算 → 合并 → [152, 2, 1280]
    + 残差连接 + LayerNorm

  FFN：
    Linear(1280→5120) → GELU → Linear(5120→1280)
    + 残差连接 + LayerNorm

  输出 x：仍然是 [152, 2, 1280]

经过 11 层后：x_11 = [152, 2, 1280]

═══ 维度转换：Transformer → EGNN ═══

x_11.transpose(0, 1) → [2, 152, 1280]（B-first）
.reshape(-1, 1280)   → [304, 1280]（B*L 个节点，每个 1280 维）

coords（此时仍是初始坐标）：
  原始 [2, 152, 3] → reshape → [304, 3]

═══ KNN 图构建 ═══

coords.view(2, 152, 3)：取出 [2, 152, 3] 的坐标
对每个 batch 独立构建 Ball Tree，K=30

edges：[2, 304*30] = [2, 9120]
  第一行是源节点索引，第二行是目标节点索引
  batch 0 的节点 ID 是 0-151，batch 1 的节点 ID 是 152-303

═══ EGNN 层 0 ═══

输入：
  h = [304, 1280]（节点特征）
  x = [304, 3]（节点坐标）
  edges = [2, 9120]（边索引）

边模型：
  对每条边拼接 source(1280) + target(1280) + radial(1) = [9120, 2561]
  edge_mlp: [9120, 2561] → [9120, 1280]

坐标模型：
  coord_mlp: [9120, 1280] → [9120, 1]（标量权重）
  加权坐标差，聚合后更新坐标
  coords_new: [304, 3]

节点模型（门控）：
  agg: [304, 1280]（聚合后的消息）
  gate = sigmoid(node_gate(agg)): [304, 1280]
  h_new = h + gate * agg: [304, 1280]

输出：h = [304, 1280], coords = [304, 3]

═══ 维度转换：EGNN → Transformer ═══

h.reshape(2, 152, 1280)  → [2, 152, 1280]
.transpose(0, 1)         → [152, 2, 1280]（T-first，回到 Transformer 格式）

coords 保持为 [304, 3]（坐标不进入 Transformer，单独存储）
```

**第 2 轮：Transformer 12-22 → EGNN 1**

```
═══ Transformer 层 12-22 ═══

输入 x：[152, 2, 1280]（来自 EGNN 0 更新后的特征）
  注意：这里的 x 已经融合了第一轮 EGNN 的几何信息
  Transformer 现在在"知道一些结构信息"的基础上做进一步的序列推理

经过 11 层后：x_22 = [152, 2, 1280]

═══ EGNN 层 1（流程同上）═══

关键区别：
  1. coords 已经被 EGNN 层 0 更新过 → KNN 图基于更好的坐标
  2. h 经过了更多的 Transformer 处理 → 包含更丰富的序列上下文
```

**第 3 轮：Transformer 23-33 → EGNN 2**

```
═══ Transformer 层 23-33 ═══

输入 x：[152, 2, 1280]（已经过两轮交错精化）

经过 11 层后：x_33 = [152, 2, 1280]

═══ EGNN 层 2（最后一轮几何精化）═══

输出：
  h_final = [304, 1280] → reshape → [2, 152, 1280]
  coords_final = [304, 3] → reshape → [2, 152, 3]
```

**参数量对比：Transformer vs EGNN**

```
33 层 Transformer 的参数量：
  每层：
    Self-Attention：W_Q + W_K + W_V + W_O = 4 × (1280 × 1280) ≈ 6.6M
    FFN：Linear(1280→5120) + Linear(5120→1280) ≈ 13.1M
    LayerNorm × 2 ≈ 5.1K
    每层小计 ≈ 19.7M
  33 层总计 ≈ 650M

3 层 EGNN 的参数量：
  每层 ≈ 9.8M（edge_mlp + coord_mlp + node_gate + att_mlp）
  3 层总计 ≈ 29.4M

比例：EGNN / Transformer ≈ 29.4M / 650M ≈ 4.5%

EGNN 只占模型总参数量的不到 5%，却赋予了模型处理三维几何的能力。
这说明几何信息的注入不需要很多参数——关键是在正确的位置注入正确的归纳偏置。
```

---

## 20. 被遮蔽残基的坐标初始化

### 20.1 问题

被遮蔽（需要设计）的残基，我们不知道其真实坐标。但 EGNN 需要初始坐标来构建 KNN 图。怎么初始化？

### 20.2 随机游走初始化

EnzyGen2 使用**球面随机游走**：从前一个残基出发，沿随机方向走一步（3.75Å）：

```python
# 来自 geometric_protein_model.py (line 256-264)
coords = torch.empty_like(coors).copy_(coors)  # 复制真实坐标
for i in range(coords.size(0)):        # 遍历 batch
    for j in range(coords.size(1)):    # 遍历残基
        if input_mask[i][j] == 1:      # 如果是被遮蔽的位置
            theta = np.random.uniform(0, np.pi)       # 极角 [0, π]
            phi = np.random.uniform(0, np.pi * 2)     # 方位角 [0, 2π]
            # 在球面上随机选一个方向，步长 3.75Å
            coords[i][j][0] = coords[i][j-1][0] + 3.75 * np.sin(theta) * np.sin(phi)
            coords[i][j][1] = coords[i][j-1][1] + 3.75 * np.sin(theta) * np.cos(phi)
            coords[i][j][2] = coords[i][j-1][2] + 3.75 * np.cos(theta)
```

### 20.3 为什么是 3.75Å

```
3.75Å ≈ Cα-Cα 之间的平均距离

在真实蛋白质中：
  相邻残基的 Cα-Cα 距离 ≈ 3.8Å（肽键几何约束）
  
EnzyGen2 用 3.75Å 作为步长，使初始坐标的局部几何大致合理。
EGNN 然后会逐步修正这些初始猜测，将残基移动到正确位置。
```

### 20.4 为什么不用全零初始化或全局随机初始化

在设计被遮蔽残基的坐标初始化策略时，有三种显而易见的方案。EnzyGen2 选择了方案 3（随机游走），而不是方案 1 或方案 2。下面分析各方案的优劣。

**方案 1：全零初始化（所有被遮蔽残基坐标设为原点 (0,0,0)）**

```
问题：KNN 图退化

  所有被遮蔽残基都在 (0,0,0)：
    ● ● ● ● ● ← 30 个遮蔽残基堆在原点
    
    距离矩阵变为：
      d(遮蔽_i, 遮蔽_j) = 0   ← 所有遮蔽残基之间距离为 0！

  KNN 图构建时：
    每个遮蔽残基的 K=30 个最近邻几乎全是其他遮蔽残基（距离=0）
    → 遮蔽残基之间形成一个"全连接的零距离团"
    → 非遮蔽残基（motif 残基）被排除在邻居之外

  EGNN 的后果：
    coord_diff = x_i - x_j = (0,0,0) - (0,0,0) = (0,0,0)
    坐标更新 = Σ (0,0,0) × φ(m) = (0,0,0)
    → 坐标永远无法离开原点！EGNN 完全失效。

  即使做了归一化（除以 norm），norm = 0 导致数值不稳定（除以零）。
```

**方案 2：全局随机初始化（在一个大的立方体内均匀随机撒点）**

```
问题：初始坐标远离正确位置

  假设在 [-50, 50]Å 的立方体内随机初始化：
    遮蔽残基可能分布在离 motif 数十Å 的位置
    
  EGNN 需要做的工作：
    将这些远离的残基"拉回"到蛋白质附近
    步幅限制（coord_mlp 的 gain=0.001）意味着每步只能移动 ~0.01Å
    → 需要数千步才能将残基拉回正确位置
    → 但 EnzyGen2 只有 3 层 EGNN，每层只更新一次！

  KNN 图的问题：
    随机散布的残基之间距离随机，KNN 图毫无物理意义
    一个本应在蛋白质 N-端的残基可能连接到 C-端的残基
    → EGNN 在错误的邻居关系上传递消息，事倍功半
```

**方案 3：从最后已知位置出发的球面随机游走（EnzyGen2 的选择）**

```
优势 1：保持序列局部性
  每个遮蔽残基从前一个残基出发（coords[i][j-1]），步长 3.75Å
  → 初始坐标自然地沿着序列方向展开
  → 保留了"序列上相邻的残基在空间中也大致相邻"的先验

优势 2：与 motif 残基空间连贯
  如果 motif 残基在位置 50-60，被遮蔽残基在位置 61-80
  → 残基 61 从残基 60（已知坐标）出发随机走一步
  → 残基 62 从残基 61 出发随机走一步
  → ...
  → 这些残基自然地延伸在 motif 附近

优势 3：KNN 图更合理
  初始坐标构成一条大致合理的聚合物链
  → KNN 图中的邻居关系大致反映了真实的空间关系
  → EGNN 可以在一个"大致正确"的起点上做精细调整
```

### 20.5 随机游走的数学分析

球面随机游走的统计性质决定了初始坐标的空间分布特征。

**单步分布**

```
每一步的坐标增量：
  Δx = 3.75 × sin(θ) × sin(φ)
  Δy = 3.75 × sin(θ) × cos(φ)
  Δz = 3.75 × cos(θ)

其中 θ ~ Uniform(0, π), φ ~ Uniform(0, 2π)

这定义了一个在球面上均匀分布的方向，乘以固定步长 3.75Å。
每步都是从当前位置出发，向随机方向走 3.75Å。
```

**N 步后的末端距离**

三维球面随机游走（fixed step length）经过 N 步后，末端到起点的期望距离：

```
E[|r_N|] = step_length × √N × √(2/π × 3)
         ≈ step_length × √N × 1.382

更简化的估算：E[|r_N|] ≈ step_length × √N

对于 step_length = 3.75Å：
  N = 10 步：E[|r|] ≈ 3.75 × √10 ≈ 11.9Å
  N = 20 步：E[|r|] ≈ 3.75 × √20 ≈ 16.8Å
  N = 50 步：E[|r|] ≈ 3.75 × √50 ≈ 26.5Å
  N = 100 步：E[|r|] ≈ 3.75 × √100 = 37.5Å
```

**与真实蛋白质尺寸的对比**

```
真实蛋白质的回转半径（radius of gyration）经验公式：
  R_g ≈ 2.2 × N^0.38 Å（其中 N 是残基数）

  N = 50 残基：R_g ≈ 2.2 × 50^0.38 ≈ 9.8Å，直径 ≈ 20Å
  N = 100 残基：R_g ≈ 2.2 × 100^0.38 ≈ 12.7Å，直径 ≈ 25Å
  N = 150 残基：R_g ≈ 2.2 × 150^0.38 ≈ 14.7Å，直径 ≈ 29Å
  N = 200 残基：R_g ≈ 2.2 × 200^0.38 ≈ 16.3Å，直径 ≈ 33Å

对比随机游走的末端距离：
  50 步随机游走：~26.5Å vs 真实 50 残基蛋白直径 ~20Å
  → 随机游走稍微"伸展"了一些，但量级正确

这个匹配并非巧合——真实蛋白质骨架本身就可以近似为一种
受约束的随机游走（constrained random walk），只是带有
二级结构（α-helix、β-sheet）引入的局部有序性。
EnzyGen2 的随机游走初始化抹去了这些局部有序性，
但保留了正确的全局尺度——这正是 EGNN 需要的"初始猜测"。
```

**步长 3.75Å 的精确匹配**

```
在真实蛋白质中：
  相邻残基的 Cα-Cα 距离 ≈ 3.8Å（高度保守，几乎不变）
  这是肽键的几何约束决定的

EnzyGen2 使用 3.75Å（略小于 3.8Å）：
  → 每步距离与真实 Cα-Cα 距离几乎一致
  → 初始链的总轮廓长度（contour length）与真实蛋白质相当
  → 局部链段的伸展度合理

唯一的区别：方向是完全随机的
  真实蛋白质：连续残基的 Cα-Cα 方向受二面角约束
  随机游走：连续步的方向完全独立
  → 这导致随机游走链比真实蛋白质更"松散"（没有 α-helix 的螺旋压缩）
  → 但 EGNN 会在后续步骤中修正这一点
```

---

## 21. GeometricProteinNCBISubstrateModel：配体结合扩展

### 21.1 继承关系

```python
# 来自 geometric_protein_model.py (line 310-311)
@register_model("geometric_protein_model_ncbi_substrate")
class GeometricProteinNCBISubstrateModel(GeometricProteinNCBIModel):
    # 继承基础模型的所有功能，新增配体处理模块
```

### 21.2 新增组件

```python
# line 312-318
def __init__(self, args, encoder, decoder):
    super().__init__(args, encoder, decoder)
    # 配体 EGNN：处理配体的 3D 原子图
    self.substrate_egnn = SubstrateEGNN(
        in_node_nf=args.encoder_embed_dim,   # 1280
        hidden_nf=args.encoder_embed_dim,     # 1280
        out_node_nf=3,                        # 3D 坐标
        n_layers=3,                           # 3 层
        mode=args.egnn_mode                   # "rm-node"
    )
    # 结合预测头：蛋白质表示 + 配体表示 → 结合/不结合
    self.score = nn.Linear(args.encoder_embed_dim * 2, 2)  # [2560] → [2]
```

### 21.3 配体处理与结合预测

```python
# 来自 geometric_protein_model.py (line 409-428)

# 蛋白质全局表示：对序列 embedding 求和池化
protein_rep = torch.sum(x, dim=1)  # [B, L, 1280] → [B, 1280]

if substrate_atom is not None:
    # 处理配体
    sub_atom = substrate_atom.reshape(-1, 5)  # [B*N_atoms, 5]
    sub_coors = substrate_coor.view(batch_size, -1, 3)
    
    # 构建配体原子的 KNN 图
    sub_k = min(self.k, substrate_length - 1)
    edges = get_edges_batch(substrate_length, batch_size, sub_coors.detach().cpu(), sub_k)
    
    # 配体 EGNN 处理
    sub_feats, _ = self.substrate_egnn(sub_atom, sub_coors, edges, k=sub_k)
    
    # 配体全局表示：求和池化
    sub_feats = sub_feats.reshape(batch_size, -1, sub_feats.size()[-1])
    sub_feats = torch.sum(sub_feats, dim=1)  # [B, 1280]
    
    # 拼接蛋白质和配体表示，预测结合概率
    scores = F.softmax(self.score(
        torch.cat((protein_rep, sub_feats), 1)  # [B, 2560]
    ), dim=-1) + 1e-6  # [B, 2] + epsilon 防止 log(0)
```

### 21.4 结合预测头的数学

结合预测头（binding prediction head）将蛋白质和配体的全局表示转化为结合概率。下面逐步追踪其数学运算。

**第 1 步：蛋白质全局表示（sum-pooling）**

```python
protein_rep = torch.sum(x, dim=1)
# x：[B, L, 1280]（经过 Transformer + EGNN 处理后的残基级表示）
# sum over dim=1（序列维度）：
#   protein_rep[b, :] = Σ_{i=0}^{L-1} x[b, i, :]
# protein_rep：[B, 1280]

具体数值示例（B=2, L=3, 简化为 4 维）：

x[0] = [[0.1, 0.5, -0.3, 0.8],    ← 残基 0
        [0.2, 1.0,  0.6, -0.1],    ← 残基 1
        [-0.5, 0.3, 1.0, 0.4]]     ← 残基 2

protein_rep[0] = [0.1+0.2-0.5, 0.5+1.0+0.3, -0.3+0.6+1.0, 0.8-0.1+0.4]
               = [-0.2, 1.8, 1.3, 1.1]
```

**第 2 步：配体全局表示（sum-pooling）**

```python
sub_feats = sub_feats.reshape(batch_size, -1, sub_feats.size()[-1])
sub_feats = torch.sum(sub_feats, dim=1)
# sub_feats：[B, N_atoms, 1280] → sum over dim=1 → [B, 1280]

具体示例（20 个原子）：
sub_feats[0] = Σ_{j=0}^{19} atom_feat[0, j, :]
# 将所有原子的 1280 维特征加在一起
# sub_feats[0]：[1280]
```

**第 3 步：拼接**

```python
concat = torch.cat((protein_rep, sub_feats), dim=1)
# protein_rep：[B, 1280]
# sub_feats：  [B, 1280]
# concat：     [B, 2560]

拼接后的向量前 1280 维编码了蛋白质的全局信息，
后 1280 维编码了配体的全局信息。
```

**第 4 步：线性预测 + Softmax**

```python
logits = self.score(concat)
# self.score = nn.Linear(2560, 2)
# logits：[B, 2]
#   logits[:, 0] = 不结合的原始分数
#   logits[:, 1] = 结合的原始分数

scores = F.softmax(logits, dim=-1) + 1e-6
# softmax 将 logits 转换为概率：
#   P(bind) = exp(logit_1) / (exp(logit_0) + exp(logit_1))
#   P(no_bind) = exp(logit_0) / (exp(logit_0) + exp(logit_1))
# scores：[B, 2]，每行和为 1（加上 epsilon 后略大于 1）

具体数值示例：
  logits[0] = [1.2, 3.5]
  softmax = [exp(1.2)/(exp(1.2)+exp(3.5)), exp(3.5)/(exp(1.2)+exp(3.5))]
           = [3.32/36.47, 33.12/36.47]
           = [0.091, 0.909]
  加 epsilon：[0.091001, 0.909001]

  解读：batch 0 中的蛋白质-配体对有 90.9% 的概率结合
```

**第 5 步：交叉熵损失**

```python
loss_binding = F.cross_entropy(scores.log(), binding_labels)
# 或等价地：
# loss_binding = -Σ_b (y_b × log(P_bind_b) + (1-y_b) × log(P_nobind_b)) / B

# binding_labels：[B]，值为 0（不结合）或 1（结合）
# 对上面的例子：
#   如果 binding_label[0] = 1（确实结合）：
#     loss = -log(0.909) = 0.095  ← 很小，预测正确
#   如果 binding_label[0] = 0（不结合）：
#     loss = -log(0.091) = 2.397  ← 很大，预测错误
```

### 21.5 为什么用 sum-pooling 而不是 mean-pooling

EnzyGen2 在生成蛋白质和配体的全局表示时选择了 **sum-pooling**（求和）而非 **mean-pooling**（求平均）。这个看似简单的选择有重要的实际意义。

**sum-pooling 保留了大小信息**

```
sum-pooling：
  protein_rep = Σ_{i=1}^{L} h_i    ← 随 L 线性增长
  
  150 残基蛋白质的表示 L2 范数 ≈ 150 × avg_norm(h_i)
  300 残基蛋白质的表示 L2 范数 ≈ 300 × avg_norm(h_i)

mean-pooling：
  protein_rep = (1/L) × Σ_{i=1}^{L} h_i    ← 与 L 无关

  150 残基蛋白质和 300 残基蛋白质的表示 L2 范数相近
```

**大小信息对结合预测很重要**

蛋白质与配体的结合倾向与蛋白质大小有关：

```
1. 更大的蛋白质通常有更多的潜在结合位点
   → sum-pooling 自然地给大蛋白质更高的"结合倾向"基线

2. 结合表面积与蛋白质大小正相关
   → 一个 300 残基的蛋白质比 100 残基的蛋白质
     有更多的表面暴露残基可以参与结合

3. 配体同理：更大的配体（更多原子）
   通常与蛋白质有更多的接触点
   → sum-pooling 后配体表示的范数更大
   → 拼接后的向量自然编码了配体大小信息
```

**mean-pooling 的问题**

```
如果使用 mean-pooling：
  一个 20 残基的小肽和一个 500 残基的大蛋白
  产生的全局表示范数相近

  → 模型丧失了区分它们大小的能力
  → 需要额外学习一个"大小编码"来补偿
  → 增加了学习难度

实际例子：
  小肽（20 残基）通常不结合大分子配体
  大蛋白（500 残基）可能有多个结合口袋
  mean-pooling 会抹去这个重要区分
```

**潜在风险与缓解**

```
sum-pooling 的风险：
  如果所有残基的特征向量方向随机，
  sum 的结果可能接近零（正负抵消）
  
  缓解：EGNN 和 Transformer 的处理会让特征向量有方向性，
  同一蛋白质中的残基特征不是完全随机的。
  
另一个风险：
  极长的蛋白质的 sum 可能数值很大，导致 Linear 层的梯度不稳定
  
  缓解：LayerNorm 在 Transformer 最后一层已经将特征归一化到合理范围。
  虽然 sum 会放大数值，但 Linear(2560→2) 的权重会在训练中自适应地缩小，
  以适应输入的范围。
```

---

## 22. 完整前向传播追踪（含每步 shape）

### 22.1 输入

```
假设：batch_size=2, seq_len=150（加BOS/EOS后=152）, K=30
     配体原子数=20

src_tokens:    [2, 152]     ← 整数序列（部分位置为 mask_idx=32）
src_lengths:   [2]          ← 每条序列的长度
coors:         [2, 152, 3]  ← Cα坐标（已中心化，BOS/EOS位置为[0,0,0]）
motifs:
  input:       [2, 152]     ← 二值掩码（1=被遮蔽，0=已知）
  output:      [2, 152]     ← 二值掩码（1=需要优化，0=保持不变）
ncbi:          [2]          ← 物种 ID（整数索引）

[可选]
substrate_coor: [2, 20, 3]  ← 配体原子坐标
substrate_atom: [2, 20, 5]  ← 配体原子特征
```

### 22.2 Step 1: 序列遮蔽

```python
tokens = input_mask * self.mask_index + (input_mask != 1) * src_tokens
# input_mask=1 的位置 → mask_index (32)
# input_mask=0 的位置 → 保留原始 token

# tokens: [2, 152]（部分位置被替换为 32）
```

### 22.3 Step 2: Token Embedding + NCBI Embedding

```python
x = self.encoder.embed_scale * self.encoder.embed_tokens(tokens)
# embed_scale = 1（ESM2不缩放）
# embed_tokens: Embedding(33, 1280)
# x: [2, 152, 1280]

ncbi_emb = self.ncbi_embeddings(ncbi.reshape(-1, 1)).reshape(batch_size, 1, -1)
# ncbi: [2] → [2, 1] → Embedding(10000, 1280) → [2, 1, 1280]
x = x + ncbi_emb  # 广播加法
# x: [2, 152, 1280]
```

### 22.4 Step 3: Token Dropout

```python
# 被遮蔽位置置零
x.masked_fill_((tokens == 32).unsqueeze(-1), 0.0)
# x: [2, 152, 1280]（遮蔽位置的 1280维向量全零）

# 缩放补偿
mask_ratio_train = 0.12
mask_ratio_observed = (tokens == 32).sum(-1).float() / src_lengths
x = x * (1 - 0.12) / (1 - mask_ratio_observed)[:, None, None]
# x: [2, 152, 1280]
```

### 22.5 Step 4: Padding 处理

```python
padding_mask = tokens.eq(self.encoder.padding_idx)  # [2, 152] bool
x = x * (1 - padding_mask.unsqueeze(-1).float())    # padding位置置零
```

### 22.6 Step 5: 坐标初始化

```python
coords = torch.empty_like(coors).copy_(coors)  # [2, 152, 3]
# 对被遮蔽位置进行球面随机游走初始化
# 每个被遮蔽残基：从前一个残基出发，随机方向走 3.75Å
coords = coords.reshape(-1, 3)  # [304, 3]（B*L = 2*152 = 304）
```

### 22.7 Step 6: 转置进入 Transformer

```python
x = x.transpose(0, 1)  # [2, 152, 1280] → [152, 2, 1280]（T-first）
```

### 22.8 Step 7: 33 层 Transformer + 3 层 EGNN 交错

```
Layer 1-10:  Transformer [152, 2, 1280] → [152, 2, 1280]
Layer 11:    Transformer [152, 2, 1280] → [152, 2, 1280]
             ↓ 转换维度
             x: [304, 1280], coords: [304, 3]
             ↓ 构建 KNN (K=30): edges [2, 304*30]
             ↓ EGNN Layer 0
             x: [304, 1280], coords: [304, 3]（坐标已更新）
             ↓ 转回 Transformer 格式
             x: [152, 2, 1280]

Layer 12-21: Transformer [152, 2, 1280] → [152, 2, 1280]
Layer 22:    Transformer → EGNN Layer 1（同上流程）

Layer 23-32: Transformer [152, 2, 1280] → [152, 2, 1280]
Layer 33:    Transformer → EGNN Layer 2（同上流程）
```

### 22.9 Step 8: 最终输出

```python
x = self.encoder.emb_layer_norm_after(x)  # LayerNorm
x = x.transpose(0, 1)  # [152, 2, 1280] → [2, 152, 1280]

# 序列预测
x = self.encoder.lm_head(x)      # Linear(1280→33)
encoder_prob = F.softmax(x, -1)   # [2, 152, 33]

# 坐标输出
coords = coords.view(batch_size, -1, 3)  # [2, 152, 3]

return encoder_prob, coords  # 序列概率 + 预测坐标
```

### 22.10 Step 9（可选）: 配体结合预测

```python
# 蛋白质全局表示
protein_rep = torch.sum(x_before_lm_head, dim=1)  # [2, 1280]

# 配体处理
sub_atom: [40, 5] → Linear(5→1280) → [40, 1280]
# 构建配体 KNN 图，执行 3 层 EGNN
sub_feats: [40, 1280] → reshape → [2, 20, 1280] → sum → [2, 1280]

# 结合预测
concat = torch.cat((protein_rep, sub_feats), dim=1)  # [2, 2560]
scores = softmax(Linear(2560→2)(concat), dim=-1)      # [2, 2]
```

### 22.11 前向传播的计算代价分析

理解前向传播中各模块的计算量分布，有助于优化训练效率和推理速度。下面以 150 个残基（加 BOS/EOS 后 L=152）的蛋白质为例，分析 FLOP（浮点运算次数）。

**Transformer 自注意力的计算量**

```
每层自注意力的核心操作：

1. QKV 投影：3 × (L × d × d)
   = 3 × (152 × 1280 × 1280)
   = 3 × 248,832,000
   ≈ 747M FLOPs

2. 注意力分数计算：L × L × d
   = 152 × 152 × 1280
   ≈ 29.6M FLOPs
   （注意：实际是 20 个头各做 152×152×64，总量相同）

3. 注意力加权求和：L × L × d
   = 152 × 152 × 1280
   ≈ 29.6M FLOPs

4. 输出投影：L × d × d
   = 152 × 1280 × 1280
   ≈ 249M FLOPs

每层自注意力总计 ≈ 1,055M FLOPs
33 层总计 ≈ 34.8G FLOPs
```

**Transformer FFN 的计算量**

```
每层 FFN：
  Linear(1280 → 5120)：L × d × 4d = 152 × 1280 × 5120 ≈ 996M FLOPs
  GELU 激活：可忽略
  Linear(5120 → 1280)：L × 4d × d = 152 × 5120 × 1280 ≈ 996M FLOPs

每层 FFN 总计 ≈ 1,992M FLOPs
33 层总计 ≈ 65.7G FLOPs
```

**Transformer 总计**

```
33 层 Transformer（Self-Attention + FFN）：
  ≈ 34.8G + 65.7G = 100.5G FLOPs
```

**EGNN 的计算量**

```
每层 EGNN（E_GCL_RM_Node）：

1. 边模型（edge_mlp）：
   输入拼接：L×K 条边，每条 2561 维
   Linear(2561→1280)：L×K × 2561 × 1280
   = 152 × 30 × 2561 × 1280
   ≈ 14.95G FLOPs
   Linear(1280→1280)：= 152 × 30 × 1280 × 1280 ≈ 7.47G FLOPs
   每层边模型 ≈ 22.4G FLOPs

2. 坐标模型（coord_mlp）：
   Linear(1280→1280)：152 × 30 × 1280 × 1280 ≈ 7.47G FLOPs
   Linear(1280→1)：152 × 30 × 1280 × 1 ≈ 5.8M FLOPs
   坐标聚合和更新：O(L×K×3) ≈ 13.7K FLOPs（可忽略）
   每层坐标模型 ≈ 7.48G FLOPs

3. 节点模型（node_gate）：
   聚合消息：O(L×K×d) ≈ 5.8M FLOPs
   Linear(1280→1280)：152 × 1280 × 1280 ≈ 249M FLOPs
   Linear(1280→1280)：同上 ≈ 249M FLOPs
   门控乘法：O(L×d) ≈ 195K FLOPs
   每层节点模型 ≈ 504M FLOPs

每层 EGNN 总计 ≈ 30.4G FLOPs
3 层 EGNN 总计 ≈ 91.2G FLOPs
```

**KNN 图构建**

```
Ball Tree 构建：O(L × log L) ≈ 152 × 7.2 ≈ 1.1K 次比较
K 近邻查询：O(L × K × log L) ≈ 152 × 30 × 7.2 ≈ 32.8K 次比较
3 次重建总计 ≈ ~100K 次比较

注意：KNN 在 CPU 上执行，不计入 GPU FLOPs。
在实际运行中，KNN 重建的墙钟时间约占总时间的 5-10%
（受 CPU-GPU 数据传输瓶颈影响）。
```

**配体处理（如果有）**

```
SubstrateEGNN（20 个原子，K=19）：
  3 层 EGNN，每层参数与蛋白质 EGNN 相同
  但 L=20 远小于 152 → 计算量约为蛋白质 EGNN 的 (20×19)/(152×30) ≈ 8.3%
  约 2.5G FLOPs

结合预测头：
  Linear(2560→2)：B × 2560 × 2 ≈ 10K FLOPs（可忽略）
```

**总结**

```
模块                     FLOPs          占比
────────────────────────────────────────────
Transformer Self-Attn    34.8G          18.1%
Transformer FFN          65.7G          34.2%
EGNN (3 层)              91.2G          47.5%
配体 EGNN (3 层)         ~2.5G           1.3%
KNN 重建                 ~0             ~0%
其他（Embedding等）      ~0.2G           0.1%
────────────────────────────────────────────
总计                     ~192G          100%

注意：EGNN 的 FLOPs 占比较高（~47.5%），主要来自 edge_mlp
中 L×K 条边的高维矩阵乘法。但由于 EGNN 只有 3 层（vs
Transformer 的 33 层），实际的参数量占比很低（~4.5%）。

在 GPU 上的实际运行时间中，Transformer 通常占主导地位（>60%），
因为 Transformer 的自注意力操作可以被 CUDA 高度优化（Flash Attention），
而 EGNN 的稀疏图操作并行度较低。EGNN 的理论 FLOPs 虽然高，
但由于 edge_mlp 的 batch 维度是 L×K（而非 L×L），
实际 GPU 利用率可能不如 Transformer 的密集矩阵乘法。
```

---

## 23. 序列恢复损失（Sequence Recovery Loss）

### 23.1 数学定义

```
L_seq = -1/|M| × Σ_{i∈M} log P(a_i | context)

其中：
  M = 被遮蔽位置的集合（output_mask == 1 的位置）
  a_i = 位置 i 的真实氨基酸
  P(a_i | context) = 模型预测的概率
  |M| = 被遮蔽位置的数量
```

### 23.2 代码实现

```python
# 来自 geometric_protein_ncbi_loss.py (line 64-65)

# 从 softmax 概率分布中取出真实 token 对应的概率
loss_encoder = -torch.log(
    encoder_out.gather(dim=-1, index=src_tokens.unsqueeze(-1)).squeeze(-1)
)
# encoder_out: [B, L, 33]
# src_tokens.unsqueeze(-1): [B, L, 1]
# gather 结果: [B, L]（每个位置的真实 token 概率）
# -log: 负对数似然

# 只在被遮蔽位置计算
loss_encoder = torch.mean(torch.sum(loss_encoder * output_mask, dim=-1))
# output_mask: [B, L]（1=需要计算损失，0=忽略）
# 先按序列维度求和，再取 batch 平均
```

### 23.3 shape 追踪

```
encoder_out:          [2, 152, 33]
src_tokens:           [2, 152]
src_tokens.unsqueeze: [2, 152, 1]
gather 结果:          [2, 152]     ← 每个位置的真实 token 概率
-log:                 [2, 152]     ← 负对数似然
× output_mask:        [2, 152]     ← 遮蔽位置置零
sum(dim=-1):          [2]          ← 每条序列的总损失
mean:                 标量          ← batch 平均
```

### 23.4 序列恢复损失的数值示例

下面用一个具体的数值例子，逐步展示序列恢复损失是如何计算的。

**场景设定：** 一条 150 残基的蛋白质，其中 100 个位置被遮蔽（output_mask=1），50 个位置是已知的 Motif（output_mask=0）。

**Step 1：模型输出概率分布**

对于每一个被遮蔽的位置，模型通过 softmax 输出一个 33 维的概率向量（对应 33 个 token）。以第 i=42 个位置为例：

```
模型预测概率（33维向量）：
  [CLS]=0.00, [PAD]=0.00, [EOS]=0.00, [UNK]=0.00,
  L=0.05,   A=0.02,   G=0.03,   V=0.01,   E=0.01,   S=0.02,
  I=0.80,   D=0.01,   ...（其余每个 ≈ 0.003）...
  总和 = 1.0

真实氨基酸：Leucine (L)，在 ESM 词表中的索引 = 4
模型对真实答案的预测概率：P(L) = 0.05
```

**Step 2：计算该位置的损失**

```
loss_i = -log(P(真实氨基酸))
       = -log(0.05)
       = -(-2.996)
       = 2.996

直觉解释：模型只给了正确答案 5% 的概率，说明预测很差，
所以损失值较高（接近 3.0）。
```

**Step 3：对比——如果模型预测得好**

```
假设模型对另一个位置 j=78 的预测：
  真实氨基酸：Alanine (A)
  模型预测 P(A) = 0.90

  loss_j = -log(0.90) = 0.105

直觉解释：模型有 90% 的信心选择了正确答案，损失非常低。
```

**Step 4：对比——如果模型完全错误**

```
极端情况：模型给真实氨基酸的概率几乎为零
  P(真实) = 0.001

  loss = -log(0.001) = 6.908

这是非常高的损失，表明模型完全没有学到这个位置的模式。
```

**Step 5：汇总——整条序列的损失**

```
100 个被遮蔽位置的损失值：
  位置 1:   -log(0.85) = 0.163
  位置 2:   -log(0.42) = 0.868
  位置 3:   -log(0.05) = 2.996
  ...
  位置 100: -log(0.73) = 0.315

50 个未遮蔽位置：不参与计算（乘以 output_mask=0 后归零）

序列总损失 = Σ(100个位置的loss) = 假设总和为 142.5
代码中的处理：sum(dim=-1) 得到 142.5（注意代码没有除以 |M|）

最终 loss = mean(batch维度) → 如果 batch=2，两条序列的总损失取平均
```

**关键细节：代码中的"求和"而非"求均值"**

注意代码是 `torch.mean(torch.sum(loss * mask, dim=-1))`，即先对序列维度**求和**，再对 batch 维度**求均值**。这意味着**更长的蛋白质（更多遮蔽位置）会贡献更大的损失**。这在 `max_tokens` 控制下通常不会造成问题，因为每个 batch 的总 token 数是固定的。

### 23.5 为什么用 NLL 而不是交叉熵

初学者可能会问：序列恢复损失看起来很像交叉熵（cross-entropy），为什么代码中用的是 NLL（负对数似然）的形式？

**数学上两者完全等价（当标签是 one-hot 时）**

```
交叉熵（完整形式）：
  CE = -Σ_{c=1}^{33} y_c × log(P_c)

当 y 是 one-hot 向量时（只有真实类别 c* 处 y_{c*}=1，其余为 0）：
  CE = -1 × log(P_{c*}) - 0 × log(P_1) - 0 × log(P_2) - ...
     = -log(P_{c*})
     = NLL

结论：one-hot 标签 + 交叉熵 = NLL，两者数学上完全相同。
```

**代码中为什么用 gather() 而不是完整的交叉熵？**

```python
# 方法 A：完整交叉熵（效率低）
y_onehot = F.one_hot(src_tokens, num_classes=33).float()  # [B, L, 33]
loss = -torch.sum(y_onehot * torch.log(encoder_out), dim=-1)  # [B, L]
# 需要创建 33 维的 one-hot 张量，做 33 次乘法

# 方法 B：gather + NLL（代码实际使用，效率高）
loss = -torch.log(
    encoder_out.gather(dim=-1, index=src_tokens.unsqueeze(-1)).squeeze(-1)
)
# gather 直接用索引从 33 维中取出第 c* 个值
# 不需要创建 one-hot 张量
# 计算量从 O(33) 降到 O(1)
```

效率差异：gather 方法在 GPU 上只需要一次内存访问（按索引取值），而完整交叉熵需要分配 one-hot 张量并做逐元素乘法。当 batch 很大时（比如 B=8, L=1024），这个差距会累积。

**softmax 与 log_softmax 的区别**

在代码中，`encoder_out` 已经经过了 softmax（概率值在 [0, 1] 之间），所以用 `-torch.log()` 计算 NLL。另一种常见做法是：

```python
# 很多框架推荐的方式：
log_probs = F.log_softmax(logits, dim=-1)       # 数值上更稳定
loss = F.nll_loss(log_probs, targets)

# EnzyGen2 的方式：
probs = F.softmax(logits, dim=-1)               # 先转概率
loss = -torch.log(probs.gather(...))             # 再取 log
```

为什么 EnzyGen2 不用 `log_softmax`？因为 `encoder_out`（概率分布）在推理时也要使用（用于 top-p 采样等），所以需要保留概率值而不是对数概率。如果只保存 log_softmax 的结果，推理时还需要 exp() 转回概率，不够方便。代价是训练时可能有轻微的数值精度损失（当概率接近 0 时，先 softmax 再 log 可能产生数值不稳定），但在实践中影响不大。

---

## 24. 结构预测损失（Structure Prediction Loss）

### 24.1 数学定义

```
L_struct = 1/B × Σ_{b=1}^{B} Σ_{i∈M_b} ||x̂_i - x_i||²

其中：
  x̂_i = 模型预测的 Cα 坐标 [3维向量]
  x_i = 真实 Cα 坐标 [3维向量]
  M_b = 第 b 条序列中被遮蔽位置的集合
  B = batch 大小
```

这是标准的**均方误差（MSE）损失**，在三维坐标上计算。

### 24.2 代码实现

```python
# 来自 geometric_protein_ncbi_loss.py (line 68-69)

loss_decoder = torch.mean(
    torch.sum(
        torch.sum(
            torch.square(decoder_out - target_coor),  # [B, L, 3] - [B, L, 3]
            dim=-1                                     # 对3维求和 → [B, L]
        ) * output_mask,                               # [B, L] × [B, L]
        dim=-1                                         # 对L求和 → [B]
    )                                                  # mean → 标量
)
```

### 24.3 shape 追踪

```
decoder_out:   [2, 152, 3]
target_coor:   [2, 152, 3]
差值:          [2, 152, 3]
平方:          [2, 152, 3]
sum(dim=-1):   [2, 152]      ← 每个残基的坐标 MSE
× output_mask: [2, 152]      ← 只计算遮蔽位置
sum(dim=-1):   [2]            ← 每条序列的总坐标损失
mean:          标量
```

### 24.4 NaN 保护

```python
# line 70-74
if torch.isfinite(loss_decoder):
    loss = self.encoder_factor * loss_encoder + self.decoder_factor * loss_decoder
else:
    loss = self.encoder_factor * loss_encoder
    # 如果坐标损失产生 NaN/Inf，只用序列损失
```

### 24.5 结构损失的物理含义

结构预测损失是 MSE（均方误差），它度量的是预测的 Cα 原子与真实 Cα 原子之间的空间距离。理解这个损失的物理含义，对于判断模型训练是否正常至关重要。

**从损失值到实际距离**

```
对于单个残基 i，MSE 的计算方式：
  Δx = x̂ - x,  Δy = ŷ - y,  Δz = ẑ - z
  MSE_i = Δx² + Δy² + Δz²

如果 MSE_i = 1.0 Å²，这意味着什么？
  假设误差在三个方向上均匀分布：Δx² ≈ Δy² ≈ Δz² ≈ 1/3
  每个方向的平均误差 ≈ sqrt(1/3) ≈ 0.577 Å
  实际的空间距离（欧几里得距离）= sqrt(MSE_i) = sqrt(1.0) = 1.0 Å

如果 MSE_i = 3.0 Å²：
  空间距离 = sqrt(3.0) ≈ 1.73 Å
  每个方向的平均误差 ≈ 1.0 Å

如果 MSE_i = 9.0 Å²：
  空间距离 = sqrt(9.0) = 3.0 Å
  这已经超过了一个 Cα-Cα 键长（3.8Å），说明预测很差
```

**从损失到 RMSD 的转换**

```
RMSD（Root Mean Square Deviation）是结构生物学中标准的距离度量：
  RMSD = sqrt( (1/|M|) × Σ_{i∈M} ||x̂_i - x_i||² )

而代码中的损失是 SUM（不是 MEAN）：
  L_struct = Σ_{i∈M} ||x̂_i - x_i||²   （对序列维度求和）

所以：RMSD = sqrt(L_struct / |M|)

数值示例：
  假设有 70 个被遮蔽位置，L_struct = 210.0
  平均每个残基的 MSE = 210.0 / 70 = 3.0 Å²
  RMSD = sqrt(3.0) ≈ 1.73 Å
  
  这个结果说明模型预测的骨架与真实结构平均偏差约 1.73 Å，
  在蛋白质设计中是一个合理的结果。
```

**典型的损失值参考**

```
训练初期（随机初始化）：
  平均每个残基 MSE ≈ 50-200 Å²（RMSD ≈ 7-14 Å）
  蛋白质的平均直径约 30-50 Å，所以随机猜测的误差约为蛋白质大小的 1/3

训练中期：
  平均每个残基 MSE ≈ 5-20 Å²（RMSD ≈ 2.2-4.5 Å）
  模型开始学到局部结构模式

训练后期（收敛）：
  平均每个残基 MSE ≈ 1-5 Å²（RMSD ≈ 1.0-2.2 Å）
  好的模型应该达到 RMSD < 2.5 Å

参考：AlphaFold2 在已知结构上的预测 RMSD 通常 < 1.0 Å，
但 EnzyGen2 的任务更难（从头设计而非预测已知蛋白质的结构），
所以 2-3 Å 的 RMSD 已经是非常好的结果。
```

**为什么只在被遮蔽位置计算**

```
Motif 位置的坐标是固定的（直接复制输入坐标到输出），
对这些位置计算 MSE 没有意义（永远为零）。

只对 scaffold（被遮蔽）位置计算结构损失，
因为只有这些位置的坐标是模型实际预测的。
这确保了损失信号完全反映模型的设计能力。
```

### 24.6 为什么结构损失不需要 Kabsch 对齐

在结构比较中，通常需要先做 Kabsch 对齐（刚性对齐），消除平移和旋转带来的误差。为什么 EnzyGen2 的结构损失可以直接用 MSE 而不需要对齐？

**核心原因：Motif 作为空间锚点**

```
EnzyGen2 的工作范式是 Motif-Scaffolding：
  1. Motif 位置的坐标在整个过程中保持不变
  2. 所有坐标已经以 Motif 中心为原点进行了中心化
  3. Scaffold 坐标是相对于 Motif 预测的

因为 Motif 坐标是固定的，预测坐标和真实坐标已经在同一个参考系中：
  - 不存在平移问题（都以 Motif 中心为原点）
  - 不存在旋转问题（Motif 定义了空间方向）
  - 因此可以直接逐原子计算 MSE
```

**与 DISCO 的对比**

```
DISCO（扩散模型）需要 Kabsch 对齐的原因：
  1. DISCO 从纯噪声开始生成坐标
  2. 每一步去噪都可能产生任意的旋转/平移
  3. 生成的坐标和真实坐标可能在完全不同的参考系中
  4. 必须先做刚性对齐，才能公平地比较两组坐标

EnzyGen2 不需要对齐的原因：
  1. Motif 坐标作为固定锚点，定义了唯一的参考系
  2. Scaffold 坐标从随机游走初始化，但 EGNN 更新是相对于 Motif 的
  3. 预测和真实坐标自然在同一参考系中
  4. 直接 MSE 就是真实的结构误差
```

**数学直觉**

```
如果两组坐标需要对齐：
  min_{R,t} Σ ||R·x̂_i + t - x_i||²    (需要求解旋转矩阵 R 和平移向量 t)

如果不需要对齐（EnzyGen2 的情况）：
  Σ ||x̂_i - x_i||²                     (直接计算，简单高效)

不需要对齐的好处：
  1. 计算速度更快（省去 SVD 分解步骤）
  2. 梯度计算更简单（MSE 的梯度是简单的线性形式）
  3. 训练更稳定（对齐步骤的梯度可能不稳定）
```

---

## 25. 配体结合预测损失（Binding Prediction Loss）

### 25.1 数学定义

```
L_bind = -1/B × Σ_{b=1}^{B} log P(y_b | protein_b, ligand_b)

其中：
  y_b ∈ {0, 1}：蛋白质 b 是否结合配体（二分类）
  P(y_b | ...) = 模型预测的结合概率
```

### 25.2 代码实现

```python
# 来自 geometric_protein_ncbi_ligand_loss.py (line 87-90)

if 'ligand_input' in sample:
    labels = ligand_input["ligand_binding"]  # [B]，0或1
    loss_binding = torch.mean(
        -torch.log(scores.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1))
    )
    # scores: [B, 2]（结合/不结合的概率）
    # gather: 取出真实标签对应的概率 [B]
    # -log: 负对数似然
    # mean: batch 平均
else:
    loss_binding = 0  # 没有配体时不计算
```

### 25.3 结合预测损失的数值示例

结合预测损失本质上是标准的二元交叉熵（Binary Cross-Entropy），只是实现方式和序列恢复损失相同（用 gather 取值而非完整交叉熵）。

**场景 1：模型对"结合"的预测比较有信心**

```
模型输出 scores: [B=4, 2]
  蛋白质 0: [P(不结合)=0.3,  P(结合)=0.7]   真实标签: 1 (结合)
  蛋白质 1: [P(不结合)=0.8,  P(结合)=0.2]   真实标签: 0 (不结合)
  蛋白质 2: [P(不结合)=0.1,  P(结合)=0.9]   真实标签: 1 (结合)
  蛋白质 3: [P(不结合)=0.6,  P(结合)=0.4]   真实标签: 1 (结合)

逐样本计算：
  蛋白质 0: gather 取 index=1 → P=0.7,  loss = -log(0.7)  = 0.357
  蛋白质 1: gather 取 index=0 → P=0.8,  loss = -log(0.8)  = 0.223
  蛋白质 2: gather 取 index=1 → P=0.9,  loss = -log(0.9)  = 0.105
  蛋白质 3: gather 取 index=1 → P=0.4,  loss = -log(0.4)  = 0.916

batch 平均损失 = (0.357 + 0.223 + 0.105 + 0.916) / 4 = 0.400
```

**场景 2：模型非常确信**

```
  P(结合) = 0.99, 真实标签 = 1
  loss = -log(0.99) = 0.01

  P(不结合) = 0.99, 真实标签 = 0
  loss = -log(0.99) = 0.01

模型非常有信心且正确时，损失接近零。
```

**场景 3：模型完全错误**

```
  P(结合) = 0.01, 真实标签 = 1（应该结合但模型说不结合）
  loss = -log(0.01) = 4.605

  如果 P(结合) = 0.0（极端情况）：
  loss = -log(0.0) = +∞   ← 无穷大！

这就是为什么代码中有 epsilon 保护。
```

**epsilon (1e-6) 的作用**

```python
# 实际代码中，scores 经过 softmax 后理论上在 (0, 1) 之间
# 但浮点数精度可能导致恰好为 0
# 如果 log(0) = -inf，整个损失会变成 NaN，训练崩溃

# 解决方案：在 softmax 之前或之后加入 epsilon
scores = F.softmax(combined_features, dim=-1)  # [B, 2]
# softmax 的输出理论上永远 > 0，但 float32 精度下
# 如果输入 logit 差值 > 88，较小值的 exp 会下溢到 0

# 因此实际使用时常见的做法：
loss = -torch.log(scores.gather(...) + 1e-6)
# 或在 softmax 前使用 log_softmax

# 加入 epsilon=1e-6 后：
# log(0 + 1e-6) = log(1e-6) = -13.8（有限值，不是 -inf）
# 这保证了训练的数值稳定性
```

**与序列恢复损失的类比**

```
序列恢复损失：
  从 [B, L, 33] 的概率分布中，按 src_tokens 索引取值 → 33 分类问题
  每个位置独立计算 → 一条序列有多个损失值

结合预测损失：
  从 [B, 2] 的概率分布中，按 labels 索引取值 → 2 分类问题
  每条蛋白质只有一个损失值

两者用完全相同的 gather + (-log) 范式，
区别仅在于分类数（33 vs 2）和是否逐位置计算。
```

### 25.4 为什么用 softmax 而不是 sigmoid

在 EnzyGen2 的结合预测中，模型输出 **2 个 logit**（bind/no-bind），然后通过 **softmax** 将它们转化为概率。这看起来似乎有些奇怪——为什么不直接用 1 个 logit + sigmoid，像传统二分类那样？

**数学等价性**

```
方法 1：Softmax（EnzyGen2 实际采用）
  输入：2 个 logit [z₀, z₁]
  P(bind)    = exp(z₁) / (exp(z₀) + exp(z₁))
  P(no_bind) = exp(z₀) / (exp(z₀) + exp(z₁))

方法 2：Sigmoid（传统二分类）
  输入：1 个 logit z
  P(bind)    = 1 / (1 + exp(-z))
  P(no_bind) = 1 - P(bind) = exp(-z) / (1 + exp(-z))

两者的等价关系：
  令 z = z₁ - z₀（两个 logit 的差值），则：
  P(bind) = exp(z₁) / (exp(z₀) + exp(z₁))
           = 1 / (1 + exp(z₀ - z₁))
           = 1 / (1 + exp(-z))
           = σ(z)

结论：2-logit softmax 和 1-logit sigmoid 在数学上完全等价。
```

**数值验证**

```
假设模型输出 logits = [1.0, 3.0]

Softmax 方式：
  exp(1.0) = 2.718
  exp(3.0) = 20.086
  P(no_bind) = 2.718 / (2.718 + 20.086) = 2.718 / 22.804 = 0.119
  P(bind)    = 20.086 / 22.804 = 0.881

Sigmoid 方式（用差值 z = 3.0 - 1.0 = 2.0）：
  P(bind) = 1 / (1 + exp(-2.0)) = 1 / (1 + 0.135) = 1 / 1.135 = 0.881

完全一致！两种方法得到相同的概率值。
```

**那为什么 EnzyGen2 选择 softmax？有四个实际原因：**

```
1. 与序列预测头的架构一致性
   序列预测头：33 个 logit → softmax → 33 类概率
   结合预测头：2 个 logit  → softmax → 2 类概率
   
   使用相同的 softmax + gather + (-log) 范式：
     序列损失 = -log(softmax(logits)[B, L, 33].gather(target))
     结合损失 = -log(softmax(logits)[B, 2].gather(target))
   代码复用最大化，减少 bug 概率。

2. 自然扩展到多分类
   如果未来需要区分"强结合"、"弱结合"、"不结合"（3 类），
   只需将输出从 2 改为 3，代码完全不变。
   如果用 sigmoid，扩展到多分类需要重写整个预测头。

3. 输出解释更清晰
   softmax 输出两个概率，且保证：
     P(bind) + P(no_bind) = 1.0（精确为 1）
   
   sigmoid 输出一个概率：
     P(bind) 和 1 - P(bind)
   虽然也满足和为 1，但在代码中需要手动计算 P(no_bind)。
   
   两个显式概率更方便调试和日志记录：
     print(f"bind={scores[0,1]:.3f}, no_bind={scores[0,0]:.3f}")

4. 数值稳定性的微妙差异
   PyTorch 的 log_softmax 内部使用了 LogSumExp trick：
     log_softmax(z_i) = z_i - log(Σ exp(z_j))
   
   这比 log(sigmoid(z)) 在极端值下更稳定。
   例如当 z₁ = 100.0, z₀ = -100.0 时：
     log_softmax: 利用 max 减去技巧，避免 exp 溢出
     log(sigmoid): log(1/(1+exp(-200))) → 可能遇到精度问题
   
   虽然 PyTorch 的 logsigmoid 也有类似保护，
   但 log_softmax 的数值稳定性是经过更广泛验证的。
```

**一个常见误解的澄清**

```
误解："用 2 个 logit 意味着模型有更多参数，所以更好"

事实：参数量的差异微不足道。
  线性层 W 的维度：
    softmax：  hidden_dim × 2（例如 1280 × 2 = 2560 个参数）
    sigmoid：  hidden_dim × 1（例如 1280 × 1 = 1280 个参数）
  
  差异仅 1280 个参数，相对于模型总参数量（~650M）完全可以忽略。
  选择 softmax 纯粹是为了架构一致性和可扩展性，不是为了增加参数。
```

---

## 26. 总损失函数与权重平衡

### 26.1 总损失公式

```
L_total = α × L_seq + β × L_struct + γ × L_bind

默认权重：
  α (encoder_factor) = 1.0    ← 序列恢复是主要目标
  β (decoder_factor) = 0.01   ← 结构预测是辅助约束
  γ (binding_factor) = 0.5    ← 结合预测（仅 Stage 3）

为什么结构损失权重这么小（0.01）？
  因为坐标 MSE 的数值远大于序列 NLL
  未归一化的坐标差值平方可能在 1~100 Å² 量级
  而序列 NLL 通常在 0~5 bits 量级
  0.01 的权重起到了数值对齐的作用
```

### 26.2 不同训练阶段的损失配置

```
Stage 1 (MLM):
  L = 1.0 × L_seq + 0.01 × L_struct
  criterion = geometric_protein_ncbi_loss

Stage 2 (Motif):
  L = 1.0 × L_seq + 0.01 × L_struct
  criterion = geometric_protein_ncbi_loss（同上，但遮蔽策略不同）

Stage 3 (Full):
  L = 1.0 × L_seq + 0.01 × L_struct + 0.5 × L_bind
  criterion = geometric_protein_ncbi_substrate_loss

Finetuning:
  L = 1.0 × L_seq + 0.01 × L_struct
  criterion = geometric_protein_ncbi_loss（回到无配体损失）
```

### 26.3 为什么三阶段使用不同的损失权重

三阶段训练的损失配置不是随意选择的，而是遵循**课程学习（Curriculum Learning）**的原则：先学简单任务，再逐步增加难度。

**Stage 1（MLM）：只有序列 + 结构损失**

```
L = 1.0 × L_seq + 0.01 × L_struct
没有 L_bind（binding_factor 不存在）

为什么不一开始就加入结合预测？
  1. 模型初始状态是 ESM2 预训练权重 + 随机初始化的 EGNN
  2. 此时模型还不理解蛋白质的三维结构（EGNN 还没学好）
  3. 如果同时加入三个目标，梯度信号会互相冲突：
     - 序列损失让模型学习氨基酸之间的共现规律
     - 结构损失让模型学习空间折叠规则
     - 结合损失要求模型理解蛋白质-配体界面
  4. 前两个目标是后一个的基础——不理解序列和结构，
     就不可能理解结合
  5. 此阶段只用了最基本的两个损失，让模型先建立
     蛋白质语言和几何的基础能力

数值示例：
  训练初期典型损失值：
    L_seq ≈ 200-300（100个位置 × 每个位置 NLL≈2-3）
    L_struct ≈ 5000-10000（100个位置 × 每个位置 MSE≈50-100 Å²）
    加权后：1.0 × 250 + 0.01 × 7500 = 250 + 75 = 325
    两个损失项大致平衡（量级相近）
```

**Stage 2（Motif）：同样的损失权重，不同的遮蔽策略**

```
L = 1.0 × L_seq + 0.01 × L_struct（与 Stage 1 完全相同）

权重不变，但任务本质改变了：
  Stage 1: 随机遮蔽 80% → 模型从任意 20% 残基恢复其余
  Stage 2: Motif 遮蔽 → 模型从功能位点恢复框架

为什么不改变损失权重？
  1. 损失的数值量级没有显著变化（还是 ~100 个位置的序列和结构损失）
  2. 改变的是模型要解决的问题类型，不是问题的难度量级
  3. 保持权重一致减少了超参数调优的复杂性

实际效果：
  Stage 2 的序列损失通常比 Stage 1 低（因为从 Stage 1 的权重开始），
  但结构损失可能更高（Motif 遮蔽模式下，scaffold 区域的结构预测更难，
  因为被遮蔽的位置可能远离 Motif 锚点）。
```

**Stage 3（Full）：新增结合预测损失**

```
L = 1.0 × L_seq + 0.01 × L_struct + 0.5 × L_bind

为什么 binding_factor = 0.5？
  1. 结合预测是二分类任务，loss 值通常在 0-1 之间
     （相比之下，序列损失可能在 50-200 之间）
  2. 0.5 的权重使得结合损失的贡献约为序列损失的 0.25-1%
  3. 这确保了结合预测是一个"轻约束"——引导方向但不主导训练

为什么此时学习率从 1e-4 降到 5e-5？
  1. 前两个阶段已经建立了良好的序列-结构表示
  2. 新增的 SubstrateEGNN 模块（随机初始化）需要稳定训练
  3. 如果学习率太大，新模块的梯度可能破坏已有的表示
  4. 较小的学习率让模型在保持已学知识的同时，缓慢适应新目标

为什么梯度裁剪从 1.5 降到 0.0001？
  1. SubstrateEGNN 的坐标更新涉及原子间距离计算
  2. 当原子距离很小时，梯度可能爆炸（距离的倒数很大）
  3. 极小的裁剪阈值（0.0001）确保每一步更新都非常保守
  4. 这是一个"安全优先"的设计——宁可训练慢，也不要训练崩溃
```

**Finetuning：去掉结合预测，回到基础损失**

```
L = 1.0 × L_seq + 0.01 × L_struct

为什么移除 L_bind？
  1. 微调数据是特定酶家族，所有样本都结合同一个底物
  2. 结合预测的二分类标签全是 1（都结合），没有负样本
  3. 全正样本的二分类损失没有信息量（模型只需要总是预测"结合"）
  4. 更重要的是：配体结合的知识已经在 Stage 3 中内化到模型权重中
     微调时不需要显式的结合损失来约束

整体课程学习策略总结：
  Stage 1: 学习蛋白质语言和基本结构 → "学会读和写蛋白质"
  Stage 2: 学习功能位点引导的设计   → "学会围绕活性位点设计框架"
  Stage 3: 学习配体结合约束         → "学会考虑底物的需求"
  Finetune: 专精于特定酶家族        → "成为某一领域的专家"
```

### 26.4 损失值的典型范围

理解损失值的正常范围对于判断训练状态至关重要。下面给出每种损失在训练不同阶段的典型数值范围，以及异常值的诊断方法。

**L_seq（序列恢复损失）的典型范围**

```
L_seq 是按被遮蔽位置平均的负对数似然（NLL per masked position）。

理论下界：0（模型完美预测每个氨基酸）
理论上界：-log(1/33) ≈ 3.50（均匀分布猜测 33 类）
随机初始化：≈ 3.50（刚开始训练，预测接近均匀分布）

训练过程中的典型值：
  ┌──────────────────┬──────────────┬──────────────────────────────────┐
  │ 训练阶段         │ 典型 L_seq   │ 含义                             │
  ├──────────────────┼──────────────┼──────────────────────────────────┤
  │ Stage 1 初期     │ 2.5 - 3.5    │ 模型刚开始学习，接近随机          │
  │ Stage 1 中期     │ 1.5 - 2.5    │ 模型学会了常见氨基酸模式          │
  │ Stage 1 收敛     │ 0.8 - 1.5    │ 模型掌握了蛋白质序列的统计规律    │
  │ Stage 2 初期     │ 1.0 - 1.8    │ 遮蔽策略改变，短暂升高            │
  │ Stage 2 收敛     │ 0.7 - 1.2    │ 模型学会了从 Motif 推断 scaffold  │
  │ Stage 3 收敛     │ 0.6 - 1.0    │ 配体信息提供了额外约束            │
  │ Finetuning 收敛  │ 0.3 - 0.8    │ 特定酶家族的序列空间更小          │
  └──────────────────┴──────────────┴──────────────────────────────────┘

数值示例：
  假设一个 200 残基的蛋白质，80% 被遮蔽（160 个位置）
  L_seq = 1.2 表示每个被遮蔽位置的平均 NLL 为 1.2
  这意味着模型对正确氨基酸的平均预测概率为 exp(-1.2) ≈ 0.30（30%）
  对比随机猜测的概率 1/33 ≈ 0.03（3%），模型好了 10 倍

异常值诊断：
  L_seq < 0.3 → 过拟合警告！模型可能在记忆训练集
    验证方法：检查验证集损失是否显著高于训练集
    常见原因：训练数据太少、学习率太高、训练轮次太多
  
  L_seq > 3.0 → 模型没有学到东西
    常见原因：学习率太低、数据加载错误、遮蔽比例太高
  
  L_seq 震荡不下降 → 学习率可能太大
    解决方案：减小学习率或增大 warmup 步数
```

**L_struct（结构预测损失）的典型范围**

```
L_struct 是每个被遮蔽位置的 Cα 坐标均方误差（MSE in Å²）。

理论下界：0（完美预测每个原子位置）
随机初始化：50 - 200 Å²（取决于蛋白质大小和坐标范围）

训练过程中的典型值：
  ┌──────────────────┬────────────────┬──────────────────────────────────┐
  │ 训练阶段         │ 典型 L_struct  │ 物理含义                         │
  ├──────────────────┼────────────────┼──────────────────────────────────┤
  │ 初始化           │ 50 - 200       │ 随机坐标，远离真实位置            │
  │ Stage 1 收敛     │ 2.0 - 5.0      │ 平均偏差 √MSE ≈ 1.4-2.2 Å       │
  │ Stage 2 收敛     │ 1.0 - 3.0      │ 平均偏差 √MSE ≈ 1.0-1.7 Å       │
  │ Stage 3 收敛     │ 1.0 - 3.0      │ 配体约束帮助活性位点附近          │
  │ Finetuning 收敛  │ 0.5 - 2.0      │ 特定酶家族结构更一致              │
  └──────────────────┴────────────────┴──────────────────────────────────┘

将 MSE 转换为物理距离：
  L_struct = 2.0 Å²
  √2.0 = 1.41 Å（RMSD）
  
  参考标准：
    同源蛋白质骨架 RMSD 通常 < 1.0 Å
    X-ray 结构解析的不确定性 ≈ 0.1-0.3 Å
    1.5 Å RMSD 在计算蛋白质设计中被认为是"良好"
    3.0 Å RMSD 是"可接受"的上限

注意：乘以 decoder_factor=0.01 后的加权值
  如果 L_struct = 3.0，加权后 = 0.01 × 3.0 = 0.03
  而 L_seq ≈ 1.0，加权后 = 1.0 × 1.0 = 1.0
  结构损失对总损失的贡献约为 3%
  这是故意设计的——结构是"软约束"，序列恢复是主目标
```

**L_bind（结合预测损失）的典型范围**

```
L_bind 是标准二元交叉熵（Binary Cross-Entropy）。

理论下界：0（完美分类）
随机猜测：-log(0.5) = 0.693
理论上界：无穷大（但实际中不超过 5-10）

训练过程中的典型值（仅 Stage 3）：
  ┌──────────────────┬──────────────┬──────────────────────────────────┐
  │ 训练阶段         │ 典型 L_bind  │ 含义                             │
  ├──────────────────┼──────────────┼──────────────────────────────────┤
  │ Stage 3 初期     │ 0.6 - 0.9    │ 接近随机猜测（0.693）            │
  │ Stage 3 中期     │ 0.3 - 0.6    │ 模型开始区分结合/不结合          │
  │ Stage 3 收敛     │ 0.1 - 0.4    │ 模型能较准确地预测结合           │
  └──────────────────┴──────────────┴──────────────────────────────────┘

数值含义：
  L_bind = 0.3：模型对正确类别的平均预测概率 = exp(-0.3) ≈ 0.74
    即 74% 的信心判断正确类别——分类准确率约 80-85%
  
  L_bind = 0.1：exp(-0.1) ≈ 0.90
    90% 的信心——分类准确率约 90-95%
  
  L_bind = 0.7：exp(-0.7) ≈ 0.50
    50% 的信心——基本等于随机猜测

乘以 binding_factor=0.5 后：
  如果 L_bind = 0.4，加权后 = 0.5 × 0.4 = 0.2
  相对于 L_seq ≈ 1.0，结合损失贡献约 20%
  这是一个适中的权重——足以引导模型但不会主导优化方向
```

**总损失的典型值与诊断表**

```
Stage 3 收敛时的典型总损失计算：
  L_total = 1.0 × L_seq + 0.01 × L_struct + 0.5 × L_bind
          = 1.0 × 0.8   + 0.01 × 2.0      + 0.5 × 0.3
          = 0.80         + 0.02             + 0.15
          = 0.97

各项贡献比例：
  序列损失：0.80 / 0.97 = 82.5%（主导）
  结构损失：0.02 / 0.97 = 2.1%（最小）
  结合损失：0.15 / 0.97 = 15.5%（辅助）

快速诊断清单：
  ┌─────────────────────────────────┬──────────────────────────────────┐
  │ 症状                           │ 可能原因与解决方案                │
  ├─────────────────────────────────┼──────────────────────────────────┤
  │ L_seq 持续 > 3.0              │ 数据问题或学习率太低              │
  │ L_seq < 0.3（训练集）          │ 过拟合，需正则化或更多数据        │
  │ L_struct > 50                  │ EGNN 没有有效学习，检查交错层     │
  │ L_bind 不下降（保持 ~0.69）    │ SubstrateEGNN 梯度消失           │
  │ 总损失突然跳升                 │ 学习率太大或梯度裁剪不足          │
  │ 训练/验证损失差距 > 0.5        │ 过拟合，减小模型或增加数据        │
  └─────────────────────────────────┴──────────────────────────────────┘
```

---

## 27. JSON 数据格式详解

### 27.1 预训练数据结构

```json
{
  "10665": {                              // NCBI Taxonomy ID（物种标识）
    "train": {
      "seq": [                            // 蛋白质序列列表
        "MTVFLKFNA...",
        "MVPSNKD..."
      ],
      "coor": [                           // 坐标（逗号分隔的浮点数）
        "1.2,3.4,5.6,7.8,9.0,1.1,...",   // 每3个数 = 一个残基的 (x,y,z)
        "2.1,4.3,6.5,..."
      ],
      "motif": [                          // Motif 残基索引
        "5,10,15-20,30",                  // 逗号分隔，支持范围
        "3,7,12-18"
      ],
      "protein_id": [                     // PDB/UniProt ID
        "3DWZ_A",
        "1ABC_B"
      ],
      "ligand_feat": [                    // 配体原子特征（嵌套列表）
        [[[1,0,0,1,0], [0,1,0,0,1], ...]],
        ...
      ],
      "ligand_coor": [                    // 配体原子坐标
        "1.1,2.2,3.3,4.4,5.5,6.6,...",
        ...
      ],
      "ligand_binding": [0, 1, ...]       // 结合标签（0=不结合，1=结合）
    },
    "valid": { ... },
    "test": { ... }
  },
  "11698": { ... },
  "9796": { ... },
  ...
}
```

### 27.2 微调数据结构

微调数据结构类似，但按反应 ID 组织：

```json
{
  "18421": {                              // Rhea 反应 ID
    "train": {
      "seq": [...],
      "coor": [...],
      "motif": [...],
      "protein_id": [...]
      // 微调数据通常不包含 ligand 信息
    },
    "valid": { ... },
    "test": { ... }
  }
}
```

### 27.3 坐标格式详解

```
坐标字符串："1.2,3.4,5.6,7.8,9.0,1.1,..."

解析方式（来自 indexed_dataset.py, line 244-248）：
  coors = line.split(",")
  protein_coor = []
  for i in range(0, len(coors), 3):
      protein_coor.append([
          float(coors[i]),      # x
          float(coors[i+1]),    # y  
          float(coors[i+2])     # z
      ])
  
  结果：[[1.2, 3.4, 5.6], [7.8, 9.0, 1.1], ...]
  每个三元组 = 一个 Cα 原子的 (x, y, z) 坐标
```

### 27.4 预训练 vs 微调数据的关键差异

预训练数据和微调数据虽然使用相同的 JSON 格式，但在内容和语义上存在根本性差异。理解这些差异对正确准备自己的数据至关重要。

**差异 1：遮蔽策略的语义不同**

```
预训练数据中的 motif 字段 → 定义了哪些位置是功能位点
  - 在 Stage 1 (MLM) 中，这个字段被忽略，取而代之的是随机遮蔽
  - 在 Stage 2 (Motif) 中，这个字段才被使用
  - motif 位置在遮蔽时被设为 0（不遮蔽/保留）

微调数据中的 motif 字段 → 定义了活性位点残基
  - 这些残基永远不会被遮蔽（它们是设计的约束条件）
  - 模型需要围绕这些固定的残基设计 scaffold
```

**差异 2：配体信息**

```
预训练数据（Stage 3 使用）：
  包含 ligand_feat、ligand_coor、ligand_binding 三个字段
  这些信息被 SubstrateEGNN 和结合预测模块使用

微调数据：
  通常不包含 ligand 相关字段
  微调使用的是基础模型（NCBIModel），不需要配体信息
  配体知识已经在 Stage 3 中内化到共享的模型权重中
```

**差异 3：数据组织方式**

```
预训练数据按 NCBI Taxonomy ID（物种）组织：
  key = "562"（E. coli）、"9606"（人类）等
  同一物种下包含多种不同的蛋白质

微调数据按 Rhea 反应 ID 组织：
  key = "18421"（ChlR 酶催化的反应）等
  同一反应下包含执行相同功能的不同蛋白质
```

**具体 JSON 示例对比**

预训练数据（包含配体，多物种）：

```json
{
  "562": {
    "train": {
      "seq": [
        "MKFLAVILG",
        "AQVINTFDG"
      ],
      "coor": [
        "10.2,5.3,3.1,13.8,6.1,4.5,17.2,7.0,5.8,...",
        "8.5,12.1,3.7,11.9,13.2,4.0,..."
      ],
      "motif": [
        "2,3,7",
        "1,4,5"
      ],
      "protein_id": ["3DWZ_A", "1XYZ_B"],
      "ligand_feat": [
        [[[6, 2, 0, 1, 0], [8, 1, -1, 0, 2], [7, 3, 0, 1, 1]]],
        [[[6, 1, 0, 0, 3], [8, 2, 0, 0, 1]]]
      ],
      "ligand_coor": [
        "1.5,2.3,4.1,3.2,5.1,2.8,4.7,3.9,6.2",
        "2.1,3.0,5.5,4.8,2.7,3.3"
      ],
      "ligand_binding": [1, 0]
    },
    "valid": { "seq": [...], "coor": [...], ... }
  },
  "9606": {
    "train": { ... },
    "valid": { ... }
  }
}
```

微调数据（无配体，按反应组织）：

```json
{
  "18421": {
    "train": {
      "seq": [
        "MSTLYDIRFAG",
        "MATLYEIRFAG"
      ],
      "coor": [
        "5.1,8.3,2.0,8.7,9.4,3.1,...",
        "5.3,8.1,2.2,8.9,9.2,3.3,..."
      ],
      "motif": [
        "3,5,6,8",
        "3,5,6,8"
      ],
      "protein_id": ["ChlR_wt", "ChlR_v1"]
    },
    "valid": { ... },
    "test": { ... }
  }
}
```

**关键区别总结表**

| 特征 | 预训练数据 | 微调数据 |
|------|-----------|---------|
| 组织方式 | 按物种 (NCBI ID) | 按反应 (Rhea ID) |
| motif 含义 | 功能位点标注 | 活性位点约束 |
| 遮蔽方式 | MLM阶段随机 / Motif阶段按标注 | 始终按标注 |
| 配体数据 | 有 (Stage 3) | 无 |
| 蛋白质多样性 | 高（多物种多功能） | 低（同一酶家族） |
| 数据量 | 数万条蛋白质 | 数十到数百条 |

---

## 28. IndexedRawTextDataset：序列加载

### 28.1 数据加载逻辑

```python
# 来自 indexed_dataset.py (line 115-171)
class IndexedRawTextDataset(FairseqDataset):
    def read_data(self, path, dictionary, split, protein, data_stage):
        data = json.load(open(path))
        
        if data_stage != "finetuning":
            if split == "train":
                for protein in data:           # 遍历所有物种
                    lines.extend(data[protein][split]["seq"])
            elif split == "valid":
                for protein in data:
                    if split in data[protein]:
                        lines.extend(data[protein][split]["seq"])
            else:  # test
                lines = data[protein][split]["seq"]  # 特定物种
        else:
            # finetuning 模式
            ...
        
        for line in lines:
            tokens = dictionary.encode_line(
                line,
                prepend_bos=True,      # 加 <cls>
                append_eos=True,       # 加 <eos>
            ).long()
            self.tokens_list.append(tokens)
            self.sizes.append(len(tokens))
```

### 28.2 编码过程

```
原始序列：    "MKVAVL"（6个氨基酸）
encode_line → [0, 20, 15, 7, 5, 7, 10, 2]
              ↑BOS  M   K   V   A   V   L  ↑EOS
sizes = 8（序列长度 + 2）
```

### 28.3 序列编码的完整数值追踪

下面用一个具体的蛋白质序列，展示从原始字符串到模型输入 token 的完整编码过程。

**ESM-1b/ESM2 氨基酸词表索引**

```
ESM2 使用的 Alphabet 共 33 个 token，索引如下：

  索引 0:  <cls>  (BOS，句首标记)
  索引 1:  <pad>  (填充标记)
  索引 2:  <eos>  (EOS，句尾标记)
  索引 3:  <unk>  (未知标记)
  索引 4:  L (亮氨酸)      索引 14: M (蛋氨酸)
  索引 5:  A (丙氨酸)      索引 15: W (色氨酸)
  索引 6:  G (甘氨酸)      索引 16: K (赖氨酸)
  索引 7:  V (缬氨酸)      索引 17: Q (谷氨酰胺)
  索引 8:  S (丝氨酸)      索引 18: E (谷氨酸)
  索引 9:  E (谷氨酸)      索引 19: S (丝氨酸)
  索引 10: R (精氨酸)      索引 20: P (脯氨酸)
  索引 11: T (苏氨酸)      索引 21: H (组氨酸)
  索引 12: I (异亮氨酸)    索引 22: D (天冬氨酸)
  索引 13: D (天冬氨酸)    索引 23: N (天冬酰胺)
  
  索引 24-32: 非标准 token
    24: B (天冬氨酸/天冬酰胺)
    25: U (硒代半胱氨酸)
    26: Z (谷氨酸/谷氨酰胺)
    27: O (吡咯赖氨酸)
    28: X (任意氨基酸)
    29: J
    30: <null_1>
    31: <mask>
    32: <cath> / <af2>

注意：ESM2 的词表顺序不是按照标准生物学顺序（ACDEFGHIKLMNPQRSTVWY），
而是按照频率/内部编码排列的。这个顺序在 esm_modules.py 的
prepend_toks 和 standard_toks 中定义。
```

**完整编码追踪**

```
Step 1：原始输入
  蛋白质序列字符串："MKVAV"（5个氨基酸）

Step 2：encode_line(prepend_bos=True, append_eos=True)
  
  字符 'M' → 查词表 → 索引 14
  字符 'K' → 查词表 → 索引 16
  字符 'V' → 查词表 → 索引  7
  字符 'A' → 查词表 → 索引  5
  字符 'V' → 查词表 → 索引  7
  
  加 BOS (索引 0) 在前，加 EOS (索引 2) 在后：
  tokens = [0, 14, 16, 7, 5, 7, 2]
            ↑    ↑   ↑  ↑  ↑  ↑  ↑
          <cls>  M   K  V  A  V  <eos>
  
  sizes = 7（5个氨基酸 + BOS + EOS = 7）

Step 3：padding（在 batch collation 阶段）
  假设 batch 中最长序列是 10 个 token
  padding 到长度 10，用 <pad> (索引 1) 填充：
  
  tokens = [0, 14, 16, 7, 5, 7, 2, 1, 1, 1]
            ↑                      ↑  ←padding→
          <cls>                  <eos>

Step 4：对应的 motif_mask
  假设位置 1（K）和 3（A）是 Motif 位点：
  motif_mask = [0, 0, 1, 0, 1, 0, 0, 0, 0, 0]
               ↑  ↑     ↑     ↑  ↑  ←padding→
             BOS  M(遮蔽) K(保留) V(遮蔽) A(保留) V(遮蔽) EOS

  mask=0 → 该位置已知（BOS/EOS/Motif）
  mask=1 → 该位置被遮蔽（需要模型预测）
```

**token dropout 的影响**

```
在 token dropout 之后（训练时随机将部分 token 替换为 <mask>）：

原始 tokens: [0, 14, 16, 7, 5, 7, 2, 1, 1, 1]
                  ↑       ↑
              被 dropout 的位置

dropout 后:  [0, 31, 16, 31, 5, 7, 2, 1, 1, 1]
                 ↑       ↑
               <mask>   <mask>

注意：只有被遮蔽（mask=1）的位置才会被替换为 <mask>
Motif 位置（mask=0）永远保留原始 token
```

---

## 29. CoordinateDataset：坐标加载与中心化

### 29.1 中心化处理

```python
# 来自 indexed_dataset.py (line 249-252)
mask = (motif_list[ind][1: -1] == 0).int().unsqueeze(-1)
# motif_list[ind]: [L+2]（包含BOS/EOS位置）
# [1:-1]: 去掉BOS/EOS → [L]
# == 0: Motif位置=0，非Motif位置=1
# 这里 mask=1 表示 Motif 位置（不被遮蔽的位置）

mean_coor = torch.sum(protein_coor * mask, dim=0) / mask.sum()
# 用 Motif（已知/固定）位置的坐标计算中心
protein_coor = protein_coor - mean_coor
# 以 Motif 中心为原点归一化所有坐标

# 添加 BOS/EOS 的 [0,0,0] padding
protein_coor = torch.cat([
    torch.tensor([[0, 0, 0]]),   # BOS
    protein_coor,                 # 中心化后的坐标
    torch.tensor([[0, 0, 0]])    # EOS
], dim=0)
```

### 29.2 为什么用 Motif 中心而不是全局中心

```
原因：Motif 位置是已知的、不会被修改的参考点
  - 用 Motif 中心归一化，保证 Motif 坐标在数值上接近原点
  - 模型只需要预测相对于 Motif 的位移
  - 推理时，将预测坐标加回中心即可恢复绝对坐标
```

### 29.3 坐标中心化的数值示例

下面用一个 5 残基蛋白质的例子，完整追踪坐标中心化的过程。

**设定**

```
5 残基蛋白质序列：M K V A L
Motif 位点：位置 1 (K) 和 位置 2 (V)（0-indexed）
motif_mask (原始): [1, 0, 0, 1, 1]
  0 = Motif（不遮蔽），1 = Scaffold（遮蔽/需要设计）

原始 Cα 坐标（单位：Å）：
  位置 0 (M): ( 6.5,  3.2,  1.8)   ← Scaffold
  位置 1 (K): (10.0,  5.0,  3.0)   ← Motif
  位置 2 (V): (13.5,  6.2,  4.1)   ← Motif
  位置 3 (A): (17.0,  7.5,  5.3)   ← Scaffold
  位置 4 (L): (20.2,  8.8,  6.0)   ← Scaffold
```

**Step 1：构建 Motif 坐标掩码**

```python
# motif_list[ind] = [0, 1, 0, 0, 1, 1, 0]  （包含 BOS 和 EOS）
#                    ↑BOS               ↑EOS
# [1:-1] = [1, 0, 0, 1, 1]  （去掉 BOS/EOS）
# == 0 → [False, True, True, False, False]
# .int() → [0, 1, 1, 0, 0]
# .unsqueeze(-1) → [[0], [1], [1], [0], [0]]  (形状: [5, 1])

# 注意反转：mask==0 的是 Motif 位置
# 取反后 mask=1 表示 Motif（用于计算中心的位置）
```

**Step 2：计算 Motif 中心**

```
只使用 Motif 位置（K 和 V）的坐标计算中心：

protein_coor * mask：
  位置 0 (M): (6.5, 3.2, 1.8) × 0 = (0.0, 0.0, 0.0)   ← 被忽略
  位置 1 (K): (10.0, 5.0, 3.0) × 1 = (10.0, 5.0, 3.0)  ← 参与计算
  位置 2 (V): (13.5, 6.2, 4.1) × 1 = (13.5, 6.2, 4.1)  ← 参与计算
  位置 3 (A): (17.0, 7.5, 5.3) × 0 = (0.0, 0.0, 0.0)   ← 被忽略
  位置 4 (L): (20.2, 8.8, 6.0) × 0 = (0.0, 0.0, 0.0)   ← 被忽略

sum(dim=0) = (10.0 + 13.5, 5.0 + 6.2, 3.0 + 4.1) = (23.5, 11.2, 7.1)
mask.sum() = 2（两个 Motif 位置）

mean_coor = (23.5/2, 11.2/2, 7.1/2) = (11.75, 5.6, 3.55)
```

**Step 3：中心化所有坐标**

```
所有坐标减去 Motif 中心 (11.75, 5.6, 3.55)：

  中心化前                    →   中心化后
  位置 0 (M): ( 6.5,  3.2,  1.8)  →  ( 6.5-11.75,  3.2-5.6,  1.8-3.55) = (-5.25, -2.40, -1.75)
  位置 1 (K): (10.0,  5.0,  3.0)  →  (10.0-11.75,  5.0-5.6,  3.0-3.55) = (-1.75, -0.60, -0.55)
  位置 2 (V): (13.5,  6.2,  4.1)  →  (13.5-11.75,  6.2-5.6,  4.1-3.55) = ( 1.75,  0.60,  0.55)
  位置 3 (A): (17.0,  7.5,  5.3)  →  (17.0-11.75,  7.5-5.6,  5.3-3.55) = ( 5.25,  1.90,  1.75)
  位置 4 (L): (20.2,  8.8,  6.0)  →  (20.2-11.75,  8.8-5.6,  6.0-3.55) = ( 8.45,  3.20,  2.45)
```

**Step 4：验证 Motif 中心在原点附近**

```
中心化后 Motif 位置的坐标：
  K: (-1.75, -0.60, -0.55)
  V: ( 1.75,  0.60,  0.55)

Motif 新中心 = ((-1.75+1.75)/2, (-0.60+0.60)/2, (-0.55+0.55)/2) = (0.0, 0.0, 0.0)

完美地以原点为中心。这保证了模型预测的 scaffold 坐标
在数值上都是相对于 Motif 中心的偏移量。
```

**Step 5：添加 BOS/EOS padding**

```
最终坐标张量（包含 BOS 和 EOS 的 [0,0,0]）：

  位置 0 (BOS): (0.00, 0.00, 0.00)   ← 特殊 token
  位置 1 (M):   (-5.25, -2.40, -1.75) ← Scaffold
  位置 2 (K):   (-1.75, -0.60, -0.55) ← Motif
  位置 3 (V):   ( 1.75,  0.60,  0.55) ← Motif
  位置 4 (A):   ( 5.25,  1.90,  1.75) ← Scaffold
  位置 5 (L):   ( 8.45,  3.20,  2.45) ← Scaffold
  位置 6 (EOS): (0.00, 0.00, 0.00)   ← 特殊 token

张量形状：[7, 3]（5 残基 + BOS + EOS = 7 个位置，每个 3 维坐标）
```

**推理时如何恢复绝对坐标**

```
模型输出预测坐标后，需要加回中心：
  pred_coords + center = absolute_coords

例如模型预测位置 0 (M) 的坐标为 (-5.10, -2.50, -1.80)（与真实值略有偏差）：
  绝对坐标 = (-5.10+11.75, -2.50+5.6, -1.80+3.55) = (6.65, 3.10, 1.75)
  真实绝对坐标 = (6.50, 3.20, 1.80)
  偏差 = sqrt((0.15)² + (-0.10)² + (-0.05)²) = sqrt(0.035) ≈ 0.19 Å
  非常准确！
```

### 29.4 为什么用 Motif 中心而不是全局中心的数学分析

这一节从数学角度深入解释为什么中心化必须以 Motif（活性位点）为参考，而不是以蛋白质的几何中心为参考。

**方案对比：两种中心化策略**

```
方案 A（全局中心，EnzyGen2 未采用）：
  center_global = mean(所有残基的坐标)
  centered_coords = all_coords - center_global

方案 B（Motif 中心，EnzyGen2 实际采用）：
  center_motif = mean(仅 Motif 残基的坐标)
  centered_coords = all_coords - center_motif
```

**用第 29.3 节的 5 残基例子做对比**

```
原始坐标：
  位置 0 (M): ( 6.5,  3.2,  1.8)   ← Scaffold
  位置 1 (K): (10.0,  5.0,  3.0)   ← Motif
  位置 2 (V): (13.5,  6.2,  4.1)   ← Motif
  位置 3 (A): (17.0,  7.5,  5.3)   ← Scaffold
  位置 4 (L): (20.2,  8.8,  6.0)   ← Scaffold

方案 A：全局中心
  center = mean(所有 5 个位置)
         = ((6.5+10+13.5+17+20.2)/5, (3.2+5+6.2+7.5+8.8)/5, (1.8+3+4.1+5.3+6)/5)
         = (67.2/5, 30.7/5, 20.2/5)
         = (13.44, 6.14, 4.04)
  
  中心化后：
    M: ( 6.5-13.44,  3.2-6.14,  1.8-4.04) = (-6.94, -2.94, -2.24)
    K: (10.0-13.44,  5.0-6.14,  3.0-4.04) = (-3.44, -1.14, -1.04)  ← Motif
    V: (13.5-13.44,  6.2-6.14,  4.1-4.04) = ( 0.06,  0.06,  0.06)  ← Motif
    A: (17.0-13.44,  7.5-6.14,  5.3-4.04) = ( 3.56,  1.36,  1.26)
    L: (20.2-13.44,  8.8-6.14,  6.0-4.04) = ( 6.76,  2.66,  1.96)

  Motif 的新中心 = ((-3.44+0.06)/2, (-1.14+0.06)/2, (-1.04+0.06)/2)
                  = (-1.69, -0.54, -0.49)
  ← 偏离原点！

方案 B：Motif 中心（实际采用）
  center = mean(位置 1, 2) = (11.75, 5.6, 3.55)
  
  中心化后：
    M: (-5.25, -2.40, -1.75)
    K: (-1.75, -0.60, -0.55)  ← Motif
    V: ( 1.75,  0.60,  0.55)  ← Motif
    A: ( 5.25,  1.90,  1.75)
    L: ( 8.45,  3.20,  2.45)

  Motif 的新中心 = ((-1.75+1.75)/2, (-0.60+0.60)/2, (-0.55+0.55)/2)
                  = (0.00, 0.00, 0.00)
  ← 精确在原点！
```

**问题 1：Scaffold 坐标会"污染"全局中心**

```
在训练时，scaffold 残基的坐标是已知的（来自真实结构）。
但在推理时，scaffold 残基的坐标是未知的——它们是模型需要预测的！

如果使用全局中心：
  推理时需要知道所有残基的坐标来计算中心，但 scaffold 坐标还没有预测出来。
  这是一个鸡生蛋问题：
    需要中心来初始化坐标 → 需要坐标来计算中心
  
  唯一的解决办法是用随机初始化的 scaffold 坐标来计算"伪全局中心"，
  但这个中心是不准确的，会随着每次随机初始化而变化。

如果使用 Motif 中心：
  推理时 Motif 残基的坐标是已知的（它们是输入的一部分，不需要预测）。
  中心计算完全确定，不受 scaffold 初始化的影响。
```

**问题 2：EGNN 的坐标更新精度与距离原点的关系**

```
EGNN 更新坐标的公式：
  x_i^(new) = x_i + Σ_j (x_i - x_j) · φ(m_ij)

其中 φ(m_ij) 是一个标量权重（通常在 -1 到 1 之间）。

关键观察：EGNN 的更新是基于"相对位移"的。
但数值精度在绝对坐标较大时会下降。

例如：
  float32 精度约 7 位有效数字。
  如果坐标值在原点附近（例如 x = 0.5 Å）：
    可以精确到 0.5000001 Å，精度 ≈ 10⁻⁷ Å
  
  如果坐标值远离原点（例如 x = 50.0 Å）：
    可以精确到 50.00001 Å，精度 ≈ 10⁻⁵ Å
    精度下降了 100 倍！

对于蛋白质设计，我们需要 0.01 Å 级别的精度来准确放置原子。
Motif 中心化确保了最关键的区域（活性位点）处于最高精度区域。

以第 29.3 节的例子为例：
  方案 B 中 Motif 坐标：K=(-1.75, -0.60, -0.55), V=(1.75, 0.60, 0.55)
    最大绝对值 = 1.75，精度 ≈ 10⁻⁷ Å ← 极高精度
  
  方案 A 中 Motif 坐标：K=(-3.44, -1.14, -1.04), V=(0.06, 0.06, 0.06)
    最大绝对值 = 3.44，精度略低
    但更重要的是，Motif 中心偏离了原点 (-1.69, -0.54, -0.49)
    模型需要额外"学习"这个偏移量
```

**问题 3：训练-推理一致性**

```
训练时：
  Motif 坐标已知，scaffold 坐标已知（来自真实结构）
  使用 Motif 中心 → Motif 坐标在原点附近 → 模型学到的 scaffold 位置
  是相对于 Motif 的偏移

推理时：
  Motif 坐标已知，scaffold 坐标需要预测
  使用 Motif 中心 → Motif 坐标同样在原点附近 → 模型预测的 scaffold 位置
  也是相对于 Motif 的偏移

一致性保证：训练和推理时，相同的 Motif 坐标经过中心化后
  得到完全相同的数值 → 模型的输入分布一致 → 预测更准确

如果使用全局中心：
  训练时：全局中心由所有真实坐标决定
  推理时：全局中心由真实 Motif + 随机 scaffold 决定
  → 中心不同 → Motif 输入坐标不同 → 分布偏移 → 预测质量下降
```

**总结：Motif 中心化的三大优势**

```
┌────────┬──────────────────────────────────┬──────────────────────────────────┐
│ 维度   │ Motif 中心（实际采用）           │ 全局中心（未采用）               │
├────────┼──────────────────────────────────┼──────────────────────────────────┤
│ 可计算 │ 推理时可精确计算（Motif 已知）   │ 推理时依赖随机初始化的 scaffold  │
│ 精度   │ 关键区域（活性位点）精度最高     │ 关键区域精度不确定               │
│ 一致性 │ 训练/推理分布完全一致            │ 训练/推理分布存在偏移            │
│ 物理   │ Motif 居中符合"围绕活性位点      │ 全局居中不反映设计目标           │
│ 意义   │ 设计框架"的设计理念              │                                 │
└────────┴──────────────────────────────────┴──────────────────────────────────┘
```

---

## 30. ProteinMotifDataset：遮蔽策略

### 30.1 两种遮蔽模式

```python
# 来自 indexed_dataset.py (line 326-344)

for line, size in zip(lines, dataset_sizes):
    mask = np.ones(size)  # 初始化全 1（全部遮蔽）
    indexes = line.strip().split(",")
    
    if data_stage == "pretraining-mlm":
        # MLM 模式：随机遮蔽 80% 的非 BOS/EOS 位置
        indexes = random.sample(
            list(range(1, size-1)),    # 排除 BOS(0) 和 EOS(size-1)
            int(0.8 * (size - 2))     # 80% 的残基
        )
    else:
        # Motif 模式：遮蔽 JSON 中指定的位置
        indexes = [int(index) + 1 for index in indexes]
        # +1 因为 BOS 占了位置 0
    
    for ind in indexes:
        mask[int(ind)] = 0  # 这些位置设为 0（不遮蔽/已知）
    mask[0] = 0   # BOS 位置不遮蔽
    mask[-1] = 0  # EOS 位置不遮蔽
    self.motif_list.append(torch.IntTensor(mask))
```

### 30.2 遮蔽掩码的含义

```
mask = 1 → 该位置被遮蔽（需要模型预测）
mask = 0 → 该位置已知（Motif或BOS/EOS）

MLM 模式：
  mask: [0, 1, 1, 0, 1, 1, 1, 0, 1, 0]  ← 80%的位置是1
        ↑BOS  ↑保留 ↑保留       ↑保留 ↑EOS
  → 模型看到 20% 的残基，预测 80%

Motif 模式：
  mask: [0, 1, 1, 0, 0, 0, 1, 1, 1, 0]
        ↑BOS  ↑设计  ↑Motif位点  ↑设计   ↑EOS
  → 模型保留 Motif 位点不变，设计其余部分
```

### 30.3 遮蔽策略的数值示例

下面用一个 10 残基蛋白质的例子，详细展示两种遮蔽模式的具体操作过程。

**设定**

```
蛋白质序列：M K V A L G S T E R（10 残基）
加上 BOS/EOS 后总长度 size = 12

token 表示：[<cls>, M, K, V, A, L, G, S, T, E, R, <eos>]
索引位置：  [  0,   1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 ]
```

**模式 1：随机遮蔽（pretraining-mlm）**

```
Step 1：初始化
  mask = np.ones(12) = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  （全部设为"遮蔽"）

Step 2：随机选择 80% 的非 BOS/EOS 位置作为"保留"位置
  可选位置：range(1, 11) = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]（共 10 个）
  选择数量：int(0.8 × 10) = 8 个位置
  
  假设随机选择了：[1, 3, 4, 5, 7, 8, 9, 10]
  
  注意：这里的逻辑容易混淆！
  代码将选中的位置 mask[ind] = 0（设为"已知/不遮蔽"）
  
  但这实际上是 MLM 模式的反直觉之处——
  在 MLM 模式下，代码选中 80% 的位置并设为 0（保留），
  剩余 20% 保持为 1（遮蔽）

Step 3：设置选中位置和 BOS/EOS
  for ind in [1, 3, 4, 5, 7, 8, 9, 10]:
      mask[ind] = 0
  mask[0] = 0   # BOS
  mask[11] = 0  # EOS
  
  最终 mask = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0]
               ↑  ↑     ↑  ↑  ↑     ↑  ↑  ↑  ↑  ↑
             BOS M  遮蔽 V  A  L  遮蔽 S  T  E  R EOS
  
  遮蔽位置：索引 2 (K) 和索引 6 (G)
  模型需要从上下文中预测这 2 个残基的序列和坐标

  注意：这意味着 MLM 模式下模型看到 80% 的残基（8/10），
  只需要预测 20%（2/10）。这看起来比较简单，
  但它对应的是 BERT 风格的预训练——
  通过大量的这种"完形填空"学习蛋白质语言。
```

**模式 2：Motif 指定遮蔽（pretraining-motif / finetuning）**

```
Step 1：初始化
  mask = np.ones(12) = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

Step 2：从 JSON 读取 motif 索引
  JSON 中的 motif 字段："2,3,7"
  表示原始蛋白质序列中位置 2, 3, 7 是活性位点残基
  
  加 1 偏移（因为 BOS 占了位置 0）：
  indexes = [2+1, 3+1, 7+1] = [3, 4, 8]

Step 3：设置 Motif 位置和 BOS/EOS
  for ind in [3, 4, 8]:
      mask[ind] = 0
  mask[0] = 0   # BOS
  mask[11] = 0  # EOS
  
  最终 mask = [0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0]
               ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑
             BOS M  K  V  A  L  G  S  T  E  R EOS
                 设 设 Mo Mo 设 设 设 Mo 设 设
                 计 计 tif tif 计 计 计 tif 计 计
  
  Motif 位置（mask=0）：V(3), A(4), T(8) — 活性位点，保持不变
  Scaffold 位置（mask=1）：M(1), K(2), L(5), G(6), S(7), E(9), R(10) — 需要设计
  
  模型需要设计 7/10 = 70% 的残基（围绕 3 个活性位点残基）
```

**两种模式的本质区别**

```
                MLM 模式                    Motif 模式
遮蔽比例     固定 ~20%（少量遮蔽）       取决于 Motif 大小（通常 50-80%）
遮蔽位置     随机选择                     由 JSON 文件固定指定
每次训练     遮蔽位置不同（随机性）       遮蔽位置相同（确定性）
学习目标     理解蛋白质语言模式           围绕功能位点设计框架
难度         较低（大量上下文可用）       较高（只有少数锚点）

注意一个反直觉的点：
  MLM 只遮蔽 20%（容易），但它的目标是学习通用的蛋白质表示
  Motif 遮蔽 50-80%（困难），但它的目标是学习设计能力
  这就是为什么 Stage 1 先用 MLM 建立基础，Stage 2 再转向 Motif
```

---

## 31. NCBITaxonomyDataset：物种分类

### 31.1 ID 映射

```python
# 来自 indexed_dataset.py (line 454-459)
class NCBITaxonomyDataset(FairseqDataset):
    def __init__(self, path, split, protein, data_stage):
        self.ncbi2id = json.load(open("data/ncbi2id.json", "r"))
        # ncbi2id 示例：{"562": 0, "9606": 1, "83332": 2, ...}
        # 将 NCBI Taxonomy ID 字符串映射到 [0, 9999] 的整数索引
```

### 31.2 微调时的物种回退

```python
# 来自 indexed_dataset.py (line 502-511)
for ncbi_item in ncbi:
    if ncbi_item not in self.ncbi2id:
        # 如果物种 ID 不在映射表中，使用默认值
        if reaction_task == "18421":     # ChlR
            ncbi_item = "562"            # → E. coli
        elif reaction_task == "20245":   # AadA
            ncbi_item = "99287"          # → Salmonella
        elif reaction_task == "Thiopurine_S_methyltransferas":  # TPMT
            ncbi_item = "9606"           # → Human
```

### 31.3 物种回退策略的具体逻辑

当微调数据中出现预训练阶段未见过的物种 NCBI ID 时，模型需要一个回退策略来处理。下面详细解释这个过程以及其局限性。

**为什么会出现未知物种 ID**

```
预训练数据中的物种集合 ≠ 微调数据中的物种集合

预训练阶段：
  ncbi2id.json 文件定义了 ~10000 个已知物种
  例如：{"562": 0, "9606": 1, "83332": 2, ..., "12345": 9999}
  Embedding 矩阵：[10000, 1280]（每个物种一个 1280 维向量）

微调数据：
  可能包含预训练中未见过的物种
  例如：某个新测序的细菌，NCBI ID = "2847364"
  这个 ID 不在 ncbi2id.json 中
```

**当前的回退策略**

```
代码采用了三种酶家族各自的硬编码回退：

1. ChlR (反应 18421)：
   未知物种 → 回退到 "562" (E. coli)
   原因：ChlR 最初在 E. coli 中被研究，E. coli 是最常见的原核宿主

2. AadA (反应 20245)：
   未知物种 → 回退到 "99287" (Salmonella typhimurium)
   原因：AadA 氨基糖苷腺苷酰转移酶最初在沙门氏菌中发现

3. TPMT：
   未知物种 → 回退到 "9606" (Homo sapiens)
   原因：TPMT 是重要的人类药物代谢酶
```

**这种策略的问题**

```
问题 1：硬编码，不可扩展
  如果你要微调一个新的酶家族（比如某个植物来源的酶），
  代码中没有对应的回退规则，会直接报 KeyError 崩溃。
  你需要手动修改代码添加新的回退。

问题 2：生物学上不够精确
  简单回退到一个固定物种，忽略了实际的进化关系。
  例如：一个来自 Pseudomonas (假单胞菌) 的蛋白质
  被回退到 E. coli，虽然都是革兰氏阴性菌，
  但它们的序列偏好可能有显著差异。

问题 3：对模型效果的影响
  物种嵌入编码了该物种蛋白质的统计偏好
  （比如密码子使用偏好、氨基酸频率等）。
  使用错误的物种嵌入可能引入轻微的偏差。
  但由于物种嵌入只是加到 token embedding 上的一个 bias，
  影响通常较小——模型主要依赖序列本身的上下文。
```

**更好的替代方案（如果你要修改代码）**

```
方案 A：最近邻物种映射
  1. 预计算 NCBI 分类树中所有物种之间的距离
  2. 对于未知物种，找到分类树中最近的已知物种
  例如：未知物种属于 Enterobacteriaceae（肠杆菌科）
       → 找到同科中最近的已知物种（可能是 E. coli）

方案 B：随机分配
  对于未知物种，随机分配一个 [0, max_id] 之间的索引
  这相当于使用一个随机的物种嵌入
  效果可能不如方案 A，但至少不会崩溃

方案 C：使用特殊的 UNK 物种嵌入
  在 ncbi2id 中预留一个 "UNK" 索引
  所有未知物种都映射到这个索引
  训练时让 UNK 嵌入学习一个"平均物种"的表示
```

---

## 32. 配体数据集三件套：原子、坐标、结合标签

### 32.1 LigandAtomDataset

```python
# 来自 indexed_dataset.py (line 546-578)
class LigandAtomDataset(FairseqDataset):
    def read_data(self, path, split, protein):
        data = json.load(open(path))
        for protein in data:
            ligand_features = [feature[0] for feature in data[protein][split]["ligand_feat"]]
            lines.extend(ligand_features)
        
        for line in lines:
            features = torch.tensor(line)  # [N_atoms, 5]
            self.tokens_list.append(features)
```

### 32.2 LigandCoordinateDataset

与 CoordinateDataset 类似，但解析 `ligand_coor` 字段。不做中心化处理。

### 32.3 LigandBindingDataset

```python
# 简单的二值标签加载
# ligand_binding: [0, 1, 1, 0, ...]
# 0 = 不结合，1 = 结合
```

### 32.4 配体数据的完整示例

下面以华法林（Warfarin，一种常见的抗凝血药物）为例，展示配体数据是如何表示的。

**华法林的基本信息**

```
化学式：C₁₉H₁₆O₄
分子量：308.33 g/mol
非氢原子数：23（19个碳 + 4个氧）
在 EnzyGen2 中：只保留非氢原子（氢原子在蛋白质-配体交互中影响较小）
```

**ligand_atom 张量：[N_atoms, 5]**

每个原子用 5 个特征描述。以华法林的前 5 个原子为例：

```
特征含义：[原子序数, 成键数, 形式电荷, 是否芳香, 氢原子数]

原子 0 (碳，苯环上):  [6, 3, 0, 1, 0]
  原子序数=6（碳）, 成键数=3（苯环碳连接3个键）,
  形式电荷=0, 是否芳香=1（在苯环上）, 连接氢原子数=0

原子 1 (碳，苯环上):  [6, 3, 0, 1, 1]
  原子序数=6（碳）, 成键数=3, 形式电荷=0,
  是否芳香=1, 连接氢原子数=1

原子 2 (氧，羰基):    [8, 1, 0, 0, 0]
  原子序数=8（氧）, 成键数=1（双键中的一个）,
  形式电荷=0, 是否芳香=0（不在芳香环上）, 连接氢原子数=0

原子 3 (碳，甲基):    [6, 4, 0, 0, 3]
  原子序数=6（碳）, 成键数=4（sp3碳）,
  形式电荷=0, 是否芳香=0, 连接氢原子数=3

原子 4 (氧，羟基):    [8, 2, 0, 0, 1]
  原子序数=8（氧）, 成键数=2, 形式电荷=0,
  是否芳香=0, 连接氢原子数=1（-OH）

...依此类推，共 23 个原子
完整 ligand_atom 张量形状：[23, 5]
```

**ligand_coor 张量：[N_atoms, 3]**

每个原子的三维坐标（来自晶体结构或计算构象）：

```
原子 0 (碳):  [ 2.145,  0.893, -0.216]
原子 1 (碳):  [ 3.421,  1.452,  0.102]
原子 2 (氧):  [-0.892,  2.341,  0.553]
原子 3 (碳):  [ 4.210, -0.563,  1.876]
原子 4 (氧):  [ 1.223,  3.112, -1.045]
...
完整 ligand_coor 张量形状：[23, 3]

注意：配体坐标不做中心化（与蛋白质坐标不同）
原因：配体坐标需要与蛋白质坐标在同一参考系中
（配体通常位于活性位点附近），中心化会破坏这个空间关系。
但蛋白质坐标是以 Motif 中心为原点的，所以配体坐标
实际上也是相对于 Motif 中心的位置——它们在数据预处理时
已经被放在了蛋白质的坐标系中。
```

**ligand_binding 标签：标量**

```
ligand_binding = 1（结合）

这个标签表示该配体确实与对应的蛋白质结合。
在训练数据中：
  正样本（binding=1）：配体-蛋白质对来自 PDB 共晶结构
  负样本（binding=0）：随机配对的配体-蛋白质对（不来自同一个晶体结构）
```

**SubstrateEGNN 如何处理这些数据**

```
1. ligand_atom [23, 5] → 线性投影 → [23, 1280]（与蛋白质特征同维度）
2. ligand_coor [23, 3] → 作为节点坐标
3. 构建配体原子之间的 KNN 图
4. 经过 3 层 EGNN 处理 → 得到更新后的 [23, 1280] 特征
5. 全局平均池化 → [1, 1280]（配体全局表示 sub_feats）
6. 与蛋白质全局表示拼接 → [1, 2560] → MLP → [1, 2]（结合概率）
```

---

## 33. Batch Collation：从样本到批次

### 33.1 Collation 函数概览

```python
# 来自 ncbi_protein_dataset.py 的 collate() 函数

def collate(samples, pad_idx, eos_idx, ...):
    # 1. 序列 padding
    src_tokens = merge("source", pad_idx, eos_idx)    # [B, L_max]
    src_lengths = ...                                   # [B]
    
    # 2. 按长度降序排序（提高计算效率）
    src_lengths, sort_order = src_lengths.sort(descending=True)
    
    # 3. 坐标 padding（用零填充）
    target_coor = collate_coor("target", pad_idx)      # [B, L_max, 3]
    
    # 4. Motif mask padding
    motif_input = collate_motif("motif", pad_idx)       # [B, L_max]
    motif_output = collate_motif("motif", pad_idx)      # [B, L_max]
    
    # 5. 配体数据 padding（如果有）
    ligand_atom = collate_sub_atom("ligand_atom", pad_idx)  # [B, N_max, 5]
    ligand_coor = collate_coor("ligand_coor", pad_idx)      # [B, N_max, 3]
    ligand_binding = collate_label("ligand_binding")         # [B]
    
    # 6. 打包成 batch 字典
    batch = {
        "id": ids,
        "nsentences": batch_size,
        "ntokens": total_tokens,
        "source_input": {
            "src_tokens": [B, L_max],
            "src_lengths": [B]
        },
        "target_input": {
            "target_coor": [B, L_max, 3],
            "tgt_lengths": [B]
        },
        "motif": {
            "input": [B, L_max],
            "output": [B, L_max]
        },
        "ligand_input": {                    # 仅预训练有
            "ligand_coor": [B, N_max, 3],
            "ligand_atom": [B, N_max, 5],
            "ligand_binding": [B]
        },
        "pdb": [str, ...],
        "ncbi": [int, ...],
        "center": [Tensor, ...]
    }
```

### 33.2 预训练 vs 微调的区别

```
NCBIDataset（预训练）：
  包含配体数据（ligand_atom, ligand_coor, ligand_binding）
  用于 Stage 3（Full Training）

NCBIFinetuneDataset（微调/推理）：
  不包含配体数据
  batch 中无 ligand_input 字段
  用于 Finetuning 和 Generation
```

### 33.3 Collation 的数值示例

下面用一个具体的例子，完整追踪 batch collation 如何将不同长度的蛋白质样本组合成统一格式的 batch 张量。

**输入：2 条蛋白质，长度不同**

```
蛋白质 A：序列 "MKVAV"（5 残基），加 BOS/EOS 后 7 个 token
蛋白质 B：序列 "GSTERALKVFG"（11 残基），加 BOS/EOS 后 13 个 token

L_max = max(7, 13) = 13
```

**Step 1：序列 token 的 padding**

```
蛋白质 A 的 tokens: [0, 14, 16, 7, 5, 7, 2]（长度 7）
蛋白质 B 的 tokens: [0, 6, 8, 11, 9, 10, 5, 4, 16, 7, 13, 6, 2]（长度 13）

padding 到 L_max=13，用 <pad>=1 填充：

src_tokens: [2, 13]
  蛋白质 A: [0, 14, 16,  7,  5,  7, 2, 1, 1, 1, 1, 1, 1]
  蛋白质 B: [0,  6,  8, 11,  9, 10, 5, 4, 16, 7, 13, 6, 2]
                                              真实数据↑  ↑padding→

注意：按长度降序排序后，蛋白质 B 排在前面：
  排序后：
  row 0 (蛋白质B): [0,  6,  8, 11,  9, 10, 5, 4, 16, 7, 13, 6, 2]
  row 1 (蛋白质A): [0, 14, 16,  7,  5,  7, 2, 1,  1, 1,  1, 1, 1]

src_lengths: [13, 7]（降序）
```

**Step 2：坐标的 padding**

```
蛋白质 A 的坐标：[7, 3]（7 个位置，每个 3 维）
  BOS: [0.0, 0.0, 0.0]
  M:   [-5.25, -2.40, -1.75]
  K:   [-1.75, -0.60, -0.55]
  V:   [1.75,  0.60,  0.55]
  A:   [5.25,  1.90,  1.75]
  V:   [8.45,  3.20,  2.45]
  EOS: [0.0, 0.0, 0.0]

padding 到 [13, 3]，用 [0.0, 0.0, 0.0] 填充后 6 个位置：

target_coor: [2, 13, 3]
  row 0 (蛋白质B): [[0,0,0], [-3.1,1.2,0.8], ..., [4.5,2.1,-0.3], [0,0,0]]  ← 全部是真实坐标
  row 1 (蛋白质A): [[0,0,0], [-5.25,-2.4,-1.75], ..., [0,0,0], [0,0,0],...,[0,0,0]]
                                                        ↑EOS      ↑padding(全零)
```

**Step 3：motif_mask 的 padding**

```
蛋白质 A 的 motif_mask：[0, 1, 0, 0, 1, 1, 0]（7 个位置）
  0=BOS, 1=scaffold(M), 0=motif(K), 0=motif(V), 1=scaffold(A), 1=scaffold(V), 0=EOS

padding 到 [13]，用 0 填充（padding 位置不参与损失计算）：

motif: [2, 13]
  row 0 (蛋白质B): [0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0]
  row 1 (蛋白质A): [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
                                         ↑EOS ←padding（全零）→

关键：padding 位置的 mask 值为 0，表示这些位置"不需要预测"。
这确保了 padding 位置不会对损失函数产生贡献。
```

**Step 4：配体数据的 padding（仅预训练 Stage 3）**

```
假设蛋白质 A 的配体有 15 个原子，蛋白质 B 的配体有 23 个原子
N_max = max(15, 23) = 23

ligand_atom: [2, 23, 5]
  row 0 (蛋白质B): 23 个原子的 5 维特征（无 padding）
  row 1 (蛋白质A): 15 个原子的 5 维特征 + 8 个 [0,0,0,0,0] padding

ligand_coor: [2, 23, 3]
  row 0 (蛋白质B): 23 个原子的 3D 坐标（无 padding）
  row 1 (蛋白质A): 15 个原子的 3D 坐标 + 8 个 [0,0,0] padding

ligand_binding: [2]
  [1, 0]（蛋白质 B 结合配体，蛋白质 A 不结合）
```

**完整 batch 字典结构**

```python
batch = {
    "id": tensor([1, 0]),             # 排序后的原始索引
    "nsentences": 2,                   # batch 大小
    "ntokens": 20,                     # 实际 token 总数 (13 + 7)
    "source_input": {
        "src_tokens": tensor([2, 13]),     # 序列 token
        "src_lengths": tensor([13, 7])     # 真实长度（降序）
    },
    "target_input": {
        "target_coor": tensor([2, 13, 3]), # 目标坐标
        "tgt_lengths": tensor([13, 7])
    },
    "motif": {
        "input": tensor([2, 13]),          # 输入遮蔽
        "output": tensor([2, 13])          # 输出遮蔽（用于损失计算）
    },
    "ligand_input": {                      # 仅 Stage 3
        "ligand_coor": tensor([2, 23, 3]),
        "ligand_atom": tensor([2, 23, 5]),
        "ligand_binding": tensor([2])
    },
    "pdb": ["1XYZ_B", "3DWZ_A"],           # PDB ID（按排序后顺序）
    "ncbi": [562, 9606],                    # 物种 ID
    "center": [tensor([3]), tensor([3])]    # Motif 中心坐标
}
```

### 33.4 为什么 padding 值是 0 而不是 -1

在 batch collation 中，不同的张量使用不同的 padding 值。理解每种选择背后的原因，需要从"padding 位置在下游计算中如何被处理"的角度思考。

**各张量的 padding 值总结**

```
┌─────────────────┬────────────┬──────────────────────────────────────────┐
│ 张量            │ Padding 值 │ 原因                                     │
├─────────────────┼────────────┼──────────────────────────────────────────┤
│ src_tokens      │ 1 (PAD)    │ PAD 是字典中 index=1 的特殊 token        │
│ target_coor     │ [0,0,0]    │ 原点坐标，中心化后的合理默认值           │
│ motif_mask      │ 0          │ 0=Motif（不预测），padding 不参与损失    │
│ ligand_atom     │ [0,0,0,0,0]│ 零向量，被 ligand mask 排除              │
│ ligand_coor     │ [0,0,0]    │ 原点坐标，与蛋白质坐标一致              │
│ ligand_binding  │ 无需padding│ 每条蛋白质只有一个标量，不存在长度差异   │
└─────────────────┴────────────┴──────────────────────────────────────────┘
```

**为什么 src_tokens 用 1 (PAD) 而不是 0 或 -1**

```
ESM-1b 的字典定义了特殊 token 的索引：
  index 0 = <cls>（CLS，句首标记）
  index 1 = <pad>（PAD，填充标记）
  index 2 = <eos>（EOS，句尾标记）
  index 3 = <unk>（UNK，未知标记）

PAD 必须是 index=1，因为 Transformer 的 attention mask 会根据
src_tokens == pad_index 来构建：
  attention_mask = (src_tokens != 1)  # True = 参与 attention
  
  例如：
    src_tokens = [0, 14, 16, 7, 2, 1, 1, 1]
    attention   = [T,  T,  T, T, T, F, F, F]
  
  PAD 位置被完全排除在 self-attention 计算之外。
  如果用 0（CLS 的索引）做 padding，模型会把 padding 当成 CLS token，
  产生错误的 attention 模式。
  如果用 -1，PyTorch embedding 层会报错（索引必须 >= 0）。
```

**为什么 motif_mask 用 0 而不是 1 或 -1**

```
回顾 motif_mask 的约定：
  0 = Motif 位置（已知，不需要预测，不计算损失）
  1 = Scaffold 位置（需要预测，参与损失计算）

padding 位置应该"不参与损失计算"。
在损失函数中，只有 motif_mask==1 的位置才会计算损失：

  loss = Σ_{i: mask[i]==1} -log P(target_i | ...)  / count(mask==1)

如果 padding 位置的 mask = 0：
  它们不会被纳入损失计算 ← 正确！

如果 padding 位置的 mask = 1：
  它们会被当成"需要预测的 scaffold 位置"
  模型会尝试预测 padding 位置的氨基酸和坐标
  但 padding 位置的 target token 是 <pad>=1，不是真实氨基酸
  → 损失函数会计算 -log P(pad_token)，产生错误的梯度信号
  → 训练被污染！

如果 padding 位置的 mask = -1：
  需要在损失函数中额外处理"跳过 -1"的逻辑
  增加代码复杂性，没有实际收益

数值验证：
  batch 中蛋白质 A（长度 7）和蛋白质 B（长度 13）
  蛋白质 A 的 motif_mask padding 后：
    [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
                         ↑EOS ←6 个 padding（全为 0）→
  
  损失只在 mask==1 的位置计算：位置 1, 4, 5（共 3 个位置）
  6 个 padding 位置全部被跳过 ← 正确！
```

**为什么坐标 padding 用 [0,0,0] 而不是 [NaN] 或 [-999]**

```
中心化后，原点 (0,0,0) 就是 Motif 的中心位置。
用 [0,0,0] 做 padding 有三个好处：

1. 数值安全
   EGNN 在计算节点间距离时会涉及 padding 位置：
   d = ||x_i - x_j||
   
   如果 x_j = [0,0,0]（原点）：
     d = ||x_i||，一个有限的正数 → 数值安全
   
   如果 x_j = [NaN, NaN, NaN]：
     d = NaN → NaN 会传播到所有依赖节点 → 训练崩溃
   
   如果 x_j = [-999, -999, -999]：
     d 会非常大 → 距离的倒数非常小 → 梯度可能数值不稳定
     而且 [-999,-999,-999] 远离蛋白质的正常坐标范围

2. 与 KNN 图的兼容性
   KNN 图根据距离选择最近的 K 个邻居。
   padding 位置在原点，距离 Motif 中心很近（0 Å）。
   但这没关系，因为 KNN 是在真实 token 之间构建的，
   padding 位置被 attention mask 排除在外。

3. 与 BOS/EOS 的一致性
   BOS 和 EOS 的坐标也是 [0,0,0]。
   padding 使用相同的值保持了一致性：
     BOS: [0,0,0]  ← 非蛋白质位置
     EOS: [0,0,0]  ← 非蛋白质位置
     PAD: [0,0,0]  ← 非蛋白质位置
   
   所有非蛋白质位置都在原点，模型可以轻松学会"原点附近的坐标不重要"。
```

**一个常见疑问：padding 的 [0,0,0] 会不会干扰结构损失？**

```
答案：不会。因为结构损失只在 motif_mask==1 的位置计算。

回顾损失计算：
  L_struct = Σ_{i: mask[i]==1} ||pred_coor[i] - true_coor[i]||² / count(mask==1)

padding 位置的 mask = 0，不在求和范围内。
即使 padding 的坐标是 [0,0,0]，也不会影响损失值。

但 padding 坐标可能间接影响 EGNN 的消息传递——如果 padding 位置
被包含在 KNN 图中的话。实际上，代码在构建 KNN 图时使用了 padding mask
来排除 padding 位置，确保消息只在真实残基之间传递。
```

---

## 34. Stage 1：掩码语言模型预训练（MLM）

### 34.1 目标

让模型学会从 20% 的已知残基恢复 80% 的被遮蔽残基的序列和坐标。

### 34.2 训练脚本

```bash
# train_EnzyGen2_mlm.sh

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7    # 8 GPU

python3 fairseq_cli/train.py data/pdb_swissprot_data_ligand.json \
  --distributed-world-size 8 \
  --save-dir models/EnzyGen2_MLM \
  --task geometric_protein_design \
  --data-stage "pretraining-mlm" \                     # ← MLM 遮蔽策略
  --criterion geometric_protein_ncbi_loss \             # ← 序列+结构损失
  --encoder-factor 1.0 --decoder-factor 1e-2 \
  --arch geometric_protein_model_ncbi_esm \             # ← 基础模型（无配体）
  --encoder-embed-dim 1280 \
  --egnn-mode "rm-node" \                               # ← EGNN 变体
  --decoder-layers 3 \
  --pretrained-esm-model esm2_t33_650M_UR50D \          # ← ESM2 预训练权重
  --knn 30 \
  --dropout 0.3 \
  --optimizer adam --adam-betas '(0.9,0.98)' \
  --lr 1e-4 --lr-scheduler inverse_sqrt \
  --warmup-updates 4000 --warmup-init-lr 5e-5 \
  --clip-norm 1.5 \
  --max-tokens 1024 \
  --max-update 1000000 \
  --validate-interval-updates 3000
```

### 34.3 数据流

```
训练数据：pdb_swissprot_data_ligand.json
  ├── 多个物种（NCBI ID 作为 key）
  ├── 每个物种有 train/valid 分割
  └── 遮蔽策略：随机遮蔽 80% 残基

每个训练步：
  1. 随机选择一批蛋白质
  2. 随机遮蔽 80% 的非 BOS/EOS 残基
  3. 模型预测被遮蔽位置的序列和坐标
  4. 计算 L_seq + 0.01 × L_struct
  5. 反向传播更新参数
```

### 34.4 命令行参数详解

```
关键参数逐一解释：

--distributed-world-size 8
  使用 8 张 GPU 进行数据并行训练。每张 GPU 处理 batch 的 1/8，
  梯度在 8 张 GPU 之间同步平均。

--data-stage "pretraining-mlm"
  控制 ProteinMotifDataset 使用随机遮蔽策略（80% 随机遮蔽）
  而不是 Motif 固定遮蔽。这是 Stage 1 的核心区分参数。

--criterion geometric_protein_ncbi_loss
  使用序列恢复 + 结构预测的损失函数（不含配体结合损失）。
  对应 geometric_protein_ncbi_loss.py 中的实现。

--encoder-factor 1.0 --decoder-factor 1e-2
  序列损失权重 α=1.0，结构损失权重 β=0.01
  结构损失的权重很小是因为 MSE 的数值量级比 NLL 大 100 倍。

--arch geometric_protein_model_ncbi_esm
  使用 GeometricProteinNCBIModel 架构（基础模型，无 SubstrateEGNN）。

--pretrained-esm-model esm2_t33_650M_UR50D
  初始化 Transformer 部分使用 ESM2-650M 的预训练权重。
  这是迁移学习的关键——不从零开始学习蛋白质语言。

--egnn-mode "rm-node"
  EGNN 使用 "rm-node" 变体（边模型 + 坐标模型，移除了原始 EGNN 的节点模型）
  这个变体在实验中效果最好。

--knn 30
  EGNN 中每个节点连接最近的 30 个邻居。
  对于平均长度 ~300 残基的蛋白质，30 个邻居覆盖了约 10% 的残基，
  足以捕捉局部空间关系。

--lr 1e-4 --lr-scheduler inverse_sqrt
  初始学习率 1e-4，使用 inverse_sqrt 调度器：
  lr(t) = lr_init × min(1, t/warmup) × sqrt(warmup/max(t, warmup))
  前 4000 步从 5e-5 线性增加到 1e-4，之后按 1/sqrt(t) 衰减。

--warmup-updates 4000 --warmup-init-lr 5e-5
  学习率预热：从 5e-5 开始，在 4000 步内线性增加到 1e-4。
  预热防止训练初期因大学习率导致的梯度爆炸。

--clip-norm 1.5
  梯度范数裁剪阈值 1.5。如果梯度范数 > 1.5，等比缩放使其 = 1.5。
  防止偶尔出现的大梯度破坏训练。

--max-tokens 1024
  每个 batch 中最多包含 1024 个 token（不是 1024 条蛋白质！）
  例如：一条 500 残基的蛋白质 = 502 token，一个 batch 放 2 条。

--max-update 1000000
  最多训练 100 万步。实际可能更早停止（通过 early stopping 或验证集）。

--validate-interval-updates 3000
  每 3000 步在验证集上评估一次。用于监控过拟合和选择最佳 checkpoint。
```

### 34.5 训练过程中的关键指标

```
典型的 Stage 1 训练日志（假设值）：

Step      L_seq    L_struct (×0.01)  L_total   LR        Grad_norm
─────────────────────────────────────────────────────────────────────
500       285.3    82.5 (0.825)      286.1     7.5e-5    1.12
1000      198.7    45.2 (0.452)      199.2     9.0e-5    0.98
4000      142.5    28.7 (0.287)      142.8     1.0e-4    0.85    ← warmup 结束
10000     98.3     18.4 (0.184)      98.5      6.3e-5    0.72
50000     65.2     12.1 (0.121)      65.3      1.4e-5    0.55
200000    48.7     8.3  (0.083)      48.8      7.1e-6    0.42
500000    42.1     6.5  (0.065)      42.2      4.5e-6    0.38    ← 接近收敛
1000000   39.5     5.8  (0.058)      39.6      3.2e-6    0.35

观察要点：
  1. L_seq 从 ~280 下降到 ~40，说明模型从随机猜测逐步学会了氨基酸预测
  2. L_struct 持续下降但幅度较小（加权后只有 ~0.05-0.8）
  3. 学习率在 step 4000 达到峰值后缓慢衰减
  4. 梯度范数逐步减小，表明训练趋于稳定

何时认为 Stage 1 训练完成：
  - L_seq 连续 3 次验证评估没有显著下降（< 1% 改善）
  - 或达到 max_update（100 万步）
  - 典型训练时间：8 GPU (A100) 约 3-7 天
```

---

## 35. Stage 2：Motif 约束预训练

### 35.1 与 MLM 的区别

```
Stage 1 (MLM)：
  遮蔽策略 = 随机遮蔽 80% 的残基
  模型学会：从少量线索恢复序列和结构（通用能力）

Stage 2 (Motif)：
  遮蔽策略 = 只保留 Motif 位点，遮蔽其余
  模型学会：给定功能位点，设计周围的框架（设计能力）

这是从"理解蛋白质"到"设计蛋白质"的关键转变。
```

### 35.2 训练脚本关键差异

```bash
# train_EnzyGen2_motif.sh

--finetune-from-model models/EnzyGen2_MLM/checkpoint_best.pt  # ← 从 Stage 1 接力
--data-stage "pretraining-motif"                                # ← Motif 遮蔽策略
# 其余参数与 Stage 1 相同
```

### 35.3 为什么 Stage 2 只改变了遮蔽策略

```
Stage 1 → Stage 2 只有两个参数变化：
  1. --finetune-from-model：从 Stage 1 的最佳 checkpoint 继续训练
  2. --data-stage：从 "pretraining-mlm" 改为 "pretraining-motif"

所有其他参数保持不变：
  - 学习率：1e-4（不降低，因为模型需要快速适应新的遮蔽模式）
  - 梯度裁剪：1.5（保持不变）
  - 模型架构：基础模型（不变）
  - 损失函数：序列+结构（不变）

为什么不降低学习率？
  虽然是从 Stage 1 的权重继续训练，但遮蔽策略的改变
  等同于一个全新的任务。模型需要足够大的学习率来快速
  适应从"完形填空"到"设计"的转变。如果学习率太小，
  模型会停留在 Stage 1 学到的"预测"模式中，无法转变。
```

### 35.4 训练过程中的关键指标

```
Stage 2 训练日志（从 Stage 1 的 checkpoint 继续）：

Step      L_seq    L_struct (×0.01)  L_total   注释
─────────────────────────────────────────────────────────────
500       165.4    35.2 (0.352)      165.8     初始损失比 Stage 1 结束时高
                                               因为遮蔽比例从 20% 变成 50-80%
2000      128.3    22.1 (0.221)      128.5     快速下降——利用了 Stage 1 的知识
10000     85.6     14.7 (0.147)      85.7      
50000     62.1     9.8  (0.098)      62.2      
200000    51.3     7.2  (0.072)      51.4      
500000    46.8     6.1  (0.061)      46.9      接近收敛

观察要点：
  1. 初始损失比 Stage 1 结束时高，这是正常的
     → 遮蔽比例增大了（从 20% → 50-80%），任务更难了
  2. 但下降速度比 Stage 1 快
     → 因为模型已经学会了蛋白质语言的基础知识
  3. 结构损失在 Stage 2 更重要
     → Motif-Scaffolding 要求精确的空间安排

何时认为 Stage 2 训练完成：
  - 验证集损失不再显著下降
  - 典型训练时间：8 GPU (A100) 约 3-7 天
  - 如果 Stage 1 训练充分，Stage 2 可能更快收敛
```

---

## 36. Stage 3：完整训练（含配体结合）

### 36.1 新增内容

Stage 3 在 Stage 2 的基础上引入配体结合预测：

```
Stage 2 → Stage 3 的变化：
  模型架构：NCBIModel → NCBISubstrateModel（新增 SubstrateEGNN + score 层）
  损失函数：ncbi_loss → ncbi_substrate_loss（新增 binding loss）
  数据：额外加载配体原子、坐标、结合标签
```

### 36.2 训练脚本关键差异

```bash
# train_EnzyGen2_full.sh

--finetune-from-model models/EnzyGen2_motif/checkpoint_best.pt  # ← 从 Stage 2 接力
--data-stage "pretraining-full"                                   # ← 完整数据（含配体）
--criterion geometric_protein_ncbi_substrate_loss \               # ← 新增结合损失
  --encoder-factor 1.0 --decoder-factor 1e-2 --binding-factor 0.5
--arch geometric_protein_model_ncbi_substrate_esm \               # ← SubstrateModel
--lr 5e-5 \                                                       # ← 更低学习率
--clip-norm 0.0001 \                                              # ← 更小梯度裁剪
--max-tokens 800 \                                                # ← 更小 batch（配体数据更大）
--max-sentences 1 \                                               # ← 每 GPU 1 条
```

### 36.3 学习率和梯度裁剪的变化

```
Stage 1-2: lr=1e-4, clip-norm=1.5
Stage 3:   lr=5e-5, clip-norm=0.0001

原因：
  1. 接近训练末期，需要更小的学习率进行精细调整
  2. 新增的 SubstrateEGNN 模块需要稳定训练
  3. 配体结合损失可能产生较大梯度，需要严格裁剪
```

### 36.4 Stage 3 新增的配体处理流程

Stage 3 最核心的改动是引入了配体（底物）信息。下面逐步说明配体数据在前向传播中的完整处理过程：

```
步骤 1：加载配体原子特征
  数据集提供 ligand_atom 张量，形状 [B, N, 5]
    B = 批次大小
    N = 配体最大原子数（短的用零填充）
    5 = 每个原子的特征维度（原子类型 one-hot 等）
  
  通过线性投影嵌入到高维空间：
    Linear(5, 1280): [B, N, 5] → [B, N, 1280]
  然后展平为图节点格式：
    reshape: [B, N, 1280] → [B*N, 1280]
  
  此时每个配体原子变成一个 1280 维的特征向量，
  与蛋白质侧的隐藏维度完全对齐。

步骤 2：加载配体原子三维坐标
  数据集提供 ligand_coor 张量，形状 [B, N, 3]
    3 = (x, y, z) 三维笛卡尔坐标
  
  展平为：[B*N, 3]
  
  这些坐标定义了配体原子在三维空间中的精确位置，
  是构建空间邻接图的基础。

步骤 3：构建配体 KNN 图
  使用 sklearn.neighbors.NearestNeighbors 在配体原子坐标上构建 K 近邻图：
    K = 30（与蛋白质侧相同的默认值）
  
  生成边列表 edges: [2, B*N*K]
    第一行是源节点索引，第二行是目标节点索引
  
  同时计算：
    radial: [B*N*K, 1]  — 每条边的距离平方
    coord_diff: [B*N*K, 3] — 每条边的坐标差向量

步骤 4：SubstrateEGNN 处理配体图（3 层）
  SubstrateEGNN 的结构与蛋白质侧的 EGNN 完全相同（E_GCL 层），
  但参数独立，专门处理配体的图结构。
  
  每一层的操作：
    第 1 层：节点特征 [B*N, 1280] + 坐标 [B*N, 3]
      → 消息传递（基于距离和坐标差）
      → 节点特征更新 + 坐标微调
    第 2 层：在第 1 层输出上继续更新
    第 3 层：在第 2 层输出上继续更新
  
  输出：
    sub_h: [B*N, 1280]  — 更新后的配体原子特征
    sub_x: [B*N, 3]     — 更新后的配体原子坐标

步骤 5：求和池化得到配体全局表示
  将每个样本内的所有配体原子特征求和：
    sub_h: [B*N, 1280] → reshape [B, N, 1280] → sum(dim=1) → [B, 1280]
  
  这一步将可变长度的配体原子级表示压缩为固定长度的分子级表示。
  使用求和（而非平均）是因为不同大小的配体应有不同的"信号强度"。

步骤 6：与蛋白质表示拼接
  蛋白质全局表示通过对 Transformer 输出做平均池化获得：
    protein_rep: [B, L, 1280] → mean(dim=1) → [B, 1280]
  
  拼接蛋白质和配体表示：
    concat([protein_rep, sub_feats], dim=-1) → [B, 2560]
  
  此时每个样本由一个 2560 维向量表示，
  前 1280 维编码蛋白质信息，后 1280 维编码配体信息。

步骤 7：线性分类 → 结合概率
  通过线性层映射到二分类：
    Linear(2560, 2): [B, 2560] → [B, 2]
  
  softmax 归一化：
    scores = softmax([B, 2], dim=-1)
    scores[:, 0] = 不结合概率
    scores[:, 1] = 结合概率
  
  交叉熵损失：
    L_bind = CrossEntropy(scores, binding_labels)
    其中 binding_labels 是 0/1 标签
```

```
完整数据流图示：

ligand_atom [B, N, 5]                    蛋白质序列+坐标
       │                                       │
  Linear(5,1280)                          Transformer + EGNN
       │                                       │
  [B*N, 1280] + ligand_coor [B*N, 3]     [B, L, 1280]
       │                                       │
    KNN 图构建                            mean 池化
       │                                       │
  SubstrateEGNN (×3层)                    [B, 1280]
       │                                       │
  sum 池化 → [B, 1280]                         │
       │                                       │
       └──────── concat ────────────────────────┘
                    │
              [B, 2560]
                    │
            Linear(2560, 2)
                    │
              softmax → [B, 2]
                    │
            L_bind = CrossEntropy
```

### 36.5 训练过程中的关键指标

Stage 3 训练时需要同时监控三个损失分量和总损失，以下是各指标的典型数值范围和诊断方法：

```
═══════════════════════════════════════════════════════════
  指标           训练初期        训练中期        训练后期
═══════════════════════════════════════════════════════════
  L_bind         ≈ 0.69         ≈ 0.40-0.50    ≈ 0.20-0.40
  L_seq          与Stage2末期    稳定或略升      ≈ Stage2水平
                 相当                           或更低
  L_struct       与Stage2末期    稳定或略升      ≈ Stage2水平
                 相当                           或更低
  总损失         轻微上升        逐渐下降        收敛
═══════════════════════════════════════════════════════════
```

**L_bind（配体结合损失）的典型变化：**

```
训练开始时：
  L_bind ≈ 0.693 = -log(0.5)
  
  这是因为模型对结合/不结合两个类别的预测概率接近 50%/50%，
  相当于随机猜测。这是二分类交叉熵的理论最大值。
  
  数学解释：
    如果 p(结合) ≈ 0.5，则：
    -log(0.5) = 0.693

训练中期（约 50k-200k 步）：
  L_bind 下降到 0.40-0.50
  
  模型开始区分结合/不结合对，但置信度还不高。
  对应的预测准确率约 65%-75%。

训练后期（约 200k+ 步）：
  L_bind 收敛到 0.20-0.40
  
  模型能较好地预测酶-底物结合关系。
  对应的预测准确率约 80%-90%。
  
  注意：L_bind 不太可能降到 0.1 以下，
  因为酶-底物结合预测本身就有不确定性。
```

**L_seq 和 L_struct 的监控要点：**

```
正常情况：
  Stage 3 开始时，L_seq 和 L_struct 的值应该接近 Stage 2 结束时的水平。
  随着训练进行，这两个值应该保持稳定或略有改善。

异常情况 1：L_seq 和 L_struct 在 Stage 3 开始后急剧上升
  原因：学习率太高，新增的 binding loss 梯度干扰了已有参数
  解决：降低学习率（从 5e-5 降到 1e-5）或增大 warmup 步数

异常情况 2：L_bind 不下降，长期停留在 0.69 附近
  原因：(1) 学习率太低，梯度无法传播到 SubstrateEGNN
        (2) 配体数据存在问题（全是正样本或全是负样本）
        (3) binding_factor 权重太小
  解决：检查数据中正/负样本的比例，适当提高 binding_factor

异常情况 3：L_bind 下降但 L_seq 上升
  原因：模型过度关注配体结合，牺牲了序列恢复能力
  解决：降低 binding_factor（从 0.5 降到 0.1-0.2），
        或增大 encoder_factor 的权重
```

**TensorBoard 监控建议：**

```bash
# 推荐同时监控以下指标
tensorboard --logdir=models/EnzyGen2_full/

# 关键曲线：
#   loss          — 总损失（应持续下降）
#   loss_seq      — 序列损失（应保持稳定）
#   loss_struct   — 结构损失（应保持稳定）
#   loss_bind     — 结合损失（应从 0.69 下降）
#   lr            — 学习率（应按 inverse_sqrt 衰减）
#   grad_norm     — 梯度范数（应被 clip_norm=0.0001 控制在低水平）
```

---

## 37. 微调（Finetuning）特定酶家族

### 37.1 微调脚本

```bash
# reah_ChlR_finetune.sh

export CUDA_VISIBLE_DEVICES=1                                    # 单 GPU

python3 fairseq_cli/finetune.py data/rhea_18421_final.json \     # ← ChlR 特定数据
  --finetune-from-model models/EnzyGen2/checkpoint_best.pt \     # ← 从 Stage 3 接力
  --data-stage "finetuning" \                                     # ← 微调模式
  --criterion geometric_protein_ncbi_loss \                       # ← 回到无配体损失
  --arch geometric_protein_model_ncbi_esm \                       # ← 回到基础模型
  --protein-task 18421 \                                          # ← 反应 ID
  --lr 5e-5 \
  --max-update 300000 \
  --max-epoch 50
```

### 37.2 微调的关键设计选择

```
为什么微调时不使用配体损失？
  1. 微调数据是特定酶家族，所有样本都针对同一个配体
  2. 配体结合约束已经在预训练 Stage 3 中学到
  3. 微调专注于序列-结构恢复，提高特定家族的设计质量

为什么从 SubstrateModel 退回 NCBIModel？
  1. 微调数据没有配体标注
  2. SubstrateEGNN 和 score 层的参数不需要继续更新
  3. 基础模型更轻量，微调更高效
  
  注意：checkpoint 加载时会自动忽略多余的参数
  （SubstrateEGNN 的权重不会被加载到 NCBIModel 中）
```

### 37.3 三个酶家族的微调配置

| 参数 | ChlR | AadA | TPMT |
|------|------|------|------|
| 数据文件 | rhea_18421_final.json | rhea_20245_final.json | thiopurine_methyltransferase_final.json |
| 反应 ID | 18421 | 20245 | Thiopurine_S_methyltransferas |
| GPU | 1 | 1 | 1 |
| 默认物种 | E. coli (562) | Salmonella (99287) | Human (9606) |

### 37.4 微调与预训练的关键差异总结

下表从多个维度系统对比微调和预训练之间的核心差异：

| 维度 | 预训练（Stage 1-3） | 微调（Finetuning） |
|------|---------------------|---------------------|
| **数据来源** | UniRef + PDB 大规模通用数据（数十万条序列） | 特定酶家族数据（通常几十到几百条序列） |
| **遮蔽策略** | Stage 1: 随机遮蔽 80%；Stage 2-3: Motif 固定遮蔽 | 固定 Motif 遮蔽（与 Stage 2-3 相同） |
| **配体信息** | Stage 1-2: 无；Stage 3: 包含配体原子+坐标+结合标签 | 无（配体知识已编码在预训练权重中） |
| **物种信息** | 多样化的物种 ID（覆盖 10000+ 物种） | 固定为目标物种（如 E. coli 562） |
| **学习率** | Stage 1-2: 1e-4（较高，快速学习）；Stage 3: 5e-5 | 5e-5（较低，精细调整，避免灾难性遗忘） |
| **训练轮数** | max_epoch = 2000-10000（大数据集需要多轮） | max_epoch = 50（小数据集，少量轮数即可） |
| **总更新步数** | max_update = 1,000,000 | max_update = 300,000 |
| **GPU 数量** | 8 GPU（大数据、大批次） | 1 GPU（小数据，单卡即可） |
| **模型架构** | Stage 1-2: NCBIModel；Stage 3: NCBISubstrateModel | NCBIModel（回退到基础模型） |
| **损失组成** | Stage 1-2: L_seq + L_struct；Stage 3: + L_bind | L_seq + L_struct（专注序列-结构恢复） |
| **损失侧重** | 均衡学习所有损失分量 | 重点关注特定家族的序列恢复质量 |
| **梯度裁剪** | Stage 1-2: 1.5；Stage 3: 0.0001 | 0.0001（保持严格裁剪） |
| **目标** | 学习通用的蛋白质设计能力 | 针对特定酶家族优化设计质量 |
| **评估指标** | 多物种多酶的平均 ESP | 单一酶家族的 ESP + 序列恢复率 |

```
直觉理解：
  预训练 = 学会"如何设计酶"的通用能力
  微调   = 在通用能力基础上，成为"设计某种特定酶"的专家

  类比：
    预训练 → 医学院通识教育（学所有科室的基础知识）
    微调   → 专科住院医师培训（深入某一个科室）
```

### 37.5 微调的注意事项

微调是一个精细且容易出错的过程。以下是四个必须注意的关键点：

**（1）学习率必须足够低，避免灾难性遗忘**

```
灾难性遗忘（Catastrophic Forgetting）是微调中最常见的问题。

当学习率过高时，模型在适应新数据的过程中会覆盖预训练阶段
学到的通用知识，导致模型在目标任务上过拟合、在通用任务上
能力退化。

推荐学习率范围：1e-5 到 5e-5
  - 1e-5：最保守，适合数据量极少（<50 条）的情况
  - 3e-5：中等，适合数据量适中（50-200 条）的情况
  - 5e-5：默认值，适合数据量较多（>200 条）的情况

如何判断学习率是否过高？
  症状：训练损失在前几个 epoch 快速下降，但验证损失开始上升
  解决：降低学习率到原来的 1/3 - 1/5

如何判断学习率是否过低？
  症状：训练损失下降极慢，50 个 epoch 后仍未收敛
  解决：提高学习率到原来的 2-3 倍
```

**（2）梯度裁剪至关重要**

```
微调数据集通常很小（几十到几百条序列），这意味着：
  - 每个 batch 中的样本多样性低
  - 梯度估计的方差大
  - 个别异常样本可能产生极大的梯度

如果不进行梯度裁剪：
  一个异常样本的大梯度可能在一步更新中破坏模型的所有预训练权重。

推荐设置：
  --clip-norm 0.0001
  
  这个值非常小（与 Stage 3 相同），确保每步更新幅度极其有限。
  即使遇到异常样本，模型参数的变化也是可控的。

对比：
  Stage 1-2: clip-norm = 1.5    → 允许较大的参数更新
  Stage 3:   clip-norm = 0.0001 → 极其保守的更新
  微调:      clip-norm = 0.0001 → 延续 Stage 3 的保守策略
```

**（3）基于验证集 ESP 评分的早停（Early Stopping）**

```
微调时不应该一直训练到 max_epoch = 50，而应该根据验证集的
ESP 评分决定何时停止。

具体做法：
  1. 将数据集分为训练集（80%）和验证集（20%）
  2. 每 5 个 epoch 在验证集上运行一次推理，计算 ESP 评分
  3. 记录历史最佳 ESP 评分及其对应的 epoch
  4. 如果连续 10 个 epoch ESP 评分不再提升，停止训练
  5. 使用最佳 epoch 的 checkpoint 进行最终推理

为什么用 ESP 而不用训练损失？
  训练损失可能持续下降（过拟合），但 ESP 评分反映的是
  设计的酶是否真的能结合目标底物——这才是最终目标。

典型观察：
  Epoch 1-10:  ESP 快速提升（利用预训练知识适应新数据）
  Epoch 10-25: ESP 缓慢提升或波动（精细优化）
  Epoch 25-50: ESP 可能开始下降（过拟合开始发生）
  
  最佳 checkpoint 通常在 epoch 15-30 之间。
```

**（4）Motif 位置必须准确定义**

```
Motif 定义是酶设计成功与否的决定性因素。
错误的 Motif 定义 = 错误的设计目标 = 无效的设计结果。

常见错误：
  错误 1：将非催化位点标记为 Motif
    后果：模型会保留不重要的位点，浪费设计自由度
    
  错误 2：遗漏关键催化残基
    后果：模型可能修改这些残基，破坏催化活性
    
  错误 3：Motif 范围过大（>50% 的残基被标记为 Motif）
    后果：模型的设计空间太小，无法有效优化
    
  错误 4：Motif 范围过小（<5% 的残基被标记为 Motif）
    后果：缺乏足够的功能约束，生成的序列可能缺乏催化活性

推荐的 Motif 定义方法：
  1. 从文献或数据库（如 UniProt、CAZy）获取已知的催化位点残基
  2. 使用结构比对工具（如 TM-align）对齐参考结构
  3. 将催化位点残基 + 其一级序列邻居（±1-2 个残基）标记为 Motif
  4. 典型的 Motif 占总长度的 10-30%

验证方法：
  在 PyMOL 中可视化 Motif 位置，确认它们：
  - 包含所有已知的催化残基
  - 位于活性位点口袋内或附近
  - 不包含远离活性位点的结构元素
```

---

## 38. 超参数对比速查表

| 参数 | Stage 1 (MLM) | Stage 2 (Motif) | Stage 3 (Full) | Finetuning |
|------|--------------|-----------------|----------------|-----------|
| **模型架构** | ncbi_esm | ncbi_esm | ncbi_substrate_esm | ncbi_esm |
| **损失函数** | ncbi_loss | ncbi_loss | ncbi_substrate_loss | ncbi_loss |
| **encoder_factor** | 1.0 | 1.0 | 1.0 | 1.0 |
| **decoder_factor** | 0.01 | 0.01 | 0.01 | 0.01 |
| **binding_factor** | - | - | 0.5 | - |
| **遮蔽策略** | 随机80% | Motif指定 | Motif指定 | Motif指定 |
| **学习率** | 1e-4 | 1e-4 | 5e-5 | 5e-5 |
| **warmup_init_lr** | 5e-5 | 5e-5 | 1e-5 | 1e-5 |
| **warmup_updates** | 4000 | 4000 | 4000 | 4000 |
| **clip_norm** | 1.5 | 1.5 | 0.0001 | 0.0001 |
| **max_tokens** | 1024 | 1024 | 800 | 1024 |
| **max_sentences** | 不限 | 不限 | 1 | 不限 |
| **max_update** | 1,000,000 | 1,000,000 | 1,000,000 | 300,000 |
| **max_epoch** | 10,000 | 10,000 | 2,000 | 50 |
| **GPU 数量** | 8 | 8 | 8 | 1 |
| **初始权重** | ESM2 预训练 | Stage 1 best | Stage 2 best | Stage 3 best |
| **encoder_embed_dim** | 1280 | 1280 | 1280 | 1280 |
| **EGNN mode** | rm-node | rm-node | rm-node | rm-node |
| **EGNN layers** | 3 | 3 | 3 | 3 |
| **KNN** | 30 | 30 | 30 | 30 |
| **Dropout** | 0.3 | 0.3 | 0.3 | 0.3 |
| **Optimizer** | Adam(0.9,0.98) | Adam(0.9,0.98) | Adam(0.9,0.98) | Adam(0.9,0.98) |
| **LR Scheduler** | inverse_sqrt | inverse_sqrt | inverse_sqrt | inverse_sqrt |

### 38.1 超参数选择的设计原则

上表中的超参数并非随意选择，每个数值背后都有明确的设计原则。以下逐一解释关键超参数在不同阶段变化的原因：

**（1）批次大小（Batch Size）逐阶段减小：512 → 128 → 64**

```
Stage 1 (MLM)：max_tokens=1024，不限 max_sentences
  有效批次大小约 512 条序列（8 GPU × 大约 64 条/GPU）
  
  原因：
  - Stage 1 只有序列+结构损失，每个样本的显存占用较小
  - 大批次能提供更稳定的梯度估计
  - 数据量最大（百万级），大批次可以更快遍历数据

Stage 2 (Motif)：max_tokens=1024，不限 max_sentences
  有效批次大小约 128 条序列
  
  原因：
  - Motif 遮蔽可能导致更多的填充（padding），实际序列更长
  - 与 Stage 1 类似，但遮蔽模式变化导致实际装载量略减

Stage 3 (Full)：max_tokens=800，max_sentences=1
  有效批次大小约 8 条序列（8 GPU × 1 条/GPU）
  
  原因：
  - 新增了配体数据（ligand_atom, ligand_coor），每个样本的显存大幅增加
  - SubstrateEGNN 需要额外的中间变量存储空间
  - 配体的 KNN 图构建需要额外内存
  - max_sentences=1 确保每个 GPU 只处理一条数据，避免 OOM（内存溢出）

核心规律：
  后期阶段的模型更复杂、数据更丰富 → 每个样本占用更多显存
  → 必须减小批次大小来适应 GPU 显存限制
```

**（2）学习率逐阶段降低：1e-4 → 1e-4 → 5e-5**

```
Stage 1: lr=1e-4
  模型需要从 ESM2 的蛋白质语言模型快速转变为蛋白质设计模型。
  较高的学习率帮助模型快速探索参数空间。

Stage 2: lr=1e-4（保持不变）
  虽然是从 Stage 1 的权重继续训练，但遮蔽策略的根本性改变
  等同于一个新任务。保持较高学习率让模型快速适应
  从"随机填空"到"Motif-Scaffolding"的转变。

Stage 3: lr=5e-5（降低一半）
  两个原因要求降低学习率：
  
  原因 A：保护已有知识
    Stage 1-2 已经花了大量计算资源学到的序列-结构协同设计能力
    不能被 Stage 3 的新目标（配体结合）所破坏。
    较低的学习率确保参数更新幅度小，已有知识得到保留。
  
  原因 B：新模块的稳定训练
    SubstrateEGNN 是随机初始化的新模块，它的梯度可能比
    已经收敛的旧模块大得多。较低的全局学习率 + 严格的梯度裁剪
    确保新旧模块之间的梯度量级平衡。

核心规律：
  训练越到后期 → 模型已经编码了越多的有价值知识
  → 学习率必须越低，以"精细调整"而非"大幅改写"已有参数
```

**（3）Warmup 步数与目标复杂度的关系**

```
所有阶段统一使用 warmup_updates=4000：
  warmup_init_lr: Stage 1-2 为 5e-5，Stage 3 为 1e-5

Warmup 的作用：
  训练开始时，模型参数（尤其是新增模块）可能处于不稳定状态。
  如果直接使用目标学习率，大梯度可能导致训练发散。
  Warmup 阶段从极低的学习率开始，逐步增加到目标值，
  给模型一个"热身"的机会。

为什么 Stage 3 的 warmup_init_lr 更低（1e-5 vs 5e-5）？
  Stage 3 新增了 SubstrateEGNN 模块，其参数是随机初始化的。
  随机初始化的参数产生的梯度方向可能是错误的。
  更低的初始学习率确保在这些梯度方向被修正之前，
  参数更新量极小，不会破坏已有知识。

Warmup 期间的学习率变化：
  Stage 1-2: 5e-5 → (线性增长 4000 步) → 1e-4
  Stage 3:   1e-5 → (线性增长 4000 步) → 5e-5
```

**（4）总更新步数随任务复杂度增加**

```
Stage 1: max_update=1,000,000，max_epoch=10,000
  任务：从 ESM2 学习序列恢复 + 从零学习结构预测
  数据量：最大（UniRef + PDB 全集）
  需要大量步数来充分学习蛋白质语言的基础知识。

Stage 2: max_update=1,000,000，max_epoch=10,000
  任务：从随机遮蔽切换到 Motif 遮蔽
  虽然任务变化不大，但模型需要充分适应新的遮蔽模式，
  因此保持相同的步数上限。

Stage 3: max_update=1,000,000，max_epoch=2,000
  任务：新增配体结合预测
  epoch 上限降低（2000 vs 10000），因为 Stage 3 的主要目标
  是在已有基础上增加配体约束，不需要像 Stage 1 那样从零开始。
  通常在 200k-500k 步内收敛。

微调: max_update=300,000，max_epoch=50
  数据量最小，50 个 epoch 可能只需要几千到几万步。
  max_update=300,000 是安全上限，实际通常在 epoch 15-30 早停。

核心规律：
  数据量越大、任务越复杂 → 需要更多的更新步数
  从通用到特定 → 步数递减（因为站在前一阶段的肩膀上）
```

---

## 39. 三种解码策略：Greedy、Top-K、Top-P

### 39.1 Greedy（贪心）

```python
# 来自 geometric_protein_design.py (line 292-295)
encoder_out[:, 1: -1, : 4] = -math.inf  # 排除特殊 token（BOS/PAD/EOS/UNK）
encoder_out[:, :, 24:] = -math.inf       # 排除非标准氨基酸（B/U/Z/O/X/MASK等）
indexes = torch.argmax(encoder_out, dim=-1)  # 每个位置选概率最高的氨基酸
```

```
操作：
  原始概率分布: [0.01, 0.01, 0.01, 0.01, 0.15, 0.25, 0.10, ...]
                 ↑cls  ↑pad  ↑eos  ↑unk  ↑L    ↑A    ↑G
  排除后:       [-inf, -inf, -inf, -inf, 0.15, 0.25, 0.10, ..., -inf, -inf]
  argmax:       → 选择 A（索引 5，概率最高）

优点：确定性，可重复
缺点：多样性低，可能陷入局部最优
```

### 39.2 Top-K 采样

```python
# 来自 geometric_protein_design.py (line 286-289)
_, top_indices = torch.topk(encoder_out, k=3, dim=-1)
# 取概率最高的 3 个氨基酸的索引

index_selects = torch.tensor(
    np.random.randint(low=0, high=3, size=(B, L))
).unsqueeze(-1)
# 从 top-3 中随机选一个

indexes = top_indices.gather(index=index_selects, dim=-1).squeeze(-1)
```

```
操作：
  概率分布: [..., A:0.25, L:0.15, G:0.10, V:0.08, ...]
  top-3:    [A, L, G]
  随机选择: → 可能是 A, L, 或 G（等概率）

优点：有多样性，但不会选择低概率的氨基酸
缺点：固定 K 值，不自适应
```

### 39.3 Top-P（Nucleus）采样

```python
# 来自 geometric_protein_design.py (line 296-308)
# 先排除特殊 token
encoder_out[:, 1: -1, : 4] = 0
encoder_out[:, :, 24:] = 0

# 按概率降序排列
sorted_logits, sorted_indices = torch.sort(encoder_out, descending=True)

# 计算累积概率
cumulative_probs = torch.cumsum(sorted_logits, dim=-1)

# 截断：只保留累积概率 ≤ p 的 token
sorted_indices_to_remove = cumulative_probs > topp_probability  # 默认 p=0.2
sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
sorted_indices_to_remove[..., 0] = 0  # 始终保留第一个

# 将截断的 token 概率置零
indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
encoder_out[indices_to_remove] = 0

# 从剩余 token 中按概率采样
indexes = torch.multinomial(encoder_out.view(-1, encoder_out.size(-1)), 1).reshape(B, -1)
```

```
操作（p=0.4 示例）：
  排序后概率: [A:0.25, L:0.15, G:0.10, V:0.08, ...]
  累积概率:   [0.25,   0.40,   0.50,   0.58,   ...]
  截断(p=0.4): [A:0.25, L:0.15, 0,      0,      ...]
  归一化采样: A 以 62.5% 概率，L 以 37.5% 概率被选中

优点：自适应——高信心时只选1-2个，低信心时选更多
ChlR 微调使用 p=0.4 进行推理
```

### 39.4 三种策略的数值对比

为了直观理解三种解码策略的差异，下面用同一个具体的概率分布进行对比分析。

假设模型对某个位置输出了以下 softmax 概率分布（只显示概率 >0 的标准氨基酸）：

```
原始概率分布（已排除特殊 token 和非标准氨基酸）：
  A (Ala) = 0.35
  V (Val) = 0.25
  L (Leu) = 0.15
  I (Ile) = 0.10
  M (Met) = 0.05
  其余 15 种氨基酸合计 = 0.10
  ─────────────────────
  总计 = 1.00
```

**（1）Greedy 解码：**

```
操作：直接选择概率最高的氨基酸
  argmax → A (概率 0.35)

结果：
  永远选择 A（丙氨酸）
  
特点：
  - 完全确定性：运行 100 次，100 次都选 A
  - 没有任何多样性
  - 如果 A 确实是最佳选择，效果最好
  - 如果 A 不是最佳选择（概率偶然偏高），无法自我修正
  
适用场景：
  基准测试（benchmark）——需要可重复的结果进行公平比较
```

**（2）Top-K 解码（K=3）：**

```
操作：
  Step 1: 选出概率最高的 K=3 个氨基酸
    → {A: 0.35, V: 0.25, L: 0.15}
  
  Step 2: 重新归一化概率
    总和 = 0.35 + 0.25 + 0.15 = 0.75
    A: 0.35/0.75 = 0.467  (46.7%)
    V: 0.25/0.75 = 0.333  (33.3%)
    L: 0.15/0.75 = 0.200  (20.0%)
  
  Step 3: 从归一化后的分布中采样
    → 可能选择 A（46.7%）、V（33.3%）或 L（20.0%）

结果（运行 1000 次的近似统计）：
  A 被选中约 467 次
  V 被选中约 333 次
  L 被选中约 200 次
  I, M 等永远不会被选中（被排除在 top-3 之外）

特点：
  - 有适度的多样性，但被严格限制在 top-K 个选项内
  - 不会选择低概率的"冒险"选项
  - K 值是固定的，不会根据概率分布自适应调整
  
问题：
  如果某个位置的概率分布非常集中（如 A=0.95, 其余都很低），
  K=3 仍然会保留 3 个选项，人为引入不必要的噪声。
  如果概率分布非常均匀（如 top-10 都在 0.08-0.12 之间），
  K=3 只保留 3 个选项，丢弃了很多合理的候选。
```

**（3）Top-P 解码（P=0.8）：**

```
操作：
  Step 1: 按概率降序排列
    A: 0.35, V: 0.25, L: 0.15, I: 0.10, M: 0.05, ...
  
  Step 2: 计算累积概率
    A:          0.35  (累积: 0.35)
    A+V:        0.60  (累积: 0.60)
    A+V+L:      0.75  (累积: 0.75)
    A+V+L+I:    0.85  (累积: 0.85) ← 首次超过 P=0.8
    A+V+L+I+M:  0.90  (累积: 0.90)
  
  Step 3: 保留累积概率达到 P=0.8 时的所有 token
    → 保留 {A, V, L, I}（4 个氨基酸，累积概率 0.85 > 0.8）
    
    注意：保留到 I 是因为加入 I 时才首次超过阈值 0.8。
    根据实现方式，阈值刚好等于 0.8 的边界 token 也被保留。
  
  Step 4: 重新归一化
    总和 = 0.35 + 0.25 + 0.15 + 0.10 = 0.85
    A: 0.35/0.85 = 0.412  (41.2%)
    V: 0.25/0.85 = 0.294  (29.4%)
    L: 0.15/0.85 = 0.176  (17.6%)
    I: 0.10/0.85 = 0.118  (11.8%)
  
  Step 5: 从归一化后的分布中采样

结果（运行 1000 次的近似统计）：
  A 被选中约 412 次
  V 被选中约 294 次
  L 被选中约 176 次
  I 被选中约 118 次
  M 及其余永远不会被选中

与 Top-K (K=3) 的对比：
  Top-P 保留了 4 个选项（vs Top-K 的 3 个），分布更宽
  Top-P 中 I 有 11.8% 的概率被选中，而 Top-K 中 I 被完全排除
  Top-P 自适应：如果概率集中，保留的选项更少；如果概率分散，保留更多
```

```
三种策略的综合对比：

                 Greedy      Top-K(K=3)    Top-P(P=0.8)
  ──────────────────────────────────────────────────────
  选择 A 的概率   100%        46.7%         41.2%
  选择 V 的概率     0%        33.3%         29.4%
  选择 L 的概率     0%        20.0%         17.6%
  选择 I 的概率     0%         0%           11.8%
  选择 M 的概率     0%         0%            0%
  ──────────────────────────────────────────────────────
  候选集大小        1           3             4
  多样性            无          中等          较高
  确定性            完全确定    部分随机      部分随机
  自适应性          N/A         固定 K        自适应
```

### 39.5 实际使用建议

不同的应用场景应选择不同的解码策略。以下是基于实际经验的具体建议：

**场景 1：基准测试和论文复现 → 使用 Greedy**

```
原因：
  - 完全确定性，保证结果可重复
  - 消除随机性的干扰，公平对比不同模型
  - 论文中报告的数字需要可验证

使用方式：
  --decoding-strategy "greedy"

注意：
  Greedy 的结果代表模型的"最自信"的设计方案。
  如果 Greedy 的 ESP 评分已经很高，说明模型对该设计任务有强信心。
  如果 Greedy 的 ESP 评分低，可能需要通过采样策略探索更多候选。
```

**场景 2：酶设计实际应用 → 使用 Top-P（p=0.85-0.95）**

```
原因：
  - 酶设计需要多样性：一次生成多个候选序列，然后实验筛选
  - Top-P 的自适应特性很重要：
    在高信心位置（催化关键残基附近）自动减少候选数量
    在低信心位置（远离活性位点的 loop 区域）自动增加多样性
  - p=0.85-0.95 是多样性和质量的良好平衡点

推荐参数：
  p=0.85：中等多样性，每个位置通常保留 2-5 个候选氨基酸
  p=0.90：较高多样性，适合需要大量候选的早期筛选阶段
  p=0.95：高多样性，可能引入一些低质量的选择

实际操作：
  1. 使用 p=0.90 生成 100 个候选序列
  2. 用 ESP 评分排序
  3. 选择 ESP top-10 进入实验验证

注意：
  EnzyGen2 源码中 ChlR 微调使用 p=0.4（非常保守）。
  但对于通用酶设计，p=0.85-0.95 通常能产生更好的候选多样性。
```

**场景 3：受限设计（仅允许特定氨基酸类型）→ 使用 Top-K（K=3-5）**

```
原因：
  - 某些设计场景中，特定位置只允许少数氨基酸类型
    例如：疏水核心只允许 A/V/L/I/F
    例如：盐桥位置只允许 D/E/K/R
  - Top-K 自然地将选择限制在概率最高的 K 个候选中
  - 不需要关心概率分布的具体形状

推荐参数：
  K=3：非常保守，每个位置最多 3 种选择
  K=5：适度保守，适合大多数受限设计场景
  
  注意：EnzyGen2 源码中 Top-K 实现使用均匀随机采样
  （等概率从 top-K 中选一个），而非按概率加权采样。
  这意味着 K=3 时，排名第 3 的氨基酸有 33.3% 的概率被选中，
  即使它的原始概率远低于排名第 1 的。

实际操作：
  如果需要按概率加权的 Top-K 采样，建议修改源码：
  将 np.random.randint(low=0, high=K, size=...) 
  替换为 torch.multinomial(renormalized_probs, 1)
```

```
选择决策树：

需要可重复结果？
  ├── 是 → Greedy
  └── 否 → 需要位置级别的约束？
            ├── 是 → Top-K (K=3-5)
            └── 否 → Top-P (p=0.85-0.95)
```

---

## 40. 生成流水线完整追踪

### 40.1 推理脚本

```bash
# generate_enzygen2_pretrain.sh

proteins=(10665 11698 9796 11706 573 ...)  # 26 个物种

for element in ${proteins[@]}; do
  python3 fairseq_cli/generate.py data/protein_ligand_enzyme_test.json \
    --task geometric_protein_design \
    --protein-task ${element} \               # 指定物种
    --path models/EnzyGen2/checkpoint_best.pt \
    --batch-size 1 \
    --results-path output/EnzyGen2/${element} \
    --generation \                             # 启用生成模式
    --decoding-strategy "greedy"               # 贪心解码
done
```

### 40.2 生成流程

```python
# 来自 generate.py 和 geometric_protein_design.py 的 valid_step()

for sample in test_dataset:
    # Step 1: 前向传播
    encoder_out, coords = model(
        src_tokens, src_lengths, target_coor, motif, ncbi
    )
    # encoder_out: [1, L, 33]  序列概率
    # coords: [1, L, 3]        预测坐标
    
    # Step 2: 解码
    indexes = decode(encoder_out, strategy="greedy")  # [1, L]
    
    # Step 3: 组合遮蔽和已知位置
    indexes = output_mask * indexes + (output_mask == 0) * src_tokens
    # 遮蔽位置用预测结果，已知位置保留原始序列
    
    # Step 4: 坐标处理
    coords = output_mask.unsqueeze(-1) * coords + \
             (output_mask.unsqueeze(-1) == 0) * target_coor
    # 遮蔽位置用预测坐标，已知位置保留原始坐标
    coords = coords[:, 1:-1, :] + centers.unsqueeze(1)
    # 去掉 BOS/EOS，加回中心坐标
    
    # Step 5: 计算 RMSD
    rmsd = torch.sqrt(
        torch.sum(torch.square(coords - target_coor) * output_mask) / output_mask.sum()
    )
    
    # Step 6: 转为字符串和 PDB
    string = alphabet.string(indexes[0])  # "MKVAVLGAAGG..."
```

### 40.3 输出文件

```
output/EnzyGen2/10665/
├── protein.txt          ← 设计的蛋白质序列（每行一条）
├── src.seq.txt          ← 原始参考序列
├── pdb.txt              ← PDB ID
├── log_likelihood.txt   ← 每个残基的对数概率
├── pred_pdbs/           ← 预测结构的 PDB 文件
│   ├── 0.pdb
│   ├── 1.pdb
│   └── ...
└── tgt_pdbs/            ← 参考结构的 PDB 文件
    ├── 0.pdb
    └── ...
```

### 40.4 生成流程的完整数值追踪

为了彻底理解生成过程中每一步的数据变化，下面用一个具体的数值示例进行完整追踪。

**问题设定：设计一个 100 残基的酶，其中 30 个残基是已知的 Motif，70 个残基需要设计。**

```
═══════════════════════════════════════════════════════════════════
  步骤 0：准备输入数据
═══════════════════════════════════════════════════════════════════

  已知信息：
    Motif 序列（30 个残基）：例如 "MKVAVLGAAGGIGQALAL..." 
    Motif 坐标（30 个 Cα 位置）：每个残基一个 (x, y, z)
    Scaffold 位置（70 个残基）：序列未知，坐标未知
  
  构造 src_tokens（序列输入）：
    [0, aa1, aa2, ..., aa30, 32, 32, ..., 32, 2]
     ↑                       ↑ (70个MASK)         ↑
    BOS  30个已知残基token     70个MASK token      EOS
    
    总长度 = 1 + 30 + 70 + 1 = 102 个 token
    
    其中：
      0 = BOS (beginning of sequence) 特殊 token
      2 = EOS (end of sequence) 特殊 token  
      32 = MASK 特殊 token
      aa1-aa30 = 4-23 范围内的标准氨基酸索引
    
    张量形状：src_tokens = [1, 102]（batch_size=1）

  构造 target_coor（坐标输入）：
    Motif 位置（30个）：使用真实的 Cα 坐标（从 PDB 结构获取）
    Scaffold 位置（70个）：使用球面随机游走初始化
      从最后一个已知坐标出发，每步 3.75Å 随机方向
    
    中心化：减去所有坐标的均值 → 质心在原点
    centers = mean(all_coords)  → [1, 3]（保存，后续加回）
    target_coor -= centers     → [1, 102, 3]

  构造遮蔽掩码：
    input_mask:  [0, 0, ..., 0, 1, 1, ..., 1, 0]  (30个0 + 70个1)
    output_mask: [0, 0, ..., 0, 1, 1, ..., 1, 0]  (相同)
    形状：[1, 102]

═══════════════════════════════════════════════════════════════════
  步骤 1：前向传播
═══════════════════════════════════════════════════════════════════

  model.forward(src_tokens, src_lengths, target_coor, motif, ncbi)
  
  内部流程：
    (a) Token 嵌入: [1, 102] → [1, 102, 1280]
    (b) NCBI 嵌入加法: + [1, 1, 1280]（广播到所有位置）
    (c) 33 层 Transformer + 3 层 EGNN（交错执行）
    (d) 序列预测头: [1, 102, 1280] → [1, 102, 33]
    (e) 坐标输出: [1, 102, 3]
  
  输出：
    encoder_out: [1, 102, 33]  — 每个位置上 33 个 token 的概率分布
    coords:      [1, 102, 3]   — 每个位置的预测 Cα 坐标

═══════════════════════════════════════════════════════════════════
  步骤 2：解码（以 Top-P, p=0.9 为例）
═══════════════════════════════════════════════════════════════════

  对 encoder_out 的 102 个位置：
    位置 0 (BOS): 忽略（特殊 token）
    位置 1-30 (Motif): 预测结果被忽略，保留原始序列
    位置 31-100 (Scaffold): 对每个位置应用 Top-P 采样
    位置 101 (EOS): 忽略（特殊 token）
  
  对于某个 Scaffold 位置（例如位置 45）：
    encoder_out[0, 45, :] = [概率分布，33个值]
    排除特殊 token (索引 0-3) 和非标准氨基酸 (索引 24+)
    在剩余的 20 个标准氨基酸中进行 Top-P 采样
    → 选出一个氨基酸，如 V (Val, 索引 8)
  
  对所有 70 个 Scaffold 位置重复此过程。

═══════════════════════════════════════════════════════════════════
  步骤 3：组合已知和预测
═══════════════════════════════════════════════════════════════════

  序列组合：
    indexes = output_mask * predicted_indexes + (1 - output_mask) * src_tokens
    
    Motif 位置：保留原始的 aa1, ..., aa30
    Scaffold 位置：使用 Step 2 中采样的结果
    
    最终序列：[1, 102] → 去掉 BOS/EOS → [1, 100]
    转为字符串："MKVAVLGAAGG...VWYLKNNF"（100 个氨基酸字母）

  坐标组合：
    coords = output_mask * predicted_coords + (1 - output_mask) * target_coor
    
    Motif 位置：保留原始 Cα 坐标
    Scaffold 位置：使用模型预测的坐标
    
    加回中心坐标：coords += centers
    去掉 BOS/EOS：coords = coords[:, 1:-1, :]
    
    最终坐标：[1, 100, 3]（100 个残基的 Cα 位置）

═══════════════════════════════════════════════════════════════════
  步骤 4：输出
═══════════════════════════════════════════════════════════════════

  最终产物：
    序列：100 个氨基酸的字符串
    坐标：100 × 3 的 Cα 坐标矩阵
    
  保存为：
    protein.txt  ← 序列字符串
    pred_pdbs/0.pdb ← 由坐标生成的 PDB 文件
    log_likelihood.txt ← 每个位置的预测概率（用于质量评估）
  
  性能：
    在 V100 GPU 上，上述全部流程耗时约 1-2 秒
    其中：
      前向传播 ≈ 0.5-1.0 秒（主要耗时）
      KNN 图构建 ≈ 0.1-0.3 秒（CPU 上运行）
      解码+后处理 ≈ 0.1 秒
    
    批量生成时，可以进一步利用 GPU 并行：
      100 个候选序列 ≈ 2-3 分钟（batch 处理）
```

---

## 41. PDB 文件生成

### 41.1 从坐标到 PDB

```python
# 来自 generate_pdb_file.py

def save_bb_as_pdb(bb_positions, residues, chain, fn):
    """
    bb_positions: [seq_len, 3]  ← Cα 坐标
    residues: "MKVAVL..."       ← 氨基酸序列
    chain: "A"                  ← 链 ID
    fn: "output.pdb"            ← 输出文件名
    """
    prot = create_bb_prot(bb_positions, residues, chain)
    pdb_string = to_pdb(prot)
    with open(fn, 'w') as f:
        f.write(pdb_string)
```

### 41.2 PDB 格式

```
生成的 PDB 文件只包含 Cα 原子（backbone only）：

ATOM      1  CA  MET A   1       1.200   3.400   5.600  1.00  0.00           C
ATOM      2  CA  LYS A   2       7.800   9.000   1.100  1.00  0.00           C
ATOM      3  CA  VAL A   3       2.300   4.500   6.700  1.00  0.00           C
...

这种简化的 PDB 文件可以：
  - 在 PyMOL/ChimeraX 中可视化
  - 用 AlphaFold2 进一步预测全原子结构
  - 作为 Rosetta 的输入进行细化设计
```

### 41.3 从 Cα 到完整骨架的重建

EnzyGen2 只预测 Cα（α碳）坐标，但实际的蛋白质骨架包含四种主链原子：N（氨基氮）、Cα（α碳）、C（羰基碳）、O（羰基氧）。要生成可用于下游分析的完整 PDB 文件，需要从 Cα 坐标重建其余原子的位置。

```
蛋白质骨架的化学结构（每个残基重复单元）：

    O
    ‖
  —N—Cα—C—N—Cα—C—N—Cα—C—
   |     |       |     |       |     |
   H    R₁      H    R₂      H    R₃
   
  其中 R₁, R₂, R₃ 是各残基的侧链

标准键长和键角（理想值）：
  N—Cα 键长:   1.47 Å
  Cα—C 键长:   1.52 Å
  C—N 键长:    1.33 Å（肽键，部分双键性质）
  C=O 键长:    1.24 Å
  
  N—Cα—C 键角:  111.0°
  Cα—C—N 键角:  116.6°
  C—N—Cα 键角:  121.7°
```

**从 Cα 坐标重建主链原子的方法：**

```
给定连续三个 Cα 坐标：Cα(i-1), Cα(i), Cα(i+1)

Step 1：确定链方向向量
  forward = normalize(Cα(i+1) - Cα(i))     # 指向下一个残基
  backward = normalize(Cα(i-1) - Cα(i))    # 指向上一个残基

Step 2：放置 N 原子
  N 位于 Cα 和前一个残基之间
  N(i) = Cα(i) + 1.47Å × backward
  
  即：N 在 Cα 沿链反方向 1.47Å 处

Step 3：放置 C 原子
  C 位于 Cα 和后一个残基之间
  C(i) = Cα(i) + 1.52Å × forward
  
  即：C 在 Cα 沿链正方向 1.52Å 处

Step 4：放置 O 原子
  O 与 C 相连，垂直于 C—N 肽键平面
  
  计算肽键平面的法向量：
    plane_normal = normalize(cross(forward, backward))
  
  O(i) = C(i) + 1.24Å × plane_normal
  
  即：O 在 C 的位置，沿垂直于主链平面的方向偏移 1.24Å

特殊处理：
  第一个残基（i=0）：没有 Cα(i-1)，使用 forward 方向的反向
  最后一个残基（i=N-1）：没有 Cα(i+1)，使用 backward 方向的反向
```

**重要注意事项：**

```
上述方法给出的是近似骨架坐标。存在以下限制：

1. 理想几何 vs 真实几何
   真实蛋白质中的键长和键角会因为局部环境（二级结构、
   侧链相互作用、溶剂效应）而偏离理想值。
   α螺旋中的 φ/ψ 角与 β折叠中的完全不同。
   仅用理想值重建会在局部引入几何偏差。

2. ω 角（肽键平面）
   理想重建假设 ω = 180°（反式肽键），但 Pro 残基
   前的肽键可能是顺式（ω ≈ 0°），这在近似重建中无法处理。

3. 缺少侧链原子
   上述方法只重建主链原子（N, Cα, C, O），
   不包含侧链原子。完整的全原子结构需要后续工具处理。

推荐的后处理流水线：

  EnzyGen2 输出（仅 Cα 坐标）
       │
       ↓
  近似骨架重建（N, Cα, C, O — 使用理想几何）
       │
       ↓
  结构精细化（选择以下方法之一）：
  
  方法 A：Rosetta FastRelax
    输入：近似骨架 PDB
    操作：能量最小化 + 侧链堆积优化
    输出：全原子结构（含侧链）
    耗时：约 5-30 分钟/结构
    
  方法 B：AlphaFold2 / ESMFold
    输入：设计的序列（不需要初始结构）
    操作：从序列预测全原子结构
    输出：高精度全原子结构
    耗时：约 1-5 分钟/结构
    优势：独立验证——如果 AF2 预测的结构与 EnzyGen2 的
          Cα 坐标一致（RMSD < 2Å），说明设计方案的可折叠性好
    
  方法 C：OpenMM 能量最小化
    输入：近似骨架 PDB + 力场参数
    操作：分子力学能量最小化
    输出：能量最小化后的结构
    耗时：约 1-10 分钟/结构
    适用：需要物理合理的键长/键角时
```

---

## 42. ESP 评估指标

### 42.1 什么是 ESP

**ESP（Enzyme-Substrate Prediction）** 是评估设计的酶蛋白是否能结合目标底物的指标。

### 42.2 评估流程

```python
# 来自 prepare_esp_evaluation_pretrain.py

# 1. 读取设计的蛋白质序列
proteins = open("protein.txt").readlines()

# 2. 读取对应的 PDB ID
pdbs = open("pdb.txt").readlines()

# 3. 根据 PDB-EC 映射找到酶分类
pdb2ec = json.load(open("protein_ligand_enzyme_test_pdb2ec.json"))

# 4. 根据 EC 分类找到底物 SMILES
ec2smiles = {
    "2.7.7.7": "C00677",          # DNA聚合酶底物
    "3.2.1.17": "C04628",         # 溶菌酶底物
    "1.1.1.1": "C00003",          # 乙醇脱氢酶底物（NAD+）
    ...
}

# 5. 输出：(蛋白质序列, 底物SMILES) 对
fw.write(protein + " " + ec2smiles[pdb2ec[pdb]])
```

### 42.3 ESP 评分

ESP 评分使用 Alexander Kroll 的工具（https://github.com/AlexanderKroll/ESP_prediction_function），输入是 `(蛋白质序列, 底物表示)` 对，输出是预测的酶-底物结合概率。

### 42.4 ESP 评分的具体含义

ESP（Enzyme-Substrate Pair）评分是评估设计的酶与目标底物匹配程度的核心指标。以下详细解释其含义、数值范围和实际应用。

**ESP 评分衡量的是什么？**

```
ESP 评分回答一个核心问题：
  "这个蛋白质序列能否催化与该底物相关的化学反应？"

它不是直接测量结合亲和力（如 Kd 或 Ki），
而是预测一个蛋白质是否具有针对特定底物的酶活性。

本质上，ESP 是一个经过训练的分类器：
  输入：(蛋白质序列, 底物分子表示)
  输出：概率值 ∈ [0, 1]
  
  高 ESP → 模型认为该蛋白质很可能催化该底物的反应
  低 ESP → 模型认为该蛋白质不太可能具有针对该底物的酶活性
```

**典型 ESP 数值范围和含义：**

```
═══════════════════════════════════════════════════════════════
  ESP 范围        含义                     典型来源
═══════════════════════════════════════════════════════════════
  0.0 - 0.1      几乎不可能具有酶活性       完全随机的蛋白质序列
  0.1 - 0.2      极不可能具有酶活性         随机蛋白质/错误的底物配对
  0.2 - 0.4      低可能性                   弱设计或部分相关的蛋白质
  0.4 - 0.5      不确定                     边界情况，需要实验验证
  0.5 - 0.6      中等可能性                 初步设计的酶候选
  0.6 - 0.7      较高可能性                 质量较好的设计酶
  0.7 - 0.8      高可能性                   天然酶或优秀设计
  0.8 - 0.9      很高可能性                 高活性天然酶
  0.9 - 1.0      极高可能性                 高度特异性的天然酶
═══════════════════════════════════════════════════════════════

基准参考值：
  随机蛋白质序列 + 随机底物 → ESP ≈ 0.1 - 0.2
  天然酶 + 其已知底物        → ESP ≈ 0.7 - 0.9
  设计良好的酶 + 目标底物    → ESP ≈ 0.5 - 0.8
  天然酶 + 错误底物          → ESP ≈ 0.1 - 0.3
```

**ESP 评分的计算流程：**

```
ESP 预测模型是一个独立的预训练模型（与 EnzyGen2 分开），
其工作流程如下：

Step 1：蛋白质序列编码
  输入序列 → ESM-1b 或类似的蛋白质语言模型 → 序列嵌入向量
  
Step 2：底物分子编码
  底物 SMILES → 分子指纹（Morgan fingerprint）或图神经网络 → 分子嵌入向量

Step 3：配对预测
  concat(序列嵌入, 分子嵌入) → 全连接网络 → sigmoid → ESP 分数

Step 4：输出
  ESP ∈ [0, 1]
```

**如何使用 ESP 评分筛选设计结果：**

```
推荐的筛选流程：

  1. 用 EnzyGen2 生成 N 个候选序列（例如 N=100）
  2. 对每个候选序列计算 ESP 评分
  3. 按 ESP 评分降序排列
  4. 选择 ESP > 0.5 的候选进入下一轮评估
  5. 对 ESP top-10 的候选进行更详细的计算评估：
     - AlphaFold2 结构预测
     - 分子对接（Molecular Docking）
     - 分子动力学模拟
  6. 选择 top-3 的候选进入实验验证

注意事项：
  - ESP 是预测值，不是实验测量值，存在假阳性和假阴性
  - ESP > 0.5 不保证实验中真的有酶活性
  - ESP < 0.5 也不排除可能有酶活性
  - 建议结合其他指标（如 pLDDT、Rosetta 能量）综合评估
```

---

## 43. 关键张量形状速查表

| 张量 | 形状 | 含义 |
|------|------|------|
| src_tokens | [B, L] | 蛋白质序列（整数 token） |
| src_lengths | [B] | 序列长度 |
| target_coor | [B, L, 3] | 目标 Cα 坐标 |
| input_mask | [B, L] | 输入遮蔽掩码（1=遮蔽） |
| output_mask | [B, L] | 输出遮蔽掩码（1=需要预测） |
| ncbi | [B] | 物种 ID（整数索引） |
| ncbi_emb | [B, 1, 1280] | 物种嵌入（广播到所有位置） |
| x（embedding后） | [B, L, 1280] | Token 嵌入 |
| x（Transformer内） | [L, B, 1280] | T-first 格式 |
| x（EGNN内） | [B*L, 1280] | 展平格式 |
| coords（EGNN内） | [B*L, 3] | 展平坐标 |
| edges | [2, B*L*K] | KNN 图的边列表 |
| radial | [B*L*K, 1] | 邻居距离平方 |
| coord_diff | [B*L*K, 3] | 邻居坐标差 |
| edge_feat | [B*L*K, 1280] | 边消息 |
| encoder_prob | [B, L, 33] | 序列概率分布 |
| decoder_out | [B, L, 3] | 预测坐标 |
| protein_rep | [B, 1280] | 蛋白质全局表示 |
| substrate_atom | [B, N, 5] | 配体原子特征 |
| substrate_coor | [B, N, 3] | 配体原子坐标 |
| sub_feats | [B, 1280] | 配体全局表示 |
| scores | [B, 2] | 结合预测概率 |

---

## 44. 关键数字速查

```
模型参数：
  ESM2 参数量：~650M
  EGNN 参数量（3层 rm-node）：~20M（估算）
  SubstrateEGNN 参数量：~20M（估算）
  NCBI Embedding：10000 × 1280 = ~12.8M 参数
  总参数量：约 700M

词表大小：33 个 token
氨基酸种类：20 种标准 + 若干特殊
隐藏维度：1280
FFN 维度：5120（= 4 × 1280）
注意力头数：20
Transformer 层数：33
EGNN 层数：3
KNN 邻居数：30
NCBI 物种数：10000 个槽位

Cα-Cα 平均距离：3.75-3.8Å
坐标初始化步长：3.75Å
Token Dropout 比例：12%（= 0.15 × 0.8）

训练规模：
  预训练数据：PDB + SwissProt（多物种多酶家族）
  预训练 GPU：8 × GPU
  微调 GPU：1 × GPU
  最大序列长度：1024 tokens
  最大更新步数：1,000,000（预训练）/ 300,000（微调）

测试物种数：26 个（预训练评估）
微调酶家族：3 个（ChlR, AadA, TPMT）
```

---

## 45. 常见误区（初学者必读）

### 误区 1：EnzyGen2 只做序列设计

**错误**：EnzyGen2 同时设计序列和三维结构（Cα 骨架坐标），这是"协同设计"的核心含义。

### 误区 2：EGNN 是独立的解码器

**错误**：EGNN 不是在 Transformer 之后独立运行的。它**交错在 Transformer 层之间**（每 11 层 Transformer 后执行一次 EGNN），实现序列和结构信息的双向交流。

### 误区 3：所有残基都被重新设计

**错误**：Motif 位点（功能基序）的序列和坐标是固定不变的。模型只设计非 Motif 位置。这确保了关键功能位点的保守性。

### 误区 4：配体结合预测是必须的

**错误**：配体结合预测只在 Stage 3 预训练中使用。微调和推理时使用基础模型（无配体模块）。配体约束的知识已经通过预训练编码到模型权重中。

### 误区 5：ESM2 的预训练权重被冻结

**错误**：ESM2 的权重在 EnzyGen2 训练过程中是**可更新的**（不是冻结的）。整个模型端到端训练。

### 误区 6：坐标输出是全原子的

**错误**：EnzyGen2 只输出 Cα（α碳）骨架坐标，不是全原子坐标。每个残基只有一个 (x, y, z) 点。要获得全原子结构，需要后续使用 Rosetta 或 AlphaFold2 进行全原子建模。

### 误区 7：推理时随机初始化坐标与训练时相同

**部分正确**：训练和推理时都对被遮蔽位置使用球面随机游走初始化（3.75Å步长），但推理时只执行一次前向传播，而训练时有多次更新步。

### 误区 8：三阶段预训练可以跳过

**不建议**：三阶段是渐进式设计——MLM 学基础语言、Motif 学设计范式、Full 学功能约束。跳过任何阶段都会降低模型性能。

### 误区 9：认为 EnzyGen2 可以设计任意长度的蛋白质

**错误**：EnzyGen2 的序列处理能力受限于底层 ESM2 模型的最大上下文长度。

```
ESM2 的位置编码最多支持约 1024 个 token。
减去 BOS 和 EOS 两个特殊 token，实际可处理的蛋白质长度上限约 1022 个残基。

在实际操作中：
  大多数酶的长度在 200-500 个残基之间 → 完全在限制范围内
  少数大型酶复合物可能超过 1000 个残基 → 无法直接处理

如果需要设计超长蛋白质，可能的解决方案：
  1. 将蛋白质分段设计，然后拼接（但会丢失段间的相互作用信息）
  2. 只设计关键的催化结构域（通常 <500 残基），其余部分保留天然序列
  3. 使用其他支持更长序列的模型（如基于扩散的方法）

即使在 1022 残基限制内，也要注意：
  - 更长的序列 → 更大的 KNN 图 → 更多的计算时间和内存
  - 序列长度 >800 时，可能需要减小 batch_size 避免 OOM
  - Stage 3 中配体数据进一步增加了显存需求，
    实际可处理的序列长度可能更短（约 600-800 残基）
```

### 误区 10：认为三层 EGNN 足以完成任意结构优化

**部分错误**：三层 EGNN 的结构优化能力是有限的，不能将坐标从任意初始位置移动到正确位置。

```
EGNN 的坐标更新机制回顾：
  每一层 EGNN 通过以下公式更新坐标：
    x_new = x_old + Σ(neighbor_j) (x_i - x_j) · φ(m_ij)
  
  其中 φ(m_ij) 的输出被 clamp 到一个有限范围（如 [-clamp, +clamp]）。
  
  这意味着每一层 EGNN 只能将每个原子的坐标移动有限的距离。

三层 EGNN 的累积移动能力：
  假设 clamp_distance ≈ 1.0（典型值），且每层的 K=30 个邻居
  贡献的平均位移约 0.5-1.0Å。
  
  3 层累积：每个 Cα 从初始位置最多移动约 3 × 1.0 = 3.0Å
  
  这意味着：
    如果初始坐标（随机游走）与真实坐标偏差 <3Å → EGNN 可以修正
    如果初始坐标与真实坐标偏差 >5Å → EGNN 可能无法完全修正

实际影响：
  对于靠近 Motif 的 Scaffold 残基：
    随机游走的起点是 Motif 的最后一个已知坐标，
    初始偏差通常较小（<5Å），EGNN 可以较好地修正。
  
  对于远离 Motif 的 Scaffold 残基：
    随机游走可能已经累积了较大的偏差（>10Å），
    EGNN 的 3 层更新可能不够。

为什么不用更多层 EGNN？
  1. 更多层 → 更多参数 → 更大的显存需求
  2. 过深的 EGNN 可能导致过平滑（over-smoothing）问题
  3. Transformer 层也在间接帮助坐标优化——通过序列特征
     携带的结构信息影响 EGNN 的输入

补偿机制：
  虽然 EGNN 的直接坐标更新能力有限，但 Transformer 层
  通过注意力机制提供了全局信息传递：
    - 即使某个 Scaffold 残基远离 Motif，
      Transformer 可以通过全局注意力"看到"整个蛋白质
    - EGNN 接收到的节点特征已经被 Transformer 丰富了全局信息
    - 这种 Transformer + EGNN 的交替设计部分弥补了 EGNN 层数不足的问题
  
  但这仍然不是万能的——对于非常长的蛋白质（>500 残基），
  远端 Scaffold 区域的坐标精度可能较低。
```

---

## 46. 源码阅读顺序建议

```
推荐的源码阅读路径（按依赖关系排序）：

第 1 步：理解数据格式
  → data/ 下的 JSON 文件（看一个具体例子）
  → indexed_dataset.py（所有数据集类，从 IndexedRawTextDataset 开始）

第 2 步：理解模型组件
  → esm_modules.py（Alphabet 类 → 理解词表）
  → esm.py（ESM2 类 → 理解 Transformer 编码器）
  → egnn.py（从 E_GCL 开始 → EGNN → SubstrateEGNN）

第 3 步：理解核心模型
  → geometric_protein_model.py（GeometricProteinNCBIModel.forward()）
  → 重点关注 line 211-303 的前向传播

第 4 步：理解训练目标
  → geometric_protein_ncbi_loss.py（序列+结构损失）
  → geometric_protein_ncbi_ligand_loss.py（新增结合损失）

第 5 步：理解数据流水线
  → ncbi_protein_dataset.py（collate 函数）
  → geometric_protein_design.py（load_protein_dataset + valid_step）

第 6 步：理解训练和推理脚本
  → train_EnzyGen2_mlm.sh → motif.sh → full.sh（三阶段训练）
  → generate_enzygen2_pretrain.sh + generate.py（推理）

第 7 步：理解评估
  → evaluation/prepare_esp_evaluation_pretrain.py
  → generate_pdb_file.py
```

---

## 47. 总结：EnzyGen2 的价值与局限

### 47.1 核心创新

1. **序列-结构协同设计**：Transformer（序列）和 EGNN（结构）交替运行，实现双向信息流
2. **配体结合约束**：通过 SubstrateEGNN 引入配体感知，确保设计的酶能结合目标底物
3. **物种感知**：NCBI Taxonomy embedding 编码进化先验，让模型理解不同物种的序列偏好
4. **三阶段渐进训练**：MLM→Motif→Full 的课程学习策略，从通用到专用逐步提升
5. **Motif-Scaffolding 范式**：保留关键功能位点，只设计框架部分，确保功能保守

### 47.2 主要局限

1. **只输出 Cα 坐标**：不输出全原子结构，需要后处理
2. **推理速度**：KNN 图构建在 CPU 上运行（sklearn.NearestNeighbors），可能成为瓶颈
3. **配体表示简单**：5 维原子特征可能不足以捕捉复杂的化学信息
4. **非自回归生成**：所有位置同时预测（one-shot），缺乏位置间的依赖建模
5. **结合预测是二分类**：只预测结合/不结合，没有亲和力（affinity）的精细度量
6. **训练资源需求高**：需要 8 GPU、百万步更新，对硬件要求较高

### 47.3 适用场景

| 场景 | 是否适合 | 原因 |
|------|---------|------|
| 酶的 Motif-Scaffolding 设计 | **非常适合** | 核心设计场景 |
| 已知活性位点的酶改造 | **适合** | 保留 Motif，设计框架 |
| 特定底物的酶从头设计 | **适合（需微调）** | 利用配体约束 |
| 非酶蛋白设计 | **一般** | 配体约束不适用 |
| 全原子结构设计 | **不适合** | 只输出 Cα |
| 蛋白质-蛋白质相互作用设计 | **不适合** | 只处理小分子配体 |

### 47.4 与其他方法的互补使用

```
推荐的设计流水线：

Step 1: EnzyGen2 → 序列 + Cα 骨架
Step 2: AlphaFold2 / ESMFold → 全原子结构预测
Step 3: Rosetta Relax → 结构精细化
Step 4: 分子动力学模拟 → 稳定性验证
Step 5: 实验合成与功能测试
```

### 47.5 与 DISCO 的详细对比

EnzyGen2 和 DISCO 是两种截然不同的酶设计方法。下面从多个维度进行系统对比：

| 维度 | EnzyGen2 | DISCO |
|------|----------|-------|
| **核心架构** | ESM2 (Transformer) + EGNN (等变图网络) | 基于 AlphaFold3 的扩散模型 |
| **生成方式** | 单次前向传播 (one-shot)：输入 Motif → 一次推理 → 输出序列+结构 | 迭代去噪 (200步扩散)：从噪声出发 → 逐步去噪 → 最终结构 |
| **坐标输出** | 仅 Cα 骨架坐标（每个残基 1 个点） | 完整骨架原子 N/CA/C/O（每个残基 4 个点） |
| **条件输入** | Motif 序列+坐标 + 物种 ID + 配体结合标签（二分类） | 配体三维结构 + 结合口袋定义 |
| **训练策略** | 三阶段课程学习：MLM → Motif → Full（渐进增加复杂度） | 单阶段扩散训练：直接学习去噪过程 |
| **推理速度** | 快：约 1-2 秒/蛋白质（V100 GPU） | 慢：约 5-30 分钟/蛋白质（取决于扩散步数和蛋白质大小） |
| **设计范式** | Motif-Scaffolding：固定功能位点，设计周围框架 | 从头设计 (de novo)：给定配体，从零生成整个蛋白质 |
| **序列设计** | 与结构同时生成（协同设计） | 结构优先，序列需要后续使用 ProteinMPNN 等工具设计 |
| **配体处理** | SubstrateEGNN 处理原子级特征（5维），全局表示用于二分类 | 配体作为扩散过程的条件，保留完整三维几何信息 |
| **物种信息** | 支持：通过 NCBI Taxonomy 嵌入编码进化先验 | 不支持：不考虑物种特异性 |
| **蛋白质语言模型** | 使用预训练的 ESM2（650M 参数），利用进化信息 | 不使用蛋白质语言模型，从结构角度学习 |
| **最大蛋白质长度** | 受 ESM2 限制，约 1022 个残基 | 受 GPU 显存限制，通常 200-500 个残基 |
| **训练数据** | UniRef + PDB（序列为主，结构为辅） | PDB 蛋白质-配体复合物（结构为主） |
| **输出多样性** | 通过解码策略控制（Greedy/Top-K/Top-P） | 通过扩散过程的随机性自然产生 |

**各自的优势场景：**

```
EnzyGen2 更适合的场景：
  1. 已知催化位点（Motif），需要快速设计周围框架
  2. 需要大量候选序列进行筛选（速度快，可批量生成）
  3. 需要利用进化信息（物种嵌入提供先验知识）
  4. 需要同时获得序列和结构

DISCO 更适合的场景：
  1. 从头设计全新的酶-配体结合口袋
  2. 需要精确的全原子骨架结构（N/CA/C/O）
  3. 配体几何形状是设计的核心约束
  4. 对结构精度要求高于生成速度

互补使用策略：
  方案 A：EnzyGen2 快速筛选 → 筛选出最优候选 → DISCO 精细化口袋
  方案 B：DISCO 设计口袋结构 → EnzyGen2 优化框架区域的序列
  方案 C：两者独立生成 → 交叉验证结果的一致性
```

---

> **阅读至此，你应该已经掌握了 EnzyGen2 的完整技术细节。** 如果你正在使用这个工具进行酶设计，建议从微调一个已有的酶家族开始，然后尝试准备自己的数据（参考 `example/prepare_example_data.py`）。祝你的蛋白质设计之旅顺利！
