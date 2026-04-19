# DISCO 训练复现指南

## 背景

DISCO（DIffusion for Sequence-structure CO-design）是一个多模态蛋白质设计模型，能同时生成蛋白质序列和3D结构。论文开源了推理代码和预训练权重，但**训练代码未公开**（`runner/train.py` 不存在）。本文档记录了从论文逆向复现训练脚本的完整过程。

---

## 第一步：理解论文的训练方法

训练复现的核心信息全部来自论文的 Supplementary Information（附录），关键章节：

### A.2.3 Multi-modal diffusion modeling（多模态扩散建模）

DISCO 的核心创新是**联合扩散**——同时对序列和结构进行扩散去噪：

- **结构**：连续空间扩散（continuous diffusion），向原子坐标添加高斯噪声
- **序列**：离散空间扩散（masked discrete diffusion），随机将氨基酸替换为 [MASK] token

训练目标是一个**联合损失**（Eq. S17）：
```
E[ 1/σ² ||D_θ^struct(x_{r,t}) - x^struct||² - (dα_r/dr · 1/(1-α_r)) · Σ δ(x^seq)^T log D_θ^seq(x_{r,t}) ]
```

关键洞察：两个模态的噪声是**独立采样**的，但共享同一个去噪网络 D_θ。这意味着训练时可以分别对结构和序列计算损失，然后加权求和。

### A.5 Training（训练细节）

论文 Table S1 提供了所有超参数，这是复现的蓝图：

| 参数 | 值 | 来源 |
|------|------|------|
| 优化器 | Adam, lr=0.00018, β=(0.9, 0.95) | Table S1 |
| 梯度裁剪 | norm=10.0 | Table S1 |
| EMA 衰减 | 0.999 | Table S1 |
| Warmup | 1000 步线性 | Table S1 |
| LR 衰减 | 0.95× 每 50000 步 | Table S1 |
| 结构噪声 | σ = σ_data · exp(-1.2 + 1.5·N(0,1)) | Section A.5.1 |
| 序列噪声 | r ~ U(0,1), 线性 schedule α_r = 1-r | Section A.5.2 |
| 损失权重 | α_seq=1, α_MSE=4, α_lddt=4, α_disto=0.03 | Eq. S23 |

### A.5.1-A.5.3 损失函数

论文详细描述了四个损失函数，这是训练脚本中最关键的部分：

1. **MSE Loss**（Eq. S19-S21）：加权刚性对齐后的坐标均方误差
2. **Smooth LDDT Loss**（Algorithm 6）：可微分的 LDDT 近似
3. **Sequence Diffusion Loss**（Eq. S8）：MDLM masked 交叉熵
4. **Distogram Loss**：配对距离预测交叉熵

---

## 第二步：分析现有代码结构

虽然没有训练代码，但推理代码包含了训练所需的所有核心组件。通过阅读源码，我梳理了数据流：

```
输入 JSON → SampleDictToFeatures → TaskManager（掩码）→ Featurizer → 特征字典
                                                                          ↓
                                                              DISCO 模型前向传播
                                                              ├── InputFeatureEmbedder
                                                              ├── PairformerStack (N_cycle 循环)
                                                              ├── CrossModalEncoding (序列+结构编码)
                                                              ├── JointDiffusionModule (去噪)
                                                              │   ├── DiffusionConditioning
                                                              │   ├── AtomAttentionEncoder
                                                              │   ├── DiffusionTransformer (24层)
                                                              │   └── AtomAttentionDecoder (×2: 结构+序列)
                                                              └── DistogramHead
```

### 关键发现

1. **`disco/model/disco.py`** 中的 `predict_x0_seq_struct()` 方法是训练时的核心前向传播
2. **`get_pairformer_output()`** 处理 trunk 特征提取（含 cross-modal recycling）
3. **`apply_subs_parameterization()`** 实现 SUBS 参数化（mask token logit → -∞）
4. **`JointDiffusionModule`** 返回去噪的坐标和序列 logits
5. 推理代码中的 `InferenceRunner` 展示了模型初始化、检查点加载的标准流程

---

## 第三步：实现训练组件

### 3.1 损失函数 (`disco/model/losses.py`)

逐一实现论文中的四个损失函数：

**MSE Loss（Eq. S19-S21）：**
```python
# 1. 计算权重 (Eq. S20): w_l = 1 + is_dna*5 + is_rna*5 + is_ligand*10
# 2. 加权刚性对齐 (Eq. S19): Kabsch 算法 (SVD) 找最优旋转
# 3. 计算 MSE (Eq. S21): L = 1/3 * mean(w * ||pred - aligned_gt||²)
```

坑：SVD 和 det 不支持 bfloat16，需要用 `@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)` 强制 float32。

**Smooth LDDT Loss（Algorithm 6）：**
```python
# 用 sigmoid 替代硬阈值的可微分 LDDT 近似
# ε = 1/4 * [sigmoid(0.5-δ) + sigmoid(1-δ) + sigmoid(2-δ) + sigmoid(4-δ)]
# 包含体半径：核酸 30Å，其他 15Å
```

**Sequence Loss（Eq. S8）：**
```python
# 标准 masked diffusion 交叉熵，只在被掩码的位置计算
# 权重 = 1/r（r 是序列时间，避免 r→0 时权重爆炸需要 clamp）
```

**全损失（Eq. S23）：**
```python
L_total = α_seq * L_seq + α_MSE * (t²+σ²)/(t+σ)² * L_MSE + α_lddt * L_lddt + α_disto * L_disto
```

### 3.2 数据管道 (`disco/data/prot2text_adapter.py`)

将 Prot2Text 数据集适配为 DISCO 训练格式：

```python
# CSV 元数据 + AlphaFold PDB 文件 → DISCO 特征字典
# 
# 关键步骤：
# 1. 读取 CSV（支持有/无序列列的两种格式）
# 2. 构建 DISCO 兼容的 sample_dict（全掩码序列用于 cogen 训练）
# 3. 通过 SampleDictToFeatures → TaskManager → Featurizer 标准流程
# 4. 对超长蛋白进行随机裁剪（crop_size=384）
```

裁剪函数是最大的工程挑战——需要同时正确处理：
- Token 级特征（N_token 维）
- 原子级特征（N_atom 维）  
- Token-pair 特征（N_token × N_token）
- Atom-pair 特征（N_atom × N_atom）
- 混合维度特征（如 `[1, N_token]`, `[4, N_token, 37]`）
- 序列索引重映射

最终采用**通用维度检测**方案：根据 tensor 的第一维是否等于 N_token 或 N_atom 来自动决定裁剪方式。

### 3.3 噪声注入 (`disco/data/train_data_pipeline.py`)

```python
# 结构噪声 (Section A.5.1):
#   t_hat = σ_data * exp(-1.2 + 1.5 * N(0,1))    # 对数正态分布
#   x_t = x_0 + t_hat * ε,  ε ~ N(0, I)

# 序列噪声 (Section A.5.2):
#   r ~ U(0, 1)                                    # 均匀采样时间
#   每个 token 独立以概率 r 替换为 [MASK]            # 线性 schedule
```

### 3.4 训练循环 (`runner/train.py`)

```python
# 单步训练流程：
# 1. 从 batch 取出 GT 坐标和序列
# 2. 独立采样结构噪声 t_hat 和序列噪声 r
# 3. 创建噪声输入
# 4. Pairformer trunk 前向传播（不含 cross-modal recycling）
# 5. DiffusionModule 前向传播 → 去噪坐标 + 序列 logits
# 6. SUBS 参数化
# 7. 计算全损失（Eq. S23）
# 8. 反向传播 + 梯度裁剪 + 优化器步进 + EMA 更新
```

---

## 第四步：调试维度问题

这是复现过程中最耗时的部分。DISCO 的推理代码使用 `N_sample` 维度来并行生成多个样本，但训练时 `N_sample=1`，导致多处维度不匹配。

### 问题1：DiffusionModule 的 reps_have_batch_dim 检查

```python
# diffusion.py line 502:
reps_have_batch_dim = z_trunk.ndim == 4 and s_trunk.ndim == 3
if not reps_have_batch_dim:
    s_trunk = expand_at_dim(s_trunk, dim=-3, n=N_sample)  # 重复扩展
    z_pair = expand_at_dim(z_pair, dim=-4, n=N_sample)
```

**问题**：如果 s_trunk 是 2D `[N_token, c_s]`，条件为 False，会被多扩展一次。

**解决**：训练时手动 `s_trunk.unsqueeze(0)` 添加 batch 维，使条件为 True。

### 问题2：AtomTransformer 的维度断言

```python
# transformer.py line 128:
assert len(z.shape) == len(a.shape) + 2
```

**问题**：当 z（pair bias）和 a（atom features）的 batch 维度不一致时触发。

**解决**：确保 s_trunk 和 z_trunk 都有 batch 维度，不传 skip connections。

### 问题3：bfloat16 的 Index put

```python
# "Index put requires source and destination dtypes match"
logits[mask_indices] = -1000000.0  # float32 值 → bf16 tensor
```

**解决**：将 autocast 范围限制在模型前向传播，损失计算在 float32 下进行。

### 问题4：SVD 不支持 bfloat16

```python
# "svd_cuda_gesvdjBatched" not implemented for 'BFloat16'
U, S, Vh = torch.linalg.svd(H)  # H 在 autocast 下变成 bf16
```

**解决**：用 `@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)` 装饰刚性对齐函数。

---

## 第五步：内存优化

DISCO 模型有 886M 参数，训练时内存是主要瓶颈：

| 蛋白质长度 | FP32 内存（无 ckpt） | FP32 + Gradient Checkpointing |
|-----------|---------------------|------------------------------|
| 100 aa | ~14 GB | ~6 GB |
| 200 aa | ~40 GB | ~7 GB |
| 384 aa | ~79 GB (OOM) | ~8 GB |

**Gradient Checkpointing** 对 PairformerStack 启用后，内存从 O(N²) 降到了几乎常数：

```python
from torch.utils.checkpoint import checkpoint
orig = model.pairformer_stack.forward
def ckpt_pairformer(s, z, **kw):
    def run(s, z): return orig(s, z, **kw)
    return checkpoint(run, s, z, use_reentrant=False)
model.pairformer_stack.forward = ckpt_pairformer
```

这是能够使用全部 132K 酶数据训练的关键。

---

## 第六步：三轮迭代训练

### V1：概念验证（445 样本, 64-80 aa）

```yaml
max_length: 80
max_steps: 5000
lr: 0.00018
```

**结果**：MSE 13→0.17（↓98.7%），验证训练流程正确。

**发现的问题**：
- 序列损失 spike 很大（最大 726）
- 445 样本严重过拟合

### V2：调参扩展（17K→80K 样本, 64-200→384 aa）

```yaml
max_length: 200→384
lr: 0.00008          # 降低学习率
gradient_clip_norm: 5.0    # 更严格裁剪
loss_weights.seq: 0.5      # 降低序列权重
loss_weights.smooth_lddt: 6.0  # 提高结构质量权重
gradient_accumulation_steps: 4→8  # 更大等效 batch
```

**调参依据**：
- 序列 spike 太大 → 降低 seq 权重 + 更严格梯度裁剪
- 结构损失已收敛得不错 → 提高 LDDT 权重进一步推
- 数据量增大 → 降低学习率避免振荡

**结果**：MSE 继续降到 0.043，spike 幅度从 726 降到 105。

### V3：全量训练（132K 样本, 64-1024 aa）

修复裁剪 bug + gradient checkpointing → 覆盖全部酶数据：

```yaml
max_length: 1024        # 全部蛋白
crop_size: 384          # 长蛋白随机裁剪
gradient_checkpointing: true  # 内存优化
```

---

## 文件清单

```
DISCO/
├── configs/
│   ├── train.yaml                      # 通用训练配置
│   ├── train_prot2text.yaml            # Prot2Text 全量配置
│   ├── train_enzyme.yaml               # 酶微调配置
│   ├── train_enzyme_h100.yaml          # H100 单卡配置
│   ├── train_enzyme_full.yaml          # V2 配置
│   ├── train_enzyme_full_v2.yaml       # V2→V3 过渡配置
│   └── train_enzyme_full_v3.yaml       # V3 全量配置
├── disco/
│   ├── data/
│   │   ├── prot2text_adapter.py        # Prot2Text 数据适配器
│   │   └── train_data_pipeline.py      # 通用训练数据管道
│   └── model/
│       └── losses.py                   # 四种训练损失函数
├── runner/
│   └── train.py                        # 主训练脚本
└── scripts/
    ├── train_disco.sh                  # 通用启动脚本
    └── train_prot2text.sh              # Prot2Text 启动脚本
```

---

## 复现要点总结

1. **论文附录是金矿**：Table S1 + Algorithm 6 + Eq. S17-S23 提供了所有必要信息
2. **推理代码包含训练组件**：模型架构、特征化、噪声调度器都已实现
3. **维度处理是最大坑**：推理用 N_sample>1 并行，训练用 N_sample=1 导致很多隐含假设失效
4. **混合精度需要小心**：SVD、det、index put 都不支持 bf16
5. **内存优化是关键**：gradient checkpointing 将内存需求从 O(N²) 降到近似常数
6. **迭代式开发**：先小数据验证正确性，再逐步扩大数据量和调参
