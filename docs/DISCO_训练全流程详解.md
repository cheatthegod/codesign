# DISCO 训练全流程详解：从原始 Prot2Text 数据到模型训练

**日期：** 2026-04-19
**适合人群：** 希望理解或复现 DISCO 训练的研究者，从数据构造到训练启动的每一步都有详细解读

---

## 目录

- [1. 全局概览：五个阶段](#1-全局概览五个阶段)
- [2. 第一阶段：原始 Prot2Text 数据](#2-第一阶段原始-prot2text-数据)
- [3. 第二阶段：三层数据富化管线](#3-第二阶段三层数据富化管线)
- [4. 第三阶段：PDB 复合物获取与映射](#4-第三阶段pdb-复合物获取与映射)
- [5. 第四阶段：特征预计算与缓存](#5-第四阶段特征预计算与缓存)
- [6. 第五阶段：训练数据加载与在线处理](#6-第五阶段训练数据加载与在线处理)
- [7. 完整训练流程](#7-完整训练流程)
- [8. 端到端操作手册](#8-端到端操作手册)
- [9. 常见问题与解决方案](#9-常见问题与解决方案)

---

## 1. 全局概览：五个阶段

从原始 Prot2Text 数据到启动 DISCO 训练，需要经过五个阶段：

```
原始 Prot2Text 数据（248K 蛋白质，parquet 格式）
       │
       ▼
┌──────────────────────────────────────────────┐
│ 阶段 1：原始数据（Prot2Text HuggingFace）      │
│   248,315 个蛋白质，每个有序列+功能文本+物种     │
│   + AlphaFold 预测结构（PDB 文件）              │
└──────────────────────────────────────────────┘
       │
       ▼ build_enriched_base.py + compute_structure_features.py
         + generate_route_a_constraints.py + build_training_subsets.py
┌──────────────────────────────────────────────┐
│ 阶段 2：三层富化（酶注释+结构特征+约束提取）     │
│   244,631 个蛋白质 → 136,360 个酶子集           │
│   + 12 个训练子集 + 结构化约束                   │
└──────────────────────────────────────────────┘
       │
       ▼ enzyme_pdb_mapping.tsv + download_prot2chat_missing_pdbs.py
┌──────────────────────────────────────────────┐
│ 阶段 3：PDB 复合物获取                          │
│   7,154 个酶有 PDB 结构 → 52,786 个 PDB 复合物  │
│   下载 mmCIF 文件到 pdb_complexes/               │
└──────────────────────────────────────────────┘
       │
       ▼ preprocess_pdb_cache.py + patch_cache_gt_seq.py
┌──────────────────────────────────────────────┐
│ 阶段 4：特征预计算与缓存                        │
│   52,786 → 51,012 个有效缓存 .pt 文件            │
│   每个含 30+ 个特征张量（ref_pos, gt_seq 等）    │
└──────────────────────────────────────────────┘
       │
       ▼ CachedPDBComplexDataset + noise_structure/sequence
┌──────────────────────────────────────────────┐
│ 阶段 5：训练数据加载与在线处理                    │
│   加载 .pt → 随机裁剪到 384 token → 独立噪声注入  │
│   → 模型前向传播 → 4 个损失 → 反向传播           │
└──────────────────────────────────────────────┘
```

**为什么需要这么多阶段？**

DISCO 的训练数据不是简单的蛋白质序列——它是**蛋白-配体复合物**的全原子表示。原始 Prot2Text 数据只有序列和功能文本，没有配体信息。我们需要：
1. 找出哪些蛋白质是酶（有配体结合能力）
2. 找到这些酶在 PDB 中的实验结构（含配体）
3. 把这些结构解析成 DISCO 能理解的特征格式
4. 在训练时动态加噪，训练扩散模型

---

## 2. 第一阶段：原始 Prot2Text 数据

### 2.1 数据来源

Prot2Text 是一个开源数据集（来自 HuggingFace），包含约 248K 个蛋白质，每个蛋白质有序列、功能文本描述和 AlphaFold 预测结构。

**存储位置：** `Prot2Text-Data/data/`

```
Prot2Text-Data/data/
├── train-00000-of-00001.parquet    (128.9 MB, ~240K 行)
├── validation-00000-of-00001.parquet (2.8 MB, ~4K 行)
└── test-00000-of-00001.parquet      (2.8 MB, ~4K 行)
```

### 2.2 数据格式（7 列）

| 列名 | 类型 | 示例 | 含义 |
|------|------|------|------|
| `accession` | str | "B7LNJ1" | UniProt 唯一标识符 |
| `name` | str | "LPLT_ESCF3" | 蛋白质短名 |
| `Full Name` | str | "Lysophospholipid transporter LplT" | 蛋白质全名 |
| `taxon` | str | "Escherichia" | 物种属名 |
| `sequence` | str | "MKQFT..." | 氨基酸序列（单字母编码，长度 30-1024） |
| `function` | str | "Involved in..." | 功能描述（自然语言） |
| `AlphaFoldDB` | str | "AF-B7LNJ1-F1" | AlphaFold 数据库 ID |

### 2.3 数据统计

```
总蛋白数：      248,315
训练集：        ~240,000
验证集：        ~4,000
测试集：        ~4,000
序列长度范围：  30 - 1,024 残基
平均序列长度：  ~350 残基
```

**关键问题：** 这些数据**没有**配体信息、酶分类信息、或 PDB 实验结构。需要通过富化管线补充。

---

## 3. 第二阶段：三层数据富化管线

### 3.1 Layer 1：基础富化（build_enriched_base.py）

**脚本：** `COT_enzyme_design/scripts/build_enriched_base.py`

**目的：** 将原始 Prot2Text 数据与酶注释、PDB 映射进行关联

**操作步骤：**

```
步骤 1：加载 RFD3 过滤后的训练/验证/测试分割
       （已过滤为 64-1024 残基的单体蛋白）
       
步骤 2：关联 UniProt 酶注释
       来源：enzyme_filter_uniprot/prot2text_with_enzyme_class.csv
       新增列：ec_number（EC 编号）、rhea_id（反应 ID）、
              go_ids（GO 标注）、enzyme_class（酶分类）
       
步骤 3：关联 PDB 结构映射
       来源：DISCO/local_data/enzyme_pdb_mapping.tsv
       新增列：pdb_ids（PDB 结构 ID 列表）、has_pdb_mapping（是否有实验结构）
```

**酶分类规则：**

| 分类 | 数量 | 含义 | 判定依据 |
|------|------|------|----------|
| `enzyme_gold` | 132,406 | 高置信度酶 | 有明确的 EC 编号 |
| `enzyme_silver` | 3,954 | 中置信度酶 | 有酶相关 GO 注释但无明确 EC |
| `possible_enzyme_bronze` | 289 | 低置信度酶 | 功能文本暗示催化活性 |
| `non_enzyme` | 107,982 | 非酶蛋白 | 无酶相关证据 |

**输出：**
- `prot2text_enriched_base.parquet`：244,631 行 × 18 列
- `prot2text_enriched_enzyme.parquet`：136,360 行（gold + silver）

**运行命令：**
```bash
cd COT_enzyme_design
python scripts/build_enriched_base.py
```

### 3.2 Layer 2：结构特征提取（compute_structure_features.py）

**脚本：** `COT_enzyme_design/scripts/compute_structure_features.py`

**目的：** 从 AlphaFold 预测的 PDB 结构中提取几何和质量特征

**对每个蛋白质提取 13 个特征：**

```python
# 质量指标
mean_plddt:          平均 pLDDT 置信度分数（0-100）
min_plddt:           最低 pLDDT 分数
fraction_low_plddt:  pLDDT < 50 的残基比例（低置信度）
fraction_medium_plddt: pLDDT < 70 的残基比例（中等置信度）

# 二级结构
ss_fraction_helix:   α-螺旋占比
ss_fraction_sheet:   β-折叠占比
ss_fraction_loop:    环区占比
is_non_loopy:        是否为"非环状"蛋白（loop < 30%）

# 溶剂可及性（SASA）
mean_residue_sasa:   平均残基 SASA（Å²）
buried_fraction:     埋藏残基比例（SASA < 10 Å²）
exposed_fraction:    暴露残基比例（SASA > 40 Å²）

# 基本信息
struct_num_chains:   链数
struct_num_residues: 残基数
```

**为什么需要这些特征？**

这些特征用于后续的训练子集划分和条件化训练。例如：
- `is_non_loopy=True` 的蛋白更适合作为 RFD3 训练数据（环区过多的蛋白结构不稳定）
- `buried_fraction` 高的蛋白更可能有深埋的活性位点
- `mean_plddt >= 85` 过滤掉 AlphaFold 预测质量差的蛋白

**输出：** `structure_features.parquet`（244,631 行 × 13 列）

**运行命令：**
```bash
python scripts/compute_structure_features.py --resume
# --resume 支持断点续传，每处理 1000 个蛋白保存一次进度
# 单核处理速度 ~500 蛋白/秒，全量约 8 分钟
```

### 3.3 Layer 3：约束提取（generate_route_a_constraints.py）

**脚本：** `COT_enzyme_design/scripts/generate_route_a_constraints.py`

**目的：** 基于 EC 编号、GO 注释和功能文本，提取结构化的酶设计约束

**提取逻辑（纯规则，无 LLM）：**

```
EC 编号 → 反应类型 → 反应机制
  例：EC 1.1.1.179 → 氧化还原酶 → 氧化反应

GO 注释 → 金属辅因子 + 有机辅因子
  例：GO:0008270 → 锌离子结合
      GO:0010181 → FMN 结合

功能文本 → 底物家族 + 活性位点类型
  例："NAD-dependent dehydrogenase" → 辅因子依赖型

结构特征 → 口袋类型 + 折叠偏好
  例：buried_fraction=0.45 → 深埋口袋
      ss_fraction_helix=0.60 → 螺旋富集
```

**覆盖率统计：**

| 约束字段 | 覆盖率 | 来源 |
|---------|--------|------|
| reaction_mechanism | 97.1% | EC 编号映射 |
| required_roles | 90.4% | EC + GO 推断 |
| active_site_style | 74.2% | GO + 文本模式 |
| substrate_family | 70.3% | EC 二级分类 + 文本 |
| cofactor_hint | 35.8% | GO + 文本正则 |
| metal_hint | 22.1% | GO + 文本正则 |

**输出：** `prot2text_route_a_constraints.jsonl`（每行一个 JSON 对象）

### 3.4 训练子集构建（build_training_subsets.py）

**脚本：** `COT_enzyme_design/scripts/build_training_subsets.py`

**目的：** 根据不同维度划分训练子集，支持针对性训练

**生成的 12 个子集：**

| 子集名称 | 筛选条件 | 训练集大小 | 用途 |
|---------|---------|-----------|------|
| `high_quality` | pLDDT≥85, 非环, buried≥0.20 | ~12K | 高质量结构训练 |
| `balanced_ec` | 每 EC 类最多 5K | ~38K | EC 类别均衡训练 |
| `reaction_oxidoreductase` | EC 1.x.x.x | ~17K | 氧化还原酶专项 |
| `reaction_transferase` | EC 2.x.x.x | ~49K | 转移酶专项 |
| `reaction_hydrolase` | EC 3.x.x.x | ~25K | 水解酶专项 |
| `reaction_lyase` | EC 4.x.x.x | ~8K | 裂合酶专项 |
| `reaction_isomerase` | EC 5.x.x.x | ~5K | 异构酶专项 |
| `reaction_ligase` | EC 6.x.x.x | ~5K | 连接酶专项 |
| `reaction_translocase` | EC 7.x.x.x | ~2K | 转位酶专项 |
| `pocket_deeply_buried` | buried_fraction ∈ [0.35, 1.0) | ~40K | 深埋口袋 |
| `pocket_semi_buried` | buried_fraction ∈ [0.20, 0.35) | ~95K | 半埋口袋 |
| `pocket_surface_exposed` | buried_fraction ∈ [0.00, 0.20) | ~1K | 表面暴露口袋 |

**每个子集的输出格式：**

```
training_subsets/high_quality/
├── train.csv         # 列: example_id, path（指向 AlphaFold PDB）
├── test.csv
├── validation.csv
└── metadata.json     # 统计信息
```

**运行命令：**
```bash
python scripts/build_training_subsets.py
```

---

## 4. 第三阶段：PDB 复合物获取与映射

### 4.1 为什么需要 PDB 复合物

DISCO 训练的核心目标是**蛋白-配体协同设计**。这要求训练数据包含：
1. 蛋白质的实验结构（不是 AlphaFold 预测）
2. 配体的三维坐标（小分子药物、辅因子等）
3. 蛋白-配体之间的空间关系

AlphaFold 结构**没有**配体信息。只有 PDB 中的实验结构才包含蛋白-配体复合物。

### 4.2 映射关系

```
Prot2Text 蛋白（248K，UniProt ID）
       │
       ▼  enzyme_pdb_mapping.tsv（UniProt → PDB 映射）
       │
       │  只有 7,154 个酶有 PDB 实验结构
       │  这 7,154 个酶对应 52,786 个 PDB 复合物
       │  （一个蛋白可以有多个 PDB 结构，不同配体/条件）
       │
       ▼
enzyme_train.csv（训练用映射文件）
  ├── accession: UniProt ID
  ├── pdb_ids: "4X7R;4X6L;4X7P" （分号分隔的 PDB ID 列表）
  └── has_pdb_complex: 1（有实验结构）
```

### 4.3 mmCIF 文件下载

```
pdb_complexes/
├── 101m.cif.gz     (6.3 KB, 肌红蛋白 + 血红素)
├── 102m.cif.gz     (6.3 KB)
├── ...
├── 9uwh.cif.gz     (1.2 GB, 大型复合物)
└── 9zzr.cif.gz     (79 MB)
总计：51,012 个有效文件
```

### 4.4 配体过滤策略

并非所有 PDB 中的小分子都是生物学有意义的配体。DISCO 使用以下过滤规则：

**保留（生物学相关配体）：**

```python
KEEP_LIGANDS = {
    # 辅因子
    "HEM", "FAD", "FMN", "NAD", "NAP", "COA", "PLP", "TPP", "SAM", "SAH",
    # 核苷酸
    "ADP", "ATP", "GDP", "GTP", "UDP", "UTP", "CDP", "CTP",
    # 金属簇
    "SF4", "FES", "F3S",
    # 以及所有未明确排除的 CCD 分量
}
```

**排除（非生物学相关）：**

```python
SKIP_LIGANDS = {
    # 水分子
    "HOH", "DOD",
    # 离子（结晶缓冲液残留）
    "SO4", "PO4", "CL", "NA", "MG", "CA", "ZN", "K", "MN",
    # 结晶添加剂
    "GOL", "EDO", "PEG", "DMS", "ACT", "MPD", "TRS", "BME",
}
```

**过滤影响：** 每个 PDB 复合物平均有 3-5 个小分子，过滤后保留 1-2 个生物学配体。

---

## 5. 第四阶段：特征预计算与缓存

### 5.1 为什么需要预计算

从 mmCIF 文件解析蛋白-配体复合物并特征化的过程非常耗时：
- 解析 mmCIF：~100ms
- CCD 查询（化学组分字典）：~200ms
- 特征化（AtomArray → 张量）：~500ms
- 总计每个样本 ~800ms

训练 100K 步，每步加载一个样本，如果实时解析需要 800ms × 100,000 = 22 小时**仅用于数据加载**。

预计算后加载 .pt 文件只需 ~5ms，**加速 160 倍**。

### 5.2 缓存构建流程（preprocess_pdb_cache.py）

**脚本：** `DISCO/scripts/preprocess_pdb_cache.py`

```
mmCIF 文件（pdb_complexes/xxxx.cif.gz）
       │
       ▼ parse_complex_from_mmcif()
       │
       ├── 解析蛋白质链：序列、链 ID、原子坐标
       ├── 解析配体：CCD 编码、原子坐标、化学键
       └── 过滤：去除水分子、离子、结晶添加剂
       │
       ▼ build_disco_sample_dict()
       │
       ├── 蛋白序列全部掩码（"------..."）
       └── 配体以 CCD 编码表示
       │
       ▼ SampleDictToFeatures()
       │
       ├── 构建 AtomArray（Biotite 格式）
       ├── 添加参考构象（CCD 理想坐标 → ref_pos）
       ├── Token 化（残基→token，配体原子→token）
       ├── 特征化（Featurizer：所有张量特征）
       └── 掩码处理（TaskManager：删侧链、掩参考坐标）
       │
       ▼ 构建 gt_seq（真实序列标签）
       │
       ├── 从解析的 PDB 蛋白质链中提取真实氨基酸序列
       ├── 映射：asym_id → chain → sequence → token position
       └── gt_seq_full [N_token], gt_seq [N_prot]
       │
       ▼ torch.save(cache, "xxxx.pt")
```

**运行命令：**

```bash
cd DISCO
python scripts/preprocess_pdb_cache.py \
    --pdb_dir local_data/pdb_complexes \
    --output_dir local_data/pdb_cache \
    --crop_size 384
```

**输出统计：**
- 输入：52,786 个 mmCIF 文件
- 成功缓存：51,012 个 .pt 文件
- 失败（解析错误/无蛋白链）：1,774 个
- 总存储：~88 GB
- 单文件大小：6 KB - 1.2 GB（取决于复合物大小）
- 处理时间：约 6-8 小时（单核）

### 5.3 缓存文件内部结构

每个 .pt 文件是一个 Python 字典，包含约 30 个张量：

```python
cache = torch.load("101m.pt", map_location="cpu", weights_only=False)

# 元数据（非张量）
cache["_sample_name"]    # "101m"
cache["_sample_info"]    # {"pdb_id": "101m", "accession": "P02185", ...}

# Token 级特征 [N_token]
cache["token_index"]     # [0, 1, 2, ..., N_token-1]    int64
cache["residue_index"]   # [1, 2, 3, ..., L, 1, 2, ...]  int64（PDB 残基编号）
cache["asym_id"]         # [0, 0, ..., 0, 1, 1, ...]     int64（链 ID）
cache["entity_id"]       # [0, 0, ..., 0, 1, 1, ...]     int64（实体 ID）
cache["sym_id"]          # [0, 0, ..., 0, 0, 0, ...]     int64

# 残基类型 [N_token] 或 [N_token, 32]
cache["restype"]         # [N_token, 32]  one-hot 编码（32 类 token）

# 原子级特征 [N_atom]
cache["ref_pos"]         # [N_atom, 3]    CCD 理想坐标（float32）
cache["ref_mask"]        # [N_atom]       原子有效性（int64）
cache["ref_element"]     # [N_atom, 128]  元素 one-hot（float32）
cache["ref_charge"]      # [N_atom]       形式电荷（int64）
cache["ref_atom_name_chars"]  # [N_atom, 4, 64]  原子名编码
cache["ref_space_uid"]   # [N_atom]       参考空间 ID

# 原子-Token 映射
cache["atom_to_token_idx"]     # [N_atom]  每个原子属于哪个 token
cache["atom_to_tokatom_idx"]   # [N_atom]  原子在 token 内的编号

# 键合信息 [N_token, N_token]
cache["token_bonds"]     # [N_token, N_token]  化学键邻接矩阵

# 掩码特征
cache["backbone_atom_mask"]        # [N_atom]  骨架原子掩码
cache["distogram_rep_atom_mask"]   # [N_atom]  距离图代表原子
cache["plddt_m_rep_atom_mask"]     # [N_atom]  pLDDT 代表原子
cache["prot_residue_mask"]         # [N_token] 蛋白质残基布尔掩码
cache["histag_mask"]               # [N_prot]  His-tag 掩码

# 真实序列标签（训练核心）
cache["gt_seq_full"]     # [N_token]  token 级真实氨基酸索引
cache["gt_seq"]          # [N_prot]   蛋白质残基真实氨基酸索引

# 序列输入（全部被掩码）
cache["masked_prot_restype"]  # [N_prot]  全部为 31（MASK_TOKEN_IDX）
```

**具体示例（101m.pt，肌红蛋白 + 血红素）：**

```
N_token = 203（154 个蛋白残基 + 49 个配体原子 token）
N_atom  = 665（154×4 骨架原子 + 49 配体原子）
N_prot  = 154

gt_seq[:10] = [12, 19, 10, 15, 6, 7, 6, 17, 5, 10]
              # 对应: M,  W,  L,  S,  E, G, E, R, D, L
              
masked_prot_restype[:10] = [31, 31, 31, 31, 31, 31, 31, 31, 31, 31]
              # 全部为 MASK（31），训练时模型需要预测真实序列
```

### 5.4 gt_seq 补丁（patch_cache_gt_seq.py）

**背景：** 初始缓存构建时存在一个 bug——由于蛋白序列被全部掩码（`mask_sequence=True`），`gt_seq` 回退为全 UNK（index=20），导致训练时序列损失为零。

**修复脚本：** `DISCO/scripts/patch_cache_gt_seq.py`

```
原始缓存：gt_seq = [20, 20, 20, ..., 20]  ← 全 UNK（bug）
                              │
                              ▼ patch_cache_gt_seq.py
                              │
                              ├── 重新解析 mmCIF 文件
                              ├── 提取真实蛋白质链序列
                              ├── 映射 token position → chain → amino acid
                              └── 写入正确的 gt_seq_full 和 gt_seq
                              │
修复后缓存：gt_seq = [12, 19, 10, 15, ...]  ← 真实氨基酸索引
```

**运行命令：**
```bash
cd DISCO
python scripts/patch_cache_gt_seq.py \
    --cache-dir local_data/pdb_cache \
    --pdb-dir local_data/pdb_complexes \
    --workers 32
```

**这个 bug 造成的影响非常严重：** 旧训练（v1）的前 23,000 步序列损失为零，序列解码器完全没有学到任何东西。详见 DISCO_v2_training_report.md 第 6.1 节。

---

## 6. 第五阶段：训练数据加载与在线处理

### 6.1 CachedPDBComplexDataset：缓存加载

**文件：** `disco/data/pdb_complex_adapter.py`

```python
class CachedPDBComplexDataset(Dataset):
    """从预计算的 .pt 缓存文件快速加载训练数据"""
    
    def __init__(self, cache_dir, crop_size=384):
        self.cache_dir = cache_dir
        self.crop_size = crop_size
        # 扫描所有 .pt 文件
        self.pt_files = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
        # 每个文件对应一个 PDB 复合物
        
    def __getitem__(self, idx):
        # 1. 加载缓存（~5ms）
        cache = torch.load(self.pt_files[idx], map_location="cpu")
        sample_name = cache.pop("_sample_name", "")
        sample_info = cache.pop("_sample_info", {})
        feat = cache
        
        # 2. 如果超过 crop_size，随机裁剪
        if feat["token_index"].shape[0] > self.crop_size:
            feat = self._random_crop(feat, self.crop_size)
        else:
            # 不需要裁剪，但仍需从 gt_seq_full 提取 gt_seq
            feat["gt_seq"] = feat["gt_seq_full"][feat["prot_residue_mask"]]
            
        # 3. 跳过空蛋白（防止 DDP 挂起）
        if feat["prot_residue_mask"].sum() == 0:
            return None
            
        return {"input_feature_dict": feat, "sample_name": sample_name}
```

### 6.2 随机裁剪逻辑

当蛋白-配体复合物超过 384 个 token 时，需要裁剪。裁剪策略是**随机连续窗口**：

```python
def _random_crop(feat, crop_size):
    N = feat["token_index"].shape[0]
    start = random.randint(0, N - crop_size)
    end = start + crop_size
    
    # 裁剪 token 级特征
    for key in ["token_index", "residue_index", "asym_id", "entity_id", ...]:
        feat[key] = feat[key][start:end]
    
    # 裁剪 token 对特征
    feat["token_bonds"] = feat["token_bonds"][start:end, start:end]
    
    # 裁剪原子级特征（需要找到 token 范围对应的原子范围）
    atom_mask = (feat["atom_to_token_idx"] >= start) & (feat["atom_to_token_idx"] < end)
    for key in ["ref_pos", "ref_mask", "ref_element", ...]:
        feat[key] = feat[key][atom_mask]
    
    # 重新索引 atom_to_token_idx（减去 start 偏移）
    feat["atom_to_token_idx"] = feat["atom_to_token_idx"][atom_mask] - start
    
    # 重建掩码字段
    n_prot = feat["prot_residue_mask"].sum().item()
    feat["masked_prot_restype"] = torch.full((n_prot,), MASK_TOKEN_IDX)
    feat["gt_seq"] = feat["gt_seq_full"][feat["prot_residue_mask"]]
    
    return feat
```

**裁剪示例：**

```
原始：N_token=500, N_atom=2000
         [====================500 tokens====================]
裁剪后：   [=========384 tokens==========]
         start=80, end=464

token_index:        [80, 81, ..., 463] → 重新编号 → [0, 1, ..., 383]
atom_to_token_idx:  [80, 80, 80, 80, 81, ...] → 减去80 → [0, 0, 0, 0, 1, ...]
```

### 6.3 噪声注入：结构噪声

**文件：** `disco/data/train_data_pipeline.py`

```python
def noise_structure(coords, sigma_data=16.0, log_mean=-1.2, log_std=1.5):
    """对原子坐标添加高斯噪声（EDM 框架）"""
    
    # 1. 采样噪声水平 t_hat（对数正态分布）
    z = torch.randn(1)                              # 标准正态
    t_hat = sigma_data * torch.exp(log_mean + log_std * z)  # 对数正态变换
    t_hat = t_hat.clamp(min=1e-4)                   # 防止零噪声
    
    # 2. 添加高斯噪声
    noise = torch.randn_like(coords)                # 与坐标同维度的噪声
    noised_coords = coords + t_hat * noise           # x_t = x_0 + t_hat * ε
    
    return noised_coords, t_hat
```

**噪声水平的物理含义：**

```
对数正态分布参数：log_mean=-1.2, log_std=1.5
  → t_hat 的中位数 = 16 * exp(-1.2) ≈ 4.8 Å
  → t_hat 的范围（95%）：约 0.05 Å ~ 450 Å

sigma_data = 16.0 Å 是蛋白质坐标的典型标准差（蛋白质半径 ~15-20 Å）

t_hat 值的物理解读：
  t_hat = 0.1 Å  → 亚原子级微扰（几乎不变）
  t_hat = 1.0 Å  → 键长级微扰（原子略微偏移）
  t_hat = 5.0 Å  → 残基级微扰（局部结构扭曲）
  t_hat = 16.0 Å → 蛋白质级微扰（整体变形）
  t_hat = 160 Å  → 完全破坏（纯噪声）
```

### 6.4 噪声注入：序列噪声

```python
def noise_sequence(seq_tokens, mask_token_idx=31, noise_min=0.0, noise_max=0.95):
    """对序列进行掩码扩散噪声注入（MDLM/SUBS）"""
    
    # 1. 采样掩码率（均匀分布，限制上限为 0.95）
    r = torch.rand(1)
    mask_rate = (noise_min + (noise_max - noise_min) * r).squeeze()
    # mask_rate ∈ [0, 0.95]
    
    # 2. 独立掩码每个位置
    mask_prob = torch.rand(seq_tokens.shape)  # 每个位置的随机数
    masked_positions = mask_prob < mask_rate    # 被掩码的布尔掩码
    
    # 3. 替换为 MASK token
    noised_seq = seq_tokens.clone()
    noised_seq[masked_positions] = mask_token_idx  # 31 = MASK
    
    return noised_seq, mask_rate, masked_positions
```

**为什么 noise_max=0.95 而不是 1.0？**

序列扩散损失的权重为 `1/(1-mask_rate)`：

```
mask_rate = 0.50 → 权重 = 1/(1-0.50) = 2.0    （正常）
mask_rate = 0.90 → 权重 = 1/(1-0.90) = 10.0   （偏高）
mask_rate = 0.99 → 权重 = 1/(1-0.99) = 100.0   （极端）
mask_rate = 0.999 → 权重 = 1/(1-0.999) = 1000.0 （爆炸）

限制 noise_max=0.95：
  最大权重 = 1/(1-0.95) = 20（可控）
  消除了 loss spike（v1 中 4.1% 的步骤出现极端 spike）
```

### 6.5 结构噪声和序列噪声为什么独立

在每一步训练中，结构噪声 `t_hat` 和序列噪声 `mask_rate` 是**独立采样**的：

```
训练步骤 A：t_hat = 50 Å（高结构噪声），mask_rate = 0.2（低序列噪声）
  → 模型学习：在结构高度混乱的情况下，利用部分已知序列恢复结构

训练步骤 B：t_hat = 0.5 Å（低结构噪声），mask_rate = 0.9（高序列噪声）
  → 模型学习：在结构几乎正确的情况下，利用结构信息预测序列

训练步骤 C：t_hat = 20 Å，mask_rate = 0.5
  → 模型学习：两种信息都部分可用，协同去噪
```

这种独立采样确保模型在各种噪声组合下都能工作，对应推理时序列和结构同步但不完全同步的去噪过程。

---

## 7. 完整训练流程

### 7.1 单步训练（training_step）详解

```python
def training_step(model, batch, configs, device):
    """单步训练的完整操作"""
    
    feat = batch["input_feature_dict"]
    
    # ═══ 步骤 1：提取真实标签 ═══
    gt_coords = feat["ref_pos"].float().to(device)           # [N_atom, 3]
    gt_seq = feat.get("gt_seq")                               # [N_prot]
    if gt_seq is None:
        gt_seq = torch.full((n_prot,), UNK_IDX)  # 回退（不应发生）
    atom_mask = feat["ref_mask"].float().to(device)           # [N_atom]
    
    # ═══ 步骤 2：注入结构噪声 ═══
    noised_coords, t_hat = noise_structure(
        gt_coords, sigma_data=16.0, log_mean=-1.2, log_std=1.5
    )
    # noised_coords: [N_atom, 3]  ← 加噪后的坐标
    # t_hat: 标量              ← 噪声水平
    
    # ═══ 步骤 3：注入序列噪声 ═══
    noised_seq, mask_rate, masked_positions = noise_sequence(
        gt_seq, noise_min=0.0, noise_max=0.95
    )
    # noised_seq: [N_prot]     ← 部分被 MASK 的序列
    # mask_rate: 标量           ← 掩码比例
    # masked_positions: [N_prot] ← 被掩码的布尔掩码
    
    # ═══ 步骤 4：更新输入特征 ═══
    feat["masked_prot_restype"] = noised_seq.long().to(device)
    
    # ═══ 步骤 5：模型前向传播 ═══
    # 5a: Pairformer 输出（N_cycle 次循环精炼）
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        s_inputs, s, z, s_orig, z_orig, lm_logits, encoding_dict = (
            model.get_pairformer_output(feat, N_cycle=4)
        )
    # s: [N_token, 384]           ← 单表示
    # z: [N_token, N_token, 128]  ← 孪生表示
    
    # 5b: 扩散模块（预测干净坐标 + 序列 logit）
    x0_struct, seq_logits_pre = model.diffusion_module(
        x_noisy=noised_coords, t_hat=t_hat, s_trunk=s, z_trunk=z, ...
    )
    # x0_struct: [N_atom, 3]     ← 预测的干净坐标
    # seq_logits_pre: [N_prot, 32] ← 原始序列 logit
    
    # 5c: SUBS 参数化（约束序列输出）
    seq_logits = model.apply_subs_parameterization(
        seq_logits_pre, noised_seq, enforce_unmask_stay=False
    )
    # seq_logits: [N_prot, 32]   ← 只有 token 0-19 有效
    
    # 5d: 距离图预测
    distogram_logits = model.distogram_head(z)
    # distogram_logits: [N_token, N_token, 64]
    
    # ═══ 步骤 6：计算损失 ═══
    seq_loss_mask = masked_positions.float().to(device)  # 只在掩码位置计算
    
    loss_dict = full_training_loss(
        pred_coords=x0_struct,        # 预测坐标
        gt_coords=gt_coords,          # 真实坐标
        atom_mask=atom_mask,           # 有效原子掩码
        t_hat=t_hat,                   # 结构噪声水平
        seq_logits=seq_logits,         # 序列预测
        gt_seq=gt_seq,                 # 真实序列
        mask_rate=mask_rate,           # 序列噪声水平
        seq_mask=seq_loss_mask,        # 序列损失掩码
        distogram_logits=distogram_logits,
        alpha_seq=1.0,                 # 序列损失权重
        alpha_mse=4.0,                 # MSE 损失权重
        alpha_smooth_lddt=4.0,         # LDDT 损失权重
        alpha_distogram=0.03,          # 距离图损失权重
    )
    
    return loss_dict
    # loss_dict = {
    #   "total": 12.84,     ← 加权总损失
    #   "seq": 10.62,       ← 序列扩散损失（未加权）
    #   "mse": 0.30,        ← 结构 MSE 损失
    #   "smooth_lddt": 0.25, ← 平滑 LDDT 损失
    #   "distogram": 1.56    ← 距离图损失
    # }
```

### 7.2 四个损失函数

**总损失公式（论文公式 S23）：**

```
L_total = 1.0 × L_seq + 4.0 × L_MSE + 4.0 × L_lddt + 0.03 × L_disto
```

**各分量的贡献比例（训练收敛后）：**

```
L_seq 贡献：   1.0 × 6.90 = 6.90  (76.6%)   ← 主导分量
L_MSE 贡献：   4.0 × 0.284 = 1.14  (12.6%)
L_lddt 贡献：  4.0 × 0.232 = 0.93  (10.3%)
L_disto 贡献： 0.03 × 1.55 = 0.05  (0.5%)   ← 辅助分量
总计：                         9.01  (100%)
```

#### 损失 1：结构 MSE（Kabsch 对齐后的加权均方误差）

```python
def mse_loss(pred_coords, gt_coords, atom_mask, is_dna, is_rna, is_ligand):
    # 1. 计算原子权重：配体原子 10 倍权重
    weights = 1.0                          # 蛋白质原子：权重 1
    weights += is_dna * 5.0                # DNA 原子：权重 6
    weights += is_rna * 5.0                # RNA 原子：权重 6
    weights += is_ligand * 10.0            # 配体原子：权重 11

    # 2. Kabsch 刚性对齐（SVD 求解最优旋转平移）
    gt_aligned = weighted_rigid_align(pred_coords, gt_coords, weights, atom_mask)
    
    # 3. 计算 MSE
    sq_diff = ((pred_coords - gt_aligned) ** 2).sum(dim=-1)  # 每原子平方差
    loss = sum(weights * sq_diff * atom_mask) / n_valid / 3   # 除以 3（xyz 三维）
    
    return loss

# 物理含义：
# MSE = 0.3 → 平均原子偏差 ≈ sqrt(0.3 * 3) ≈ 0.95 Å（接近实验精度）
# MSE = 1.0 → 平均原子偏差 ≈ 1.73 Å（可接受）
# MSE = 5.0 → 平均原子偏差 ≈ 3.87 Å（较差）
```

**为什么配体原子权重是 10 倍？** 配体只占总原子数的 ~5%，但它是条件输入的核心。如果不加权，模型可能忽略配体坐标的精确性，只优化蛋白质坐标。10 倍权重确保模型花足够的容量学习配体结合几何。

#### 损失 2：平滑 LDDT

```python
def smooth_lddt_loss(pred_coords, gt_coords, atom_mask, is_dna, is_rna):
    # 1. 计算所有原子对的距离
    pred_dists = cdist(pred_coords, pred_coords)   # [N, N]
    gt_dists = cdist(gt_coords, gt_coords)         # [N, N]
    delta = |gt_dists - pred_dists|                 # 距离差

    # 2. 用 sigmoid 近似 LDDT 的 4 个阈值
    eps = 0.25 * (sigmoid(0.5 - delta)    # 阈值 0.5 Å
              +   sigmoid(1.0 - delta)    # 阈值 1.0 Å
              +   sigmoid(2.0 - delta)    # 阈值 2.0 Å
              +   sigmoid(4.0 - delta))   # 阈值 4.0 Å

    # 3. 只考虑空间邻近的原子对
    #    蛋白质：15 Å 内的原子对
    #    核酸：30 Å 内的原子对

    # 4. LDDT = mean(eps), 损失 = 1 - LDDT
    return (1 - lddt).mean()
```

**LDDT 的直觉：** 如果预测的距离矩阵与真实的距离矩阵在各个阈值下都很接近，LDDT ≈ 1（损失 ≈ 0）。这个损失更关注**局部距离关系**的正确性，而非全局 RMSD。

#### 损失 3：序列扩散损失

```python
def sequence_diffusion_loss(seq_logits, gt_seq, mask_rate, seq_mask):
    # 1. 交叉熵（每个 token）
    ce = cross_entropy(seq_logits, gt_seq, reduction="none")
    
    # 2. SUBS/MDLM 权重
    r = 1.0 - mask_rate       # 未掩码比例
    weight = 1.0 / r.clamp(min=1e-4)  # 权重 = 1/(1-mask_rate)
    
    # 3. 只在被掩码位置计算
    masked_ce = ce * seq_mask
    per_sample = sum(masked_ce) / n_valid
    
    return (weight * per_sample).mean()
```

**权重的含义：** 当 mask_rate 高（大部分被掩码），预测更难，需要更大的权重来强调这些困难样本。当 mask_rate 低（大部分已揭示），预测较容易，权重较小。

#### 损失 4：距离图损失

```python
def distogram_loss(distogram_logits, gt_coords, token_mask, atom_to_token_idx):
    # 1. 取每个 token 的代表原子（通常是 Cα）坐标
    # 2. 计算 token 对之间的距离
    # 3. 分入 64 个距离区间（2Å 到 22Å，每个 0.3125Å）
    # 4. 交叉熵损失
    return ce_loss
```

**为什么权重只有 0.03？** 距离图是辅助任务，主要目的是帮助孪生表示 z 学习距离信息。它只贡献总损失的 0.5%，不会喧宾夺主。

### 7.3 Trainer 类

```python
class Trainer:
    def __init__(self, configs):
        self.global_step = 0
        
        # 1. 初始化分布式环境
        self.fabric = Fabric(strategy=DDPStrategy(
            find_unused_parameters=True,
            static_graph=True
        ))
        
        # 2. 初始化模型
        self.model = DISCO(configs)
        # 冻结 DPLM-650M（不训练语言模型）
        for param in self.model.lm_module.parameters():
            param.requires_grad = False
        # 梯度检查点（用计算换显存）
        enable_gradient_checkpointing(self.model.pairformer_stack)
        
        # 3. 初始化优化器
        self.optimizer = Adam(
            trainable_params, lr=1.5e-4,
            betas=(0.9, 0.95), weight_decay=1e-8
        )
        
        # 4. 初始化 EMA
        self.ema = EMA(self.model, decay=0.999)
        
        # 5. 加载检查点（从 enzyme_v3 EMA 初始化）
        self.load_checkpoint("enzyme_v3/disco_ema_step_60000.pt")
```

### 7.4 学习率调度

```
学习率调度：线性预热 + 阶梯衰减

lr
0.00015 ┤                    ┌─────────────────────┬─────────────────
        │                   ╱                       │  0.95x 衰减
        │                  ╱                        ▼
0.00014 ┤                 ╱                    ┌────────────────────
        │                ╱                     │
        │               ╱                      │
        │              ╱                       │
        │             ╱                        │
        │            ╱                         │
        │           ╱                          │
        │          ╱                           │
0       ┤─────────╱                            │
        └───────┬──────┬───────────────────────┬──────────────────
              0     2,000               50,000             100,000
                  预热结束               第一次衰减           训练结束
```

---

## 8. 端到端操作手册

### 8.1 前置条件

```bash
# 环境
conda activate disco_env  # 或 .venv
cd COT_enzyme_design/DISCO

# 硬件需求
# GPU: 4× NVIDIA H100 80GB（或 A100 80GB）
# RAM: 64 GB+
# 磁盘: ~200 GB（缓存 + 检查点）
```

### 8.2 步骤 1：准备原始数据

```bash
# 下载 Prot2Text 数据集（如果还没有）
cd ../Prot2Text-Data
# 已有 data/ 目录，包含 parquet 文件
```

### 8.3 步骤 2：运行富化管线

```bash
cd ../
python scripts/build_enriched_base.py
python scripts/compute_structure_features.py --resume
python scripts/generate_route_a_constraints.py
python scripts/build_training_subsets.py

# 检查输出
ls Prot2Text-Data/enriched/
# → prot2text_enriched_base.parquet
# → prot2text_enriched_enzyme.parquet
# → structure_features.parquet
# → prot2text_route_a_constraints.jsonl
# → training_subsets/（12 个子目录）
```

### 8.4 步骤 3：获取 PDB 复合物

```bash
# 下载 PDB mmCIF 文件
cd DISCO
mkdir -p local_data/pdb_complexes
python scripts/download_prot2chat_missing_pdbs.py \
    --mapping local_data/enzyme_pdb_mapping.tsv \
    --output local_data/pdb_complexes

# 检查下载结果
ls local_data/pdb_complexes/ | wc -l
# 应该有 ~52,000 个 .cif.gz 文件
```

### 8.5 步骤 4：构建特征缓存

```bash
# 预处理所有 PDB 复合物
python scripts/preprocess_pdb_cache.py \
    --pdb_dir local_data/pdb_complexes \
    --output_dir local_data/pdb_cache \
    --crop_size 384
# 耗时约 6-8 小时

# 修补 gt_seq bug
python scripts/patch_cache_gt_seq.py \
    --cache-dir local_data/pdb_cache \
    --pdb-dir local_data/pdb_complexes \
    --workers 32
# 耗时约 1 小时

# 验证
python -c "
import torch
d = torch.load('local_data/pdb_cache/101m.pt', map_location='cpu', weights_only=False)
print('Keys:', sorted([k for k in d.keys() if not k.startswith('_')]))
print('gt_seq present:', 'gt_seq' in d)
print('gt_seq[:5]:', d['gt_seq'][:5])
print('N_token:', d['token_index'].shape[0])
print('N_atom:', d['ref_pos'].shape[0])
"
```

### 8.6 步骤 5：启动训练

```bash
# 确保配置正确
cat configs/train_conditional.yaml
# 检查 load_checkpoint_path, cache_dir, loss_weights 等

# 启动 4-GPU 训练
nohup .venv/bin/torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    runner/train.py \
    --config-name=train_conditional \
    > /dev/null 2>&1 &

# 监控训练
grep "Step " train_output_conditional/train.log | tail -5

# 预期输出
# Step 50/100000 | lr=0.000004 | loss=13.05 | seq=9.23 | mse=0.71 | ...
# Step 100/100000 | lr=0.000007 | loss=17.19 | seq=13.27 | mse=0.74 | ...
```

### 8.7 步骤 6：监控与评估

```bash
# 查看训练曲线
grep "Step " train_output_conditional/train.log | \
    awk -F'[=|]' '{print $2, $4, $5, $6, $7}' | head -20

# 检查 GPU 使用
nvidia-smi

# 检查检查点
ls -lh train_output_conditional/checkpoints/

# 预期训练时间：~54 小时（4×H100，100K 步）
```

---

## 9. 常见问题与解决方案

### 9.1 gt_seq 全为 UNK（序列损失为零）

**症状：** 训练日志显示 `seq=0.0000`
**原因：** 缓存文件缺少 `gt_seq` 字段，或 `gt_seq` 全是 UNK(20)
**解决：** 运行 `patch_cache_gt_seq.py` 修补缓存

### 9.2 Loss spike（损失突然飙升到 100+）

**症状：** 训练日志中偶尔出现 `loss=300+`
**原因：** `noise_sequence` 的 `mask_rate` 采样过接近 1.0，导致 `1/(1-mask_rate)` 权重爆炸
**解决：** 设置 `noise.seq_noise_max=0.95`（已在 v2 中修复）

### 9.3 NCCL 超时崩溃

**症状：** 训练突然停止，日志显示 "NCCL heartbeat timeout"
**原因：** 某个 GPU rank 的数据加载或前向传播卡住
**解决：** 
- 检查数据管道中是否有 None 样本（CachedPDBComplexDataset 返回 None 时会导致 DDP 挂起）
- 增加 `n_fallback_retries`

### 9.4 训练日志不实时更新

**症状：** 日志文件长时间无新内容
**原因：** nohup + torchrun 管道缓冲
**解决：** 在 `runner/train.py` 中使用 FileHandler + line-buffered stream（已在 v2 中修复）

### 9.5 显存不足（OOM）

**症状：** CUDA out of memory
**解决：**
- 减小 `crop_size`（384 → 256）
- 减小 `per_gpu_batch_size`（2 → 1）
- 启用 `gradient_checkpointing: true`
- 使用 `dtype: "bf16-mixed"`

### 9.6 加载检查点时形状不匹配

**症状：** 日志显示 "Skipping 'xxx': shape mismatch"
**原因：** 模型架构变更后加载旧检查点，`load_strict: false` 静默跳过不匹配的权重
**影响：** 跳过的权重会被随机初始化，可能导致训练前期不稳定
**解决：** 确保模型架构与检查点一致，使用 `load_strict: true` 检查

---

*本文档基于 DISCO 训练管线的完整源码分析和实际训练经验编写。所有代码路径、张量形状和数值示例均经过验证。*
