# DISCO 详细解读：从零开始理解蛋白质-配体联合设计模型

> 本文档面向初学者，从基础概念出发，逐步深入解析 DISCO 仓库的每一个细节。

---

## 目录

1. [背景知识：为什么需要 DISCO？](#1-背景知识为什么需要-disco)
2. [DISCO 是什么？核心思想](#2-disco-是什么核心思想)
3. [仓库整体结构](#3-仓库整体结构)
4. [依赖与环境](#4-依赖与环境)
5. [核心概念：数据表示](#5-核心概念数据表示)
6. [数据流水线：从输入 JSON 到特征张量](#6-数据流水线从输入-json-到特征张量)
7. [模型架构详解](#7-模型架构详解)
   - 7.1 [总体架构：DISCO 主模块](#71-总体架构disco-主模块)
   - 7.2 [输入嵌入：InputFeatureEmbedder](#72-输入嵌入inputfeatureembedder)
   - 7.3 [语言模型集成：LMWrapper + DPLM-650M](#73-语言模型集成lmwrapper--dplm-650m)
   - 7.4 [PairformerStack：孪生表示精炼](#74-pairformerstack孪生表示精炼)
   - 7.5 [结构编码器：LigandMPNN](#75-结构编码器ligandmpnn)
   - 7.6 [扩散模块：JointDiffusionModule](#76-扩散模块jointdiffusionmodule)
   - 7.7 [Distogram Head](#77-distogram-head)
8. [注意力机制详解](#8-注意力机制详解)
9. [扩散过程：噪声调度与去噪](#9-扩散过程噪声调度与去噪)
   - 9.1 [结构扩散：EDM 框架](#91-结构扩散edm-框架)
   - 9.2 [序列扩散：MDLM 掩码扩散](#92-序列扩散mdlm-掩码扩散)
10. [推理流程：完整的采样过程](#10-推理流程完整的采样过程)
    - 10.1 [Noisy Guidance：引导生成](#101-noisy-guidance引导生成)
    - 10.2 [PathPlanning 序列解码策略](#102-pathplanning-序列解码策略)
11. [配置系统：Hydra](#11-配置系统hydra)
12. [输入格式：JSON 规范](#12-输入格式json-规范)
13. [输出格式](#13-输出格式)
14. [脚本说明](#14-脚本说明)
15. [关键常量与查找表](#15-关键常量与查找表)
16. [已知局限性](#16-已知局限性)
17. [论文核心贡献总结](#17-论文核心贡献总结)

---

## 1. 背景知识：为什么需要 DISCO？

### 1.1 蛋白质设计的挑战

传统的蛋白质工程依赖于大量实验和专家知识。计算机辅助蛋白质设计的目标是：**给定一个期望的功能，自动设计出具有该功能的蛋白质序列和三维结构**。

### 1.2 配体结合蛋白设计

一个特别重要的场景是设计能与**特定小分子（配体）结合**的蛋白质。应用包括：
- 酶设计（设计能催化特定化学反应的酶）
- 药物靶点设计（设计能结合药物分子的蛋白质）
- 生物传感器设计

传统方法通常分两步：先设计结构，再设计序列。DISCO 的核心创新是**同时**设计序列和结构。

### 1.3 现有方法的局限

- **RFdiffusion**：只能设计结构（骨架），不能同时设计序列
- **ProteinMPNN**：只能在已知结构的基础上设计序列
- **AlphaFold3**：能预测结构，但不能从头设计

DISCO 的目标：**一步到位地联合设计蛋白质序列 + 三维结构，同时考虑配体/RNA/DNA 的存在**。

---

## 2. DISCO 是什么？核心思想

**DISCO** = **DI**screte-continuous **S**equence-structure **CO**-design

核心思想：
1. **连续扩散**（Diffusion）负责生成三维原子坐标（结构）
2. **离散掩码扩散**（Masked Discrete Diffusion）负责生成氨基酸序列
3. 两个过程**同时进行**，互相约束，产生序列-结构协调一致的输出

模型架构基于 **AlphaFold3**（AF3），但做了关键修改：
- 在 Diffusion Module 内部增加了**序列解码器**（Atom Attention Decoder for Sequence）
- 引入了**蛋白质语言模型**（DPLM-650M 或 ESM2）作为序列先验
- 支持配体、RNA、DNA 等非蛋白质分子的条件输入

---

## 3. 仓库整体结构

```
DISCO/
├── pyproject.toml              # Python 项目配置，依赖声明
├── README.md                   # 项目文档
├── .project-root               # rootutils 标记文件（用于路径定位）
│
├── runner/                     # 推理入口点（顶层包）
│   ├── inference.py            # 主推理脚本，@hydra.main 装饰器标记入口
│   ├── dumper.py               # DataDumper：保存 CIF/PDB + JSON 置信度文件
│   └── utils.py                # 辅助工具：打印配置树、生成 biomol 输出行
│
├── disco/                      # 核心模型与数据包
│   ├── __init__.py             # 定义 DISCO_ROOT 路径常量
│   ├── data/                   # 数据处理模块
│   │   ├── constants.py        # 所有残基/原子/元素查找表
│   │   ├── ccd.py              # CCD（化学组件字典）组件加载
│   │   ├── featurizer.py       # 特征化器：token、参考、键、掩码特征
│   │   ├── json_parser.py      # JSON → AtomArray：聚合物/配体构建
│   │   ├── json_to_feature.py  # SampleDictToFeatures：完整特征流水线
│   │   ├── parser.py           # AddAtomArrayAnnot：注释方法
│   │   ├── tokenizer.py        # Token, TokenArray, AtomArrayTokenizer
│   │   ├── task_manager.py     # TaskManager：掩码（训练/推理）
│   │   ├── data_pipeline.py    # 训练数据流水线（未公开）
│   │   ├── infer_data_pipeline.py  # InferenceDataset, get_inference_dataloader
│   │   ├── substructure_perms.py   # 配体对称性/原子排列
│   │   └── utils.py            # save_structure_cif/pdb, CIFWriter 工具
│   │
│   ├── model/                  # 模型定义模块
│   │   ├── disco.py            # DISCO nn.Module（主 AF3 风格模型）
│   │   ├── utils.py            # InferenceNoiseScheduler, 几何工具
│   │   ├── modules/            # 子模块
│   │   │   ├── diffusion.py    # DiffusionConditioning, JointDiffusionModule
│   │   │   ├── pairformer.py   # PairformerBlock, PairformerStack
│   │   │   ├── embedders.py    # 输入嵌入器, 相对位置编码, 傅里叶嵌入
│   │   │   ├── plm.py          # 蛋白质语言模型包装器 + 注册表
│   │   │   ├── transformer.py  # 注意力, 扩散 Transformer, 原子 Transformer
│   │   │   ├── head.py         # DistogramHead
│   │   │   ├── primitives.py   # 自适应层归一化, Transition, Attention
│   │   │   └── lmpnn.py        # LigandMPNN 包装器（结构编码器）
│   │   └── cogen_inference/    # 联合推理引擎
│   │       ├── cogen_generator.py                  # 主扩散采样循环
│   │       ├── cogen_inference_loop_strategies.py  # 策略模式
│   │       ├── cogen_inference_loop_body_impl.py   # 单步去噪实现
│   │       ├── sequence_sampling_strategy.py       # 序列采样策略
│   │       ├── sequence_inference_noise_scheduler.py # 序列噪声调度器
│   │       └── feature_dict_updater.py             # 多进程特征字典更新器
│   │
│   └── utils/                  # 通用工具
│       ├── seed.py             # seed_everything()
│       ├── torch_utils.py      # to_device, cdist, 自动精度转换
│       ├── geometry.py         # random_transform, DistOneHotCalculator
│       ├── distributed.py      # 分布式训练工具
│       ├── metrics.py          # 简单指标聚合器
│       ├── file_io.py          # JSON/pickle 文件 I/O
│       ├── logger.py           # 日志过滤器
│       ├── np_utils.py         # NumPy 辅助函数
│       ├── scatter_utils.py    # Scatter 操作
│       └── seq/                # 序列常量
│           ├── amino_acid_constants.py  # atom37 类型列表
│           ├── rna_constants.py
│           ├── dna_constants.py
│           ├── ligand_constants.py
│           └── res_constant.py
│
├── configs/                    # Hydra 配置文件系统
│   ├── inference.yaml          # 顶层推理配置
│   ├── base.yaml               # 基础配置（列出所有默认值）
│   ├── inference/default.yaml  # 种子, 输出目录, 检查点路径
│   ├── model/default.yaml      # 完整架构超参数
│   ├── experiment/
│   │   ├── designable.yaml     # 开启引导, PathPlanning, lambda=0.1
│   │   └── diverse.yaml        # 无引导, gamma0=1.6（多样性模式）
│   ├── effort/
│   │   ├── max.yaml            # N_step=200, N_cycle=4（最高质量）
│   │   └── fast.yaml           # N_step=100, N_cycle=2（快速模式）
│   ├── fabric/default.yaml     # num_nodes: 1
│   ├── structure_encoder/default.yaml  # LigandMPNN 配置
│   └── sequence_sampling_strategy/    # 序列采样策略配置
│
├── input_jsons/                # 示例输入文件
│   ├── unconditional_config.json  # 无条件生成
│   ├── PLP.json                   # PLP 配体（磷酸吡哆醛）
│   ├── warfarin.json              # 华法林（抗凝血药）
│   ├── NDI.json                   # NDI 配体
│   ├── heme_b.json                # 血红素 B
│   ├── thyroxine.json             # 甲状腺素
│   ├── 6YMC_rna.json              # RNA 条件生成
│   ├── 7S03_dna.json              # DNA 条件生成
│   └── all_priorities_ligands_split_{0,1,2,3}.json  # Studio-179 基准测试
│
├── scripts/                    # 运行脚本
│   ├── common_env.sh           # 设置环境变量
│   ├── activate_env.sh         # 激活虚拟环境
│   ├── setup_env.sh            # 安装依赖（uv sync）
│   ├── doctor_env.sh           # 环境诊断
│   ├── download_assets.sh      # 下载模型权重和 CCD 文件
│   ├── run_inference.sh        # 推理启动脚本
│   └── train_disco.sh          # 占位符（训练代码未发布）
│
├── studio-179/                 # 179 个 SDF 配体文件（基准测试用）
│   └── priority_{1,2,3}/*.sdf
│
└── packages/                   # 第三方包（供应商化）
    ├── LigandMPNN/             # LigandMPNN（dauparas）含 OpenFold 子包
    └── openfold/               # OpenFold 包（结构相关工具）
```

---

## 4. 依赖与环境

### 4.1 Python 版本要求

```
Python >= 3.11, < 3.13
```

### 4.2 包管理器：uv（workspace 模式）

DISCO 使用 `uv` 作为包管理器，并采用**工作区（workspace）**模式管理多个子包：
- `packages/LigandMPNN`
- `packages/openfold`
- `runner`（推理入口）

### 4.3 关键依赖详解

| 包 | 版本 | 用途 |
|---|---|---|
| `torch` | >=2.5 | PyTorch 深度学习框架 |
| `lightning` | >=2.6.0 | 训练框架（继承 PyTorch Lightning） |
| `hydra-core` | >=1.3.2 | 配置管理系统 |
| `deepspeed` | >=0.18.3 | 分布式训练优化（训练用） |
| `transformers` | >=4.50.0 | HuggingFace 变换器（加载 PLM） |
| `fair-esm` | >=2.0.0 | Meta AI 的 ESM 蛋白质语言模型 |
| `biopython` | ==1.83 | 生物信息学工具（固定版本） |
| `biotite` | ==1.4.0 | 原子数组操作（固定版本） |
| `rdkit` | via pdbeccdutils | 化学信息学（配体处理） |
| `gemmi` | >=0.6.7 | 晶体学文件格式（CIF 解析） |
| `huggingface_hub` | latest | 自动下载模型权重 |
| `prody` | >=2.6.1 | 蛋白质动力学分析 |
| `wandb` | >=0.23.1 | 实验追踪（训练用） |

### 4.4 安装流程

```bash
# 1. 克隆仓库
git clone <repo_url>
cd DISCO

# 2. 安装依赖
bash scripts/setup_env.sh

# 3. 激活环境
source scripts/activate_env.sh

# 4. 下载模型权重和 CCD 文件
bash scripts/download_assets.sh

# 5. 诊断环境
bash scripts/doctor_env.sh
```

`setup_env.sh` 内部流程：
1. 调用 `uv sync` 安装所有依赖到 `.venv/`
2. 可选：通过 `TORCH_BACKEND` 环境变量重新安装特定后端的 PyTorch
3. 验证关键包（torch、hydra、lightning、rdkit、huggingface_hub）可以正常导入

`download_assets.sh` 从 HuggingFace 下载：
- `DISCO.pt` — 模型权重检查点
- `components.v20240608.cif` — CCD（化学组件字典）文件
- `components.v20240608.cif.rdkit_mol.pkl` — 预计算的 RDKit 分子缓存

---

## 5. 核心概念：数据表示

理解 DISCO 的数据表示是读懂整个代码库的基础。

### 5.1 AtomArray（原子数组）

DISCO 使用 **biotite** 库的 `AtomArray` 作为分子的核心数据结构。

`AtomArray` 本质上是一个结构化的 NumPy 数组，每一行代表一个原子，每一列代表一种属性：

```
原子索引  chain_id  res_id  res_name  atom_name  element  x      y      z    ...
0        "A"       1       "ALA"     "N"        "N"      1.2    3.4    5.6   ...
1        "A"       1       "ALA"     "CA"       "C"      2.1    4.2    6.1   ...
2        "A"       1       "ALA"     "C"        "C"      3.4    5.1    7.2   ...
...
```

DISCO 在标准 `AtomArray` 基础上添加了**自定义注释**（通过 `parser.py` 中的 `AddAtomArrayAnnot`）：

| 自定义注释 | 类型 | 含义 |
|---|---|---|
| `mol_type` | str | "protein" / "rna" / "dna" / "ligand" |
| `centre_atom` | bool | 是否是 token 的中心原子（蛋白质=CA，配体=所有原子） |
| `distogram_rep_atom` | bool | 是否是 distogram 预测的代表原子 |
| `cano_seq_resname` | str | 规范化残基名（被掩码的位置 = "MSK"） |
| `ref_pos` | float[3] | 来自 CCD ideal 构象的参考坐标 |
| `ref_charge` | float | 参考部分电荷 |
| `ref_mask` | bool | 参考坐标是否有效 |

### 5.2 Token（令牌）系统

DISCO 的模型在 **token** 层面操作，而非原子层面。Token 的定义：
- **蛋白质残基**：每个氨基酸 = 1 个 token（无论它有多少个原子）
- **RNA/DNA 核苷酸**：每个核苷酸 = 1 个 token
- **配体原子**：每个**原子** = 1 个 token（配体以原子为粒度）

这个设计允许模型以统一的方式处理不同类型的分子。

`tokenizer.py` 中的 `Token` 类表示单个 token，`TokenArray` 是 token 列表，`AtomArrayTokenizer` 将 `AtomArray` 转换为 `TokenArray`。

### 5.3 残基类型编码

`constants.py` 定义了所有分子量的索引：

```python
# 氨基酸：0-20（20种标准 + UNK）
PRO_STD_RESIDUES = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4,
    "GLN": 5, "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9,
    "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
    "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
    "UNK": 20
}

# RNA 核苷酸：21-25
RNA_STD_RESIDUES = {"A": 21, "G": 22, "C": 23, "U": 24, "N": 29}

# DNA 核苷酸：25-30
DNA_STD_RESIDUES = {"DA": 25, "DG": 26, "DC": 27, "DT": 28, "DN": 30}

# 掩码残基：31（推理时被设计的位置）
MASK_STD_RESIDUES = {"MSK": 31}
# 因此 MASK_TOKEN_IDX = 31
```

**总共 32 种 token 类型（0-31）**，其中 31 是推理时的掩码位置。

---

## 6. 数据流水线：从输入 JSON 到特征张量

完整的数据处理流程如下：

```
输入 JSON 文件
     │
     ▼
json_parser.py
（build_polymer / rdkit_mol_to_atom_array）
     │ 每个entity → AtomArray
     ▼
json_to_feature.py（SampleDictToFeatures）
  ├── add_entity_atom_array()      → 各实体的 AtomArray
  ├── build_full_atom_array()      → 拼接所有实体，分配链标签 A/B/C...
  ├── add_bonds_between_entities() → 添加实体间共价键（如辅基）
  ├── mse_to_met()                → 硒代甲硫氨酸 → 甲硫氨酸
  ├── add_atom_array_attributes()  → AddAtomArrayAnnot 注释
  └── get_feature_dict()
        ├── AtomArrayTokenizer → TokenArray
        ├── TaskManager（掩码蛋白质待设计位置）
        └── Featurizer.get_all_input_features()
              ├── get_token_features()    → restype, residue_index, ...
              ├── get_reference_features() → ref_pos, ref_element, ...
              ├── get_bond_features()     → 键连接矩阵
              ├── get_extra_features()    → atom_to_token_idx, mol_type 掩码
              └── get_mask_features()    → 骨架原子掩码, distogram 掩码...
                         │
                         ▼
                   特征字典（Python dict of tensors）
                         │
                         ▼
              InferenceDataset.__getitem__()
              ├── 添加 dummy MSA 特征（全零）
              ├── 创建 masked_prot_restype 张量
              └── 创建 prot_residue_mask 布尔掩码
                         │
                         ▼
                   DataLoader → 批次特征
```

### 6.1 JSON 解析器详解（json_parser.py）

#### 蛋白质链解析（build_polymer）

对于 `proteinChain`：
1. 遍历序列字符串中的每个字符（`-` 表示待设计位置，字母表示固定氨基酸）
2. 对每个残基，从 CCD 查找标准原子列表（包括参考坐标）
3. 通过 C-N 肽键连接相邻残基，删除多余的"离去原子"（-OH 基团）
4. 构建连续聚合物的 `AtomArray`

#### 配体解析（rdkit_mol_to_atom_array + add_reference_features）

对于配体（SMILES 字符串或 SDF 文件）：
1. 用 RDKit 解析分子
2. 生成 3D 坐标（如果没有的话，用 ETKDG 方法嵌入）
3. 转换为 `AtomArray`，每个原子为一个 token
4. 附加 `ref_pos`（来自 CCD 理想构象或 SMILES 3D 嵌入）、`ref_charge`、`ref_mask`

#### CCD 模块（ccd.py）

CCD = Chemical Component Dictionary（化学组件字典），是 PDB 数据库中所有标准残基和小分子的定义库。

DISCO 使用 `components.v20240608.cif` 文件（约 500MB），包含：
- 每个分子的原子类型、化学键、理想三维坐标
- 部分电荷
- 分子类型（氨基酸、RNA、DNA、小分子等）

`ccd.py` 中的关键函数：
- `get_component_atom_array(code)` → 给定 CCD 代码，返回 AtomArray（包含 ref_pos）
- `get_ccd_ref_info(code)` → 返回参考位置、电荷、掩码
- `get_mol_type(code)` → 返回分子类型字符串

使用 `@functools.lru_cache` 缓存结果，避免重复加载。

### 6.2 特征化器详解（featurizer.py）

`Featurizer.get_all_input_features()` 生成以下特征（均为张量）：

#### Token 特征（get_token_features）

| 特征名 | 形状 | 含义 |
|---|---|---|
| `restype` | `[N_token, 32]` | 残基类型的 one-hot 编码（32 维） |
| `residue_index` | `[N_token]` | 残基序列位置索引 |
| `asym_id` | `[N_token]` | 不对称单元 ID（标识链） |
| `entity_id` | `[N_token]` | 实体 ID（标识分子类型） |
| `sym_id` | `[N_token]` | 对称相关的 ID |
| `token_index` | `[N_token]` | 全局 token 索引 |

#### 参考特征（get_reference_features）

这些特征来自 CCD 理想构象，而不是实际坐标（推理时不知道真实坐标）：

| 特征名 | 形状 | 含义 |
|---|---|---|
| `ref_pos` | `[N_atom, 3]` | CCD 理想构象中的原子坐标 |
| `ref_mask` | `[N_atom]` | 该原子的参考坐标是否有效 |
| `ref_element` | `[N_atom, 128]` | 元素的 one-hot 编码（128 种元素） |
| `ref_charge` | `[N_atom]` | 参考部分电荷 |
| `ref_atom_name_chars` | `[N_atom, 4, 64]` | 原子名称字符编码（最多4个字符） |
| `ref_space_uid` | `[N_atom]` | 用于跟踪"参考空间"的唯一 ID |

#### 键特征（get_bond_features）

| 特征名 | 形状 | 含义 |
|---|---|---|
| `token_bonds` | `[N_token, N_token]` | token 间的化学键邻接矩阵 |

#### 掩码特征（get_mask_features）

| 特征名 | 形状 | 含义 |
|---|---|---|
| `backbone_atom_mask` | `[N_token]` | 该 token 是否有骨架原子（N/CA/C/O） |
| `plddt_m_rep_atom_mask` | `[N_token]` | pLDDT 计算用代表原子掩码 |
| `distogram_rep_atom_mask` | `[N_token]` | Distogram 用代表原子掩码 |
| `bond_mask` | `[N_token]` | 键合掩码 |

### 6.3 TaskManager：推理时的掩码处理（task_manager.py）

在推理阶段，蛋白质中用 `-` 标记的位置（待设计）需要特殊处理：

1. **mask_ref_information**：将这些位置的参考坐标替换为 "MSK" 残基的参考坐标，并可选应用随机旋转平移（数据增强）
2. **mask_side_chains**：删除侧链原子，只保留骨架原子（N/CA/C/O），将残基名设为 "MSK"

这确保模型在推理时不会"看到"它应该预测的信息。

---

## 7. 模型架构详解

### 7.1 总体架构：DISCO 主模块

`disco/model/disco.py` 中的 `DISCO` 类实现了 **AlphaFold3 Algorithm 1** 的整体流程，但做了重要修改。

```python
class DISCO(nn.Module):
    def __init__(self, config):
        # 关键子模块
        self.input_embedder = InputFeatureEmbedder(...)   # 输入嵌入
        self.relative_position_encoding = RelativePositionEncoding(...)  # 相对位置编码
        self.lm_module = LMWrapper(lm_name="dplm_650M", freeze_lm=True)  # 冻结的 PLM
        self.pairformer_stack = PairformerStack(n_blocks=8, ...)  # 孪生表示精炼
        self.structure_encoder = LigandMPNN(...)  # 可选：结构编码器（循环用）
        self.diffusion_module = JointDiffusionModule(...)  # 联合扩散模块
        self.distogram_head = DistogramHead(c_z=128, no_bins=64)  # 距离图预测头
```

**关键维度**：
- `c_s = 384`：单表示（single representation）维度
- `c_z = 128`：孪生表示（pair representation）维度
- `c_token = 384`：token 嵌入维度
- `c_atom = 128`：原子嵌入维度
- `c_atompair = 16`：原子对嵌入维度
- `sigma_data = 16.0`：扩散过程的数据标准差参数

**推理函数调用流程**：

```
DISCO.forward(input_feature_dict, sample2feat)
    │
    ├─► get_pairformer_output()    # N_cycle 次循环，精炼 s/z 表示
    │       └─► 每次循环：
    │           ├─► InputFeatureEmbedder（初始嵌入）
    │           ├─► RelativePositionEncoding（相对位置偏置）
    │           ├─► LMWrapper（PLM 提取蛋白质表示）
    │           └─► PairformerStack（8 层 Pairformer 精炼）
    │
    └─► main_inference_loop()     # 扩散采样
            └─► sample_diffusion_cogen()
                    └─► N_step 步去噪循环
```

### 7.2 输入嵌入：InputFeatureEmbedder

实现 AlphaFold3 **Algorithm 2**（输入嵌入器）。

输入特征 → 嵌入向量的映射：

```
token 级特征 → Linear → c_s 维单表示 s
原子级特征  → Linear → c_atom 维原子表示
原子对特征  → Linear → c_atompair 维原子对表示

然后通过 AtomAttentionEncoder（Algorithm 5）
将原子表示聚合到 token 级别，加入 s
```

**相对位置编码（RelativePositionEncoding，Algorithm 3）**：

计算 token 对 `(i, j)` 之间的相对位置关系，转换为 `c_z` 维嵌入，加入孪生表示 `z`：

```python
rel_pos = clamp(residue_index[i] - residue_index[j] + 32, 0, 64)
# 编码为 one-hot 然后 Linear 投影到 c_z
```

同时编码：
- 同链/异链（asym_id 是否相同）
- 同实体（entity_id 是否相同）
- 键合关系（token_bonds）

### 7.3 语言模型集成：LMWrapper + DPLM-650M

这是 DISCO 相比原始 AF3 的关键新增。

#### 支持的语言模型（MODEL_REGISTRY）

```python
MODEL_REGISTRY = {
    # DPLM（Diffusion Protein Language Model）系列
    "dplm_150M": "airkingbd/dplm_150m",
    "dplm_650M": "airkingbd/dplm_650m",   # 默认使用
    "dplm_3B": "airkingbd/dplm_3b",
    
    # ESM2 系列（Meta AI）
    "esm2_8M":   "facebook/esm2_t6_8M_UR50D",
    "esm2_35M":  "facebook/esm2_t12_35M_UR50D",
    "esm2_150M": "facebook/esm2_t30_150M_UR50D",
    "esm2_650M": "facebook/esm2_t33_650M_UR50D",
    "esm2_3B":   "facebook/esm2_t36_3B_UR50D",
    "esm2_15B":  "facebook/esm2_t48_15B_UR50D",
    
    # EvoFlow 系列
    "evoflow_650M": ...,
}
```

默认使用 `dplm_650M`（DPLM-650M），它是专门为蛋白质生成设计的扩散语言模型，有 6.5 亿参数。

#### LMWrapper 的工作流程

```python
class LMWrapper(nn.Module):
    def forward(self, feature_dict, s, z):
        # 1. 从特征字典中提取蛋白质 token 子矩阵
        prot_mask = feature_dict["is_protein"]  # 布尔掩码
        prot_seq = feature_dict["restype"][prot_mask]  # 只看蛋白质残基
        
        # 2. 通过冻结的 PLM
        with torch.no_grad():
            plm_output = self.plm(prot_seq)
        
        # 3. 提取隐藏状态（各层的加权和）
        hidden_states = plm_output.hidden_states  # [n_layers, N_prot, d_plm]
        weighted_sum = sum(layer_weight[l] * hidden_states[l] for l in range(N_layers))
        
        # 4. 投影到 c_s 维度
        prot_s_update = self.proj_single(weighted_sum)  # [N_prot, c_s]
        
        # 5. 将 PLM 的注意力图插入孪生表示的蛋白质-蛋白质子块
        attn_map = plm_output.attentions  # [N_heads, N_prot, N_prot]
        z_prot_block += self.proj_pair(attn_map)  # 更新蛋白质对表示
        
        # 将更新写回 s 和 z
        s[prot_mask] += prot_s_update
        add_matrix_subset(z, prot_mask, z_prot_block)
```

**为什么冻结 PLM？**

语言模型在大量无标签蛋白质序列上预训练，学到了丰富的序列语义。冻结它可以：
1. 减少训练参数量（节省显存和计算）
2. 保留语言模型的泛化能力
3. 防止在数量较少的结构数据上过拟合

### 7.4 PairformerStack：孪生表示精炼

实现 **AlphaFold3 Algorithm 17**。由 8 个 `PairformerBlock` 串联组成。

#### PairformerBlock 内部（Algorithm 17）

每个 block 按顺序执行以下操作：

```
输入：s [N_token, c_s], z [N_token, N_token, c_z]

# ── 孪生表示更新（更新 z）──────────────────────────────────
1. z ← z + TriangleMultiplicationOutgoing(z)   # "三角乘法-外出"
2. z ← z + TriangleMultiplicationIncoming(z)   # "三角乘法-进入"
3. z ← z + TriangleAttentionStartingNode(z)    # "三角注意力-起点"
4. z ← z + TriangleAttentionEndingNode(z)      # "三角注意力-终点"
5. z ← z + pair_transition(z)                  # 前馈网络

# ── 单表示更新（更新 s，以 z 为偏置）────────────────────────
6. s ← s + AttentionPairBias(s, z)             # 带孪生偏置的注意力
7. s ← s + single_transition(s)               # 前馈网络

输出：更新后的 s, z
```

#### 三角乘法（Triangle Multiplication）

这是 AF3/AF2 的核心操作之一，强制执行"三角不等式"约束（如果 i 与 j 相关，j 与 k 相关，则 i 与 k 也应该相关）：

```
# Outgoing（外出）：更新 z[i,j]
message = sum_k( gate_k * z[i,k] ⊙ z[j,k] )  # 对第三个 token k 求和
z[i,j] ← LN(z[i,j] + gate_ij * Linear(message))

# Incoming（进入）：更新 z[i,j]
message = sum_k( gate_k * z[k,i] ⊙ z[k,j] )  # 对所有路过 (i,j) 的 k 求和
```

#### AttentionPairBias（Algorithm 24）

在更新单表示 `s` 时，使用孪生表示 `z` 作为注意力偏置：

```python
# 计算注意力权重
attn_logits = QK^T / sqrt(d_head) + Linear(z)  # 加入孪生偏置
attn_weights = softmax(attn_logits)

# 更新单表示
s ← gate * (attn_weights @ V)
```

这让模型在更新每个 token 的表示时，能够考虑到 token 对之间的关系信息。

#### 激活检查点（Activation Checkpointing）

训练时每个 PairformerBlock 都启用激活检查点，这是一种用计算换显存的技术：不保存中间激活值，而是在反向传播时重新计算。这允许在有限显存上训练更大模型。

### 7.5 结构编码器：LigandMPNN

`disco/model/modules/lmpnn.py` 中的 `LigandMPNN` 类包装了 `ProteinMPNN_Ligand`（来自 `packages/LigandMPNN`）。

**用途**：在扩散采样的每一步，将当前 3D 坐标编码为序列表示，注入单表示 `s`，为下一步预测提供结构信息。

**工作流程**：
1. 将 DISCO 内部格式转换为 LigandMPNN 格式（包含 backbone 坐标、配体坐标、链信息）
2. 通过 3 层消息传递图神经网络（ProteinMPNN_Ligand）提取结构编码
3. 将编码（hidden_dim=128）通过 LayerNorm + Linear 投影到 c_s=384 维

```python
# LigandMPNN 内部：3 层消息传递
for layer in self.mpnn_layers:
    h_V = layer(h_V, h_E, topology)  # 更新节点特征（残基表示）

# 投影到 c_s
lmpnn_s = self.proj(LN(h_V))  # [N_prot, c_s]
```

**为什么需要 LigandMPNN？**

在扩散迭代过程中，当前的结构坐标不断变化。LigandMPNN 作为"结构感知"的编码器，让序列预测知道当前结构是什么样的，从而使序列与结构互相协调。

### 7.6 扩散模块：JointDiffusionModule

这是 DISCO 最核心的创新之一。

#### DiffusionModule（Algorithm 20 基础版）

```python
class DiffusionModule(nn.Module):
    """
    给定噪声坐标 x_noisy 和噪声水平 sigma，预测去噪后的坐标 x_denoised
    """
    def forward(self, x_noisy, sigma, feature_dict, s, z):
        # 1. EDM 标准化缩放
        r_noisy = x_noisy / sqrt(sigma_data^2 + sigma^2)
        
        # 2. DiffusionConditioning（Algorithm 21）：用噪声水平条件化 s 和 z
        s_cond = s + FourierEmbedding(sigma)  # 傅里叶噪声嵌入
        
        # 3. AtomAttentionEncoder（Algorithm 5）：从原子特征到原子嵌入
        atom_embeds = AtomAttentionEncoder(r_noisy, ref_features, s_cond)
        
        # 4. DiffusionTransformer（Algorithm 23）：24 层注意力
        atom_embeds = DiffusionTransformer(atom_embeds, s_cond, z)
        
        # 5. AtomAttentionDecoder（Algorithm 6）：原子嵌入 → 3D 坐标更新
        r_update = AtomAttentionDecoder(atom_embeds)
        
        # 6. EDM 输出缩放
        x_denoised = x_skip * x_noisy + x_out * r_update
        
        return x_denoised
```

**EDM 缩放公式**（Karras et al. 2022）：

```python
s_ratio = sigma / sigma_data
# 跳连权重
x_skip = 1 / (1 + s_ratio^2)
# 输出权重  
x_out = sigma / sqrt(1 + s_ratio^2) * sigma_data / sigma_data
```

这个缩放确保网络在低噪声（sigma → 0）时近似恒等映射，在高噪声时更多依赖网络输出。

#### JointDiffusionModule（DISCO 扩展版）

```python
class JointDiffusionModule(DiffusionModule):
    """
    在 DiffusionModule 基础上增加序列解码器
    """
    def __init__(self):
        super().__init__()
        # 额外增加：序列解码头（用于预测氨基酸类型）
        self.atom_attention_decoder_seq = AtomAttentionDecoder(
            output_dim=20  # 输出 20 种氨基酸的 logit（而不是 3D 坐标）
        )
    
    def f_forward(self, ...):
        # ... 与 DiffusionModule 相同的前半部分 ...
        
        # 结构解码：原子嵌入 → 3D 坐标更新
        r_update = self.atom_attention_decoder_coord(atom_embeds)
        
        # 序列解码：相同的原子嵌入 → 氨基酸 logit
        seq_logits = self.atom_attention_decoder_seq(atom_embeds)
        # [N_prot, 20] 每个蛋白质残基的 20 种氨基酸得分
        
        return r_update, seq_logits
```

#### DiffusionTransformer（Algorithm 23，24 层）

扩散 Transformer 是一个 **24 层**的 Transformer，配置：
- `n_heads = 16`（16 个注意力头）
- `c_token = 768`（扩散过程中的 token 维度，大于 PairformerStack 的 c_token=384）

每个 `DiffusionTransformerBlock` 包含：
1. **AttentionPairBias**：带孪生偏置的多头注意力
2. **ConditionedTransitionBlock**：以噪声水平条件化的 MLP（通过 AdaptiveLN 实现）

**AdaptiveLayerNorm（Algorithm 26）**：

标准 LayerNorm 的升级版，允许噪声水平全局调节归一化的缩放和偏移：

```python
class AdaptiveLayerNorm(nn.Module):
    def forward(self, x, s):  # s 是噪声条件信号
        # 从条件信号中计算 scale 和 shift
        scale, shift = self.linear(s).chunk(2, dim=-1)
        # 标准 LayerNorm，然后用条件调整
        x = LayerNorm(x) * (1 + scale) + shift
        return x
```

#### AtomTransformer（Algorithm 7）：局部窗口注意力

对于原子级别的注意力，使用**局部窗口注意力**（避免全局 O(N²) 复杂度）：

```python
# 仅在局部窗口内做注意力
n_queries = 32   # 每次查询 32 个原子
n_keys = 128     # 每个窗口最多 128 个键
```

这大幅降低了计算复杂度，允许处理含有大量原子的配体或长蛋白质序列。

#### FourierEmbedding（Algorithm 22）：噪声水平编码

将标量噪声水平 σ 编码为向量：

```python
class FourierEmbedding(nn.Module):
    def __init__(self, n_fourier=256):
        # 随机采样频率 w 和偏移 b（固定，不训练）
        self.w = torch.randn(n_fourier)
        self.b = torch.rand(n_fourier)
    
    def forward(self, sigma):
        # cos(2π(σ·w + b))，得到 n_fourier 维向量
        return torch.cos(2 * pi * (sigma * self.w + self.b))
```

这让模型知道当前处于扩散的哪个阶段（高噪声 vs 低噪声），从而调整预测行为。

### 7.7 Distogram Head

`head.py` 中的 `DistogramHead` 预测 token 对之间的距离分布：

```python
class DistogramHead(nn.Module):
    def __init__(self, c_z=128, no_bins=64):
        self.linear = nn.Linear(c_z, no_bins)  # 64 个距离区间
    
    def forward(self, z):
        # z: [N_token, N_token, c_z]
        # 对称化，并预测距离 logit
        logits = self.linear((z + z.permute(1, 0, 2)) / 2)
        return logits  # [N_token, N_token, 64]
```

Distogram 的距离范围通常是 2-22Å，分为 64 个区间。用于辅助训练损失（监督 Cα-Cα 距离分布）。

---

## 8. 注意力机制详解

DISCO 中有多种注意力机制，各有其用途：

### 8.1 标准多头注意力（primitives.py）

```python
class Attention(nn.Module):
    def forward(self, q, k, v, bias=None):
        # Q, K, V 投影
        Q = self.proj_q(q)  # [N, n_heads, d_head]
        K = self.proj_k(k)
        V = self.proj_v(v)
        
        # 缩放点积注意力
        attn = softmax(QK^T / sqrt(d_head) + bias)
        
        # 门控（DISCO 特有）
        gate = sigmoid(self.proj_gate(q))
        output = gate * (attn @ V)
        
        return self.proj_out(output)
```

## 8.2 三角注意力（Triangle Attention）

分为两种方向：

**TriangleAttentionStartingNode**：对 `z[i,:]`（第 i 行）做注意力，偏置来自 `z[:,:]`
**TriangleAttentionEndingNode**：对 `z[:,j]`（第 j 列）做注意力，偏置来自 `z[:,:]`

这确保距离矩阵满足三角不等式。

---

## 9. 扩散过程：噪声调度与去噪

### 9.1 结构扩散：EDM 框架

DISCO 使用 **EDM（Elucidated Diffusion Models，Karras et al. 2022）**框架来扩散 3D 坐标。

#### 正向过程（加噪）

```
坐标 x₀ → 加入高斯噪声 → x_t = x₀ + ε，ε ~ N(0, σ(t)² I)
```

其中 σ(t) 从大（高噪声）到小（低噪声）：σ(0) = σ_max = 160.0，σ(T) = σ_min = 4e-4

#### 噪声调度（InferenceNoiseScheduler，model/utils.py）

使用幂律噪声调度（EDM 默认）：

```python
def __call__(self, N_step=200):
    """生成 N_step 步噪声水平序列"""
    t_steps = torch.linspace(0, 1, N_step)
    # 幂律插值
    sigma = (
        sigma_data * (s_max^(1/rho) + t * (s_min^(1/rho) - s_max^(1/rho)))^rho
    )
    return sigma  # 从 160.0 → 4e-4 的序列
```

默认参数：
- `s_max = 160.0`（最大噪声水平，约等于整个蛋白质的尺寸）
- `s_min = 4e-4`（最小噪声水平，接近原子尺寸）
- `ρ = 7`（ρ 控制调度的形状；ρ=7 是 EDM 论文的推荐值）
- `sigma_data = 16.0`（训练数据坐标的估计标准差）

#### 反向过程（去噪）：随机微分方程采样

每步去噪：

```python
# 1. 注入额外噪声（随机性）
gamma = min(gamma0, sqrt(2) - 1)  # 0.8（可控随机性强度）
sigma_hat = sigma * (1 + gamma)
x_hat = x_t + sqrt(sigma_hat^2 - sigma^2) * noise_scale_lambda * randn()

# 2. 预测去噪后的坐标 x0
x0_pred = model(x_hat, sigma_hat)

# 3. Euler/Heun 积分更新
d = (x_hat - x0_pred) / sigma_hat  # 当前"方向"（分数）
x_next = x_hat + (sigma_next - sigma_hat) * d  # Euler 步
```

关键参数（configs/model/default.yaml）：
- `N_step = 200`（采样步数，max 模式）
- `gamma0 = 0.8`（随机性控制，`diversemode使用1.6`)
- `noise_scale_lambda = 1.003`（噪声缩放，designable 模式用 0.1 以减少随机性）
- `step_scale_eta = 1.5`（步长缩放，控制 Euler 步长）

### 9.2 序列扩散：MDLM 掩码扩散

序列使用**掩码离散语言模型（MDLM，Sahoo et al. 2024）**框架，这是离散空间的扩散等价物。

#### 基本思想

```
正向过程：逐渐将氨基酸 token 替换为 [MASK]（token 31）
反向过程：从全部 [MASK] 开始，逐步预测真实氨基酸
```

与连续扩散类比：
- 连续：x_t = x_0 + ε → 连续加随机噪声
- 离散：seq_t = seq_0 with p(t) fraction MASKED → 离散掩码

#### 序列噪声调度（sequence_inference_noise_scheduler.py）

控制每步应该有多少比例是 [MASK]：

**多项式调度（默认）**：
```python
def get_mask_fraction(perc_done):
    # perc_done: 完成的推理进度 (0→1)
    t = 1 - perc_done^power  # power=2（默认），二次衰减
    return t  # 从 1（全掩码）到 0（全预测）
```

**余弦调度**：
```python
t = cos(π * perc_done / 2)^power
```

**幂比率调度**：
```python
t = (base^perc_done - 1) / (base - 1)
```

---

## 10. 推理流程：完整的采样过程

### 10.1 主推理循环

```
sample_diffusion_cogen() 调用流程：

1. 初始化
   ├── x_t = N(0, σ_max² I)     # 纯随机结构坐标
   └── seq = [MASK, MASK, ...]   # 纯掩码序列

2. 建立 FeatureDictUpdater（多进程池：序列更新时重建特征字典）

3. 获取初始 Pairformer 表示
   └── s, z = PairformerStack(initial_seq)

4. 扩散去噪循环（N_step=200 步）
   └── 每步 InferenceLoopImpl.execute()：
       │
       ├─ a. 用当前 (x_t, seq_t) 更新 Pairformer 表示
       │      └── IF structure_encoder 可用：
       │              LigandMPNN → s_struct_update → s
       │          ELSE: 用 seq_t 重新跑 PLM + Pairformer
       │
       ├─ b. 模型预测（predict_x0_seq_struct）
       │      └── JointDiffusionModule(x_t, sigma_t, s, z)
       │              → x0_struct_pred    # 预测的 clean 结构坐标
       │              → x0_seq_logits     # 预测的氨基酸 logit [N_prot, 20]
       │
       ├─ c. （可选）Noisy Guidance（见 10.1）
       │
       ├─ d. 结构更新（Euler/Heun 积分）
       │      └── x_{t+1} = x_t + (σ_{t+1} - σ_t) * (x_t - x0_pred) / σ_t
       │
       ├─ e. 序列更新（sequence sampling strategy，见 10.2）
       │      └── seq_{t+1} = sample(x0_seq_logits, mask_fraction(t+1))
       │
       └─ f. 如果序列更新了：重建特征字典（FeatureDictUpdater）

5. 返回 (x0_struct, seq)
```

### 10.1 Noisy Guidance：引导生成

类似于**分类器自由引导（CFG，Classifier-Free Guidance）**，但作用于噪声坐标。

**核心思想**：通过对比两次模型预测（一次条件式，一次非条件式），放大条件信息的影响。

```python
if noisy_guidance.enabled and guidance_start_frac < perc_done < guidance_end_frac:
    # 第一次前向传播：正常噪声水平（条件式预测）
    x0_cond, seq_cond = model(x_t, sigma_t, ...)
    
    # 第二次前向传播：更高噪声水平（弱条件预测）
    sigma_uncond = sigma_t * (1 + delta)  # 更高噪声 → 约等于"无条件"
    x0_uncond, seq_uncond = model(x_t, sigma_uncond, ...)
    
    # CFG 融合
    x0_guided_struct = x0_uncond + omega_struct * (x0_cond - x0_uncond)
    x0_guided_seq = x0_uncond_seq + omega_seq * (x0_cond_seq - x0_uncond_seq)
```

配置（configs/experiment/designable.yaml）：
```yaml
noisy_guidance:
  enabled: true
  omega_struct: 1.5   # 结构引导强度（>1 强化条件信息）
  omega_seq: 2.0      # 序列引导强度
  guidance_start_frac: 0.3  # 在推理进度 30% 时开始引导
  guidance_end_frac: 0.8    # 在推理进度 80% 时停止引导
```

**为什么只在 30%-80% 时段引导？**
- 前期（0-30%）：结构整体布局，引导可能过度约束
- 中期（30-80%）：细节形成的关键阶段，引导效果最好
- 后期（80-100%）：精修阶段，让模型自由精炼

### 10.2 PathPlanning 序列解码策略

有两种序列采样策略：

#### VanillaMDLMSamplingStrategy（基础版，configs/sequence_sampling_strategy/default.yaml）

标准 MDLM 解码：

```python
def sample_step(self, seq_t, logits, mask_fraction):
    # 对所有掩码位置，从 logit 分布中采样新 token
    probs = softmax(logits / temperature)
    new_tokens = multinomial(probs)
    
    # 以 mask_fraction 比例保留掩码
    keep_mask = bernoulli(mask_fraction)
    seq_next = where(keep_mask, MASK, new_tokens)
    return seq_next
```

#### PathPlanningSamplingStrategy（高级版，configs/sequence_sampling_strategy/path_planning.yaml）

基于置信度的**有序揭示**策略：

```python
def sample_step(self, seq_t, logits, mask_fraction):
    # 1. 计算每个掩码位置的"得分"（揭示优先级）
    if score_type == "confidence":
        # 使用模型置信度（最高 logit 的 softmax 概率）
        scores = max(softmax(logits), dim=-1)
    elif score_type == "random":
        # 随机打分
        scores = uniform_random()
    
    # 2. 确定本步需要揭示多少位置
    n_currently_masked = count(seq_t == MASK)
    n_target_masked = round(total_n * mask_fraction)
    n_to_unmask = n_currently_masked - n_target_masked
    
    # 3. 选择得分最高的 n_to_unmask 个位置进行揭示
    top_positions = argsort(scores, descending=True)[:n_to_unmask]
    
    # 4. 在揭示位置，从 logit 分布中采样（可选自适应温度）
    if entropy_adaptive_temp:
        # 对高置信度位置使用更高温度（增加多样性）
        temp_per_pos = base_temp * (1 + beta * confidence_score)
    
    new_tokens = multinomial(softmax(logits[top_positions] / temp_per_pos))
    
    # 5. 更新序列
    seq_next = seq_t.clone()
    seq_next[top_positions] = new_tokens
    
    return seq_next
```

配置（configs/sequence_sampling_strategy/path_planning.yaml）：
```yaml
score_type: random        # "random" 或 "confidence"
logits_temp: 0.8          # 基础采样温度
entropy_adaptive_temp: true  # 是否使用自适应温度
allow_remasking: false    # 是否允许已揭示位置重新被掩码
should_ensure_unmasked_stay: true  # 确保固定位置不被掩码
```

---

## 11. 配置系统：Hydra

DISCO 使用 **Hydra**（Meta AI 的配置管理框架）管理所有超参数。

### 11.1 配置层次结构

```
configs/
├── base.yaml           # 最基础的配置，列出 defaults
├── inference.yaml      # 推理特有的配置，覆盖 base.yaml
└── 各模块配置...
```

`base.yaml` 通过 `defaults` 列表组合多个子配置：
```yaml
defaults:
  - _self_
  - model: default      # 使用 configs/model/default.yaml
  - inference: default  # 使用 configs/inference/default.yaml
  - experiment: designable  # 使用 configs/experiment/designable.yaml
  - effort: max         # 使用 configs/effort/max.yaml
  - fabric: default
  - structure_encoder: default
  - sequence_sampling_strategy: path_planning
```

### 11.2 关键配置参数（model/default.yaml）

```yaml
# 整体架构
lm_name: dplm_650M        # 使用的语言模型
n_blocks: 8               # Pairformer 层数
c_s: 384                  # 单表示维度
c_z: 128                  # 孪生表示维度
c_token: 384              # token 嵌入维度
sigma_data: 16.0          # 扩散数据标准差

# Pairformer
pairformer:
  n_blocks: 8             # 8 层
  n_heads: 16             # 16 个注意力头
  dropout: 0.25           # 训练时的 dropout

# 扩散 Transformer
diffusion_module:
  transformer:
    n_blocks: 24          # 24 层
    n_heads: 16           # 16 个注意力头
  c_token: 768            # 扩散 Transformer 内部维度

# 采样
sample_diffusion:
  N_step: 200             # 去噪步数
  N_cycle: 4              # Pairformer 循环次数
  gamma0: 0.8             # 随机性控制
  noise_scale_lambda: 1.003  # 噪声缩放
  step_scale_eta: 1.5     # 步长缩放

# 噪声调度
inference_noise_scheduler:
  s_max: 160.0            # 最大噪声
  s_min: 4e-4             # 最小噪声
  rho: 7                  # 调度指数
  sigma_data: 16.0
```

### 11.3 实验模式

| 模式 | 配置文件 | 特点 |
|---|---|---|
| `designable` | experiment/designable.yaml | 开启 Noisy Guidance，PathPlanning，noise_scale_lambda=0.1，适合获得高质量可设计结构 |
| `diverse` | experiment/diverse.yaml | 无 Guidance，gamma0=1.6（更随机），适合探索多样化结构空间 |

### 11.4 计算模式

| 模式 | 配置文件 | 参数 |
|---|---|---|
| `max` | effort/max.yaml | N_step=200, N_cycle=4（最高质量，最慢） |
| `fast` | effort/fast.yaml | N_step=100, N_cycle=2（快速模式） |

### 11.5 环境变量控制

```bash
DISCO_EXPERIMENT=designable  # 实验模式
DISCO_EFFORT=max             # 计算强度
DISCO_SEEDS="[0,1,2,3,4]"   # 随机种子列表
DISCO_CHECKPOINT=/path/to/DISCO.pt  # 模型权重路径
LAYERNORM_TYPE=fast_layernorm       # 启用优化的 LayerNorm
```

---

## 12. 输入格式：JSON 规范

每个输入 JSON 文件是一个**列表**，每个元素代表一个设计任务。

### 12.1 基本结构

```json
[
  {
    "name": "my_design_001",
    "sequences": [
      { "entity1": ... },
      { "entity2": ... }
    ],
    "covalent_bonds": []
  }
]
```

### 12.2 实体类型

#### 蛋白质链（proteinChain）

```json
{
  "proteinChain": {
    "sequence": "-------MKVFGE--------",
    "count": 1
  }
}
```

- `-`（破折号）= 待设计位置（对应 MASK_TOKEN_IDX=31）
- 字母 = 固定的氨基酸（不会被改变）
- `count`：该链的拷贝数（对称设计用）

#### RNA 序列（rnaSequence）

```json
{
  "rnaSequence": {
    "sequence": "AUCGAUCG",
    "count": 1
  }
}
```

#### DNA 序列（dnaSequence）

```json
{
  "dnaSequence": {
    "sequence": "ATCGATCG",
    "count": 1
  }
}
```

#### 配体（ligand）

**方式 1：SMILES 字符串**
```json
{
  "ligand": {
    "ligand": "CC1=CC(=CC=C1)C(=O)NC2CCC3=CC(=C(C=C23)OC)OC",
    "count": 1
  }
}
```

**方式 2：SDF 文件路径**
```json
{
  "ligand": {
    "ligand": "FILE_studio-179/priority_1/warfarin.sdf",
    "count": 1
  }
}
```

注意：文件路径必须以 `FILE_` 前缀标记，其后为相对于仓库根目录的路径。

### 12.3 实际示例文件解析

#### unconditional_config.json（无条件生成）

```json
[
  {"name": "length_70",  "sequences": [{"proteinChain": {"sequence": "----...(70个-)", "count": 1}}]},
  {"name": "length_100", "sequences": [{"proteinChain": {"sequence": "----...(100个-)", "count": 1}}]},
  {"name": "length_200", "sequences": [{"proteinChain": {"sequence": "----...(200个-)", "count": 1}}]},
  {"name": "length_300", "sequences": [{"proteinChain": {"sequence": "----...(300个-)", "count": 1}}]}
]
```

无任何约束，模型自由生成不同长度的蛋白质。

#### PLP.json（PLP 配体条件生成）

PLP = 磷酸吡哆醛（Pyridoxal Phosphate），维生素 B6 活性形式，许多酶的辅因子。

```json
[
  {
    "name": "PLP_len150",
    "sequences": [
      {
        "proteinChain": {
          "sequence": "----...(150个-)",
          "count": 1
        }
      },
      {
        "ligand": {
          "ligand": "FILE_studio-179/priority_1/PLP.sdf",
          "count": 1
        }
      }
    ]
  },
  {"name": "PLP_len200", ...},
  {"name": "PLP_len250", ...}
]
```

模型会设计出能结合 PLP 的蛋白质。

#### 6YMC_rna.json（RNA 条件生成）

```json
[
  {
    "name": "6YMC_rna_len50",
    "sequences": [
      {
        "proteinChain": {
          "sequence": "----...(50-80个-)",
          "count": 1
        }
      },
      {
        "rnaSequence": {
          "sequence": "GCGCUCAUAGCUCAGUGGUA...",
          "count": 1
        }
      }
    ]
  }
]
```

RNA 序列固定，设计与该 RNA 相互作用的蛋白质。

### 12.4 共价键（covalent_bonds）

对于包含共价辅基的酶（如 PLP 依赖型酶），可以指定蛋白质与配体之间的共价键：

```json
{
  "covalent_bonds": [
    {
      "entity1": 0,          // 第一个实体的索引（蛋白质）
      "res_idx1": 42,        // 蛋白质中残基索引
      "atom_name1": "NZ",    // 赖氨酸的 NZ 原子
      "entity2": 1,          // 第二个实体的索引（配体）
      "res_idx2": 0,
      "atom_name2": "C4A"    // PLP 的 C4A 原子
    }
  ]
}
```

---

## 13. 输出格式

### 13.1 结构文件

**模式 1：无条件/单链蛋白质（output_format=unconditional_monomer_protein）**
```
{dump_dir}/pdbs/{sample_name}_sample_{seed}.pdb
```
PDB 格式，只包含蛋白质骨架（N/CA/C/O，bb_only=true）

**模式 2：有配体/RNA/DNA（output_format=null，biomol 模式）**
```
{dump_dir}/pdbs/predictions/{sample_name}_sample_{seed}.cif
```
CIF 格式，包含所有分子类型。

### 13.2 序列文件

```
{dump_dir}/sequences/{sample_name}_sample_{seed}.txt
```

内容格式：
```
>cogen_seq 0
CHAIN A type=protein mode=design seq=MKVFEGLPRHA...
CHAIN B type=rna mode=fixed seq=AUCGAUCGAUC...
ligand_smiles CC(=O)Oc1ccccc1C(=O)O
```

- `mode=design`：该链是被设计的
- `mode=fixed`：该链是固定的输入

### 13.3 置信度文件

```
{dump_dir}/pdbs/predictions/{sample_name}_sample_{seed}_confidence_scores.json
```

包含 pLDDT（per-residue Local Distance Difference Test）等置信度分数。

### 13.4 配体 SMILES 文件

```
{dump_dir}/pdbs/{sample_name}_sample_{seed}_ligands.txt
```

当输入包含 SMILES 配体时输出。

### 13.5 错误文件

```
{dump_dir}/ERR/{sample_name}.txt
```

如果某个样本处理时发生异常，错误信息写入此文件。

---

## 14. 脚本说明

### 14.1 run_inference.sh（主推理脚本）

```bash
# 设置参数
export DISCO_EXPERIMENT=designable  # 或 diverse
export DISCO_EFFORT=max             # 或 fast
export DISCO_SEEDS="[101,102]"      # 随机种子
export DISCO_CHECKPOINT="/path/to/DISCO.pt"  # 可选

# 运行推理
INPUT_JSON=input_jsons/warfarin.json \
DUMP_DIR=./output \
bash scripts/run_inference.sh
```

内部调用：
```bash
python runner/inference.py \
  experiment=${EXPERIMENT} \
  effort=${EFFORT} \
  input_json_path=${INPUT_JSON} \
  seeds=${SEEDS} \
  dump_dir=${DUMP_DIR} \
  [load_checkpoint_path=${CHECKPOINT}]
```

### 14.2 runner/inference.py（推理入口，@hydra.main）

```python
@hydra.main(config_path="../configs", config_name="inference")
def main(cfg):
    # 1. 根据配置初始化模型
    runner = InferenceRunner(cfg)  # 加载模型权重
    
    # 2. 准备数据
    dataloader = get_inference_dataloader(
        input_json_path=cfg.input_json_path,
        seeds=cfg.seeds,
        ...
    )
    
    # 3. 推理循环
    for batch in dataloader:
        data, atom_array, sample2feat, error_message = batch[0]
        
        if error_message:
            log.error(error_message)
            continue
        
        # 4. 模型推理
        data["input_feature_dict"]["atom_array"] = atom_array
        prediction = runner.predict(data, sample2feat)
        
        # 5. 保存结构
        structure_path = runner.dumper.dump(
            prediction, atom_array, sample2feat, seed, name
        )
        
        # 6. 提取序列并保存
        decoder_pred = prediction["decoder_pred"]
        sequences = extract_sequences(decoder_pred)
        save_sequences(sequences, output_path)
```

### 14.3 setup_env.sh（环境安装）

```bash
# 使用 uv 安装所有依赖（workspace 模式）
uv sync

# 可选：跳过 deepspeed（如果 CUDA 版本不兼容）
DISCO_SKIP_DEEPSPEED=1 bash scripts/setup_env.sh

# 可选：指定 PyTorch 后端
TORCH_BACKEND=cu121 bash scripts/setup_env.sh
```

### 14.4 doctor_env.sh（环境诊断）

自动检查：
- ✓ 虚拟环境是否存在
- ✓ 关键包是否可导入
- ✓ CUDA 是否可用
- ✓ `nvidia-smi` 信息
- ✓ 资产文件是否存在（DISCO.pt, components.v20240608.cif）
- ✓ 训练入口点是否存在（注：训练代码未公开）
- ✓ input_jsons/ 中的 JSON 文件数量
- ✓ studio-179/ 中的 SDF 文件数量

---

## 15. 关键常量与查找表

### 15.1 LIGAND_EXCLUSION（约 120 个 CCD 代码）

训练时排除的 CCD 代码，主要是结晶辅剂（不代表真实生物功能）：
```python
CRYSTALLIZATION_AIDS = {"SO4", "GOL", "EDO", "PO4", "ACT", "PEG", "DMS", ...}
```

### 15.2 PBV2_COMMON_NATURAL_LIGANDS（~50 个常见生物配体）

用于 Studio-179 基准测试的天然配体，例如：
```python
{
    "FAD",   # 黄素腺嘌呤二核苷酸（氧化还原酶辅因子）
    "NAD",   # 烟酰胺腺嘌呤二核苷酸
    "ATP",   # 腺苷三磷酸
    "PLP",   # 磷酸吡哆醛（维生素 B6）
    "HEM",   # 血红素
    "COA",   # 辅酶 A
    ...
}
```

### 15.3 BACKBONE_ATOMS

```python
BACKBONE_ATOMS = ["CA", "C", "N", "O"]
```

蛋白质骨架的 4 个原子。DISCO 在 `bb_only=true` 模式下只预测这 4 个原子的坐标。

### 15.4 LigandMPNN 残基索引映射

由于 DISCO 和 LigandMPNN 使用不同的残基索引方案，`constants.py` 提供了双向映射：

```python
OUR_RES_IDX_TO_LMPNN_RES_IDX = {int: int}  # DISCO 索引 → LigandMPNN 索引
LMPNN_RES_IDX_TO_OUR_RES_IDX = {int: int}  # LigandMPNN 索引 → DISCO 索引
```

### 15.5 元素编码（ELEMS）

128 种元素的 one-hot 编码，从原子序数映射：

```python
ELEMS = {
    "H": 0, "He": 1, "Li": 2, ..., "C": 5, "N": 6, "O": 7, ...
}
# 共 128 种元素，对应 ref_element 特征的 128 维 one-hot
```

---

## 16. 已知局限性

DISCO README 和代码中明确指出的限制：

### 16.1 单链蛋白质限制

> "DPLM-650M was trained on single protein chains. Multi-chain protein co-design is not supported."

DPLM-650M 在单链蛋白质上预训练，因此：
- ✅ 单链蛋白质设计（最常见场景）
- ❌ 多链蛋白质联合设计（如抗体-抗原复合物设计）
- ✅ 蛋白质 + 配体（配体不算蛋白质链）
- ✅ 蛋白质 + RNA/DNA（非蛋白质分子）

### 16.2 骨架专用模式

`bb_only=true` 在所有推理配置中都是默认开启的。这意味着：
- 模型只生成蛋白质骨架坐标（N/CA/C/O）
- **不预测侧链坐标**
- 侧链需要后处理工具（如 Rosetta Packer 或 ProteinMPNN）填充

### 16.3 训练代码未公开

`scripts/train_disco.sh` 只是一个占位符，打印"训练代码即将发布"的信息。目前仅支持推理。

### 16.4 设计质量评估

论文使用 **Chai-1** 重折叠验证：
- 将设计的序列重新折叠（无结构模板）
- 用 RMSD 评估：骨架 RMSD < 2Å **且** 配体质心 RMSD < 2Å 为"通过"
- 这被称为"co-designability"（联合可设计性）指标

---

## 17. 论文核心贡献总结

### 17.1 方法创新

| 创新点 | 具体内容 |
|---|---|
| **联合序列-结构设计** | 在同一扩散过程中同时生成序列（MDLM）和结构（EDM），而非分步进行 |
| **序列解码器内嵌** | 在 JointDiffusionModule 中增加 atom_attention_decoder_seq，与结构解码器共享中间表示 |
| **冻结 PLM 先验** | 使用 DPLM-650M 作为序列先验，为序列预测提供进化信息 |
| **Noisy Guidance** | 类 CFG 引导，通过两次前向传播放大条件信息（配体/RNA/DNA）的影响 |
| **PathPlanning 解码** | 基于置信度分数的有序揭示策略，比 VanillaMDLM 产生更一致的序列 |

### 17.2 基准测试：Studio-179

DISCO 构建了包含 179 种配体的基准测试集（studio-179/），按优先级分类：
- `priority_1/`：最重要的天然生物配体（FAD、PLP、血红素等）
- `priority_2/`：次重要配体
- `priority_3/`：其他配体

评估指标：co-designability（Chai-1 验证通过率）

### 17.3 整体系统流程图

```
输入：
┌─────────────────────────────┐
│  配体 SMILES / SDF 文件      │
│  + 蛋白质长度（全 "-"）       │
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  数据处理流水线               │
│  JSON → AtomArray → Features│
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  DISCO 模型                  │
│  ┌───────────────────────┐  │
│  │ PLM（DPLM-650M，冻结） │  │
│  │ + PairformerStack(×8) │  │
│  └────────────┬──────────┘  │
│               │ s, z 表示    │
│  ┌────────────▼──────────┐  │
│  │ JointDiffusionModule  │  │
│  │ ┌───────────────────┐ │  │
│  │ │ AtomAttEncoder    │ │  │
│  │ │ DiffusionTransf.  │ │  │
│  │ │ (24层, 16头)      │ │  │
│  │ │ AtomAttDecoder_3D │ │  │ → 结构坐标更新
│  │ │ AtomAttDecoder_Seq│ │  │ → 氨基酸 logit
│  │ └───────────────────┘ │  │
│  └───────────────────────┘  │
└─────────────┬───────────────┘
              │ 迭代 N_step 步
              ▼
┌─────────────────────────────┐
│  输出                        │
│  • PDB/CIF 结构文件          │
│  • 氨基酸序列 .txt 文件       │
│  • 置信度分数 .json 文件       │
└─────────────────────────────┘
```

---

## 附录 A：重要文件快速索引

| 想了解 | 查看文件 |
|---|---|
| 模型总体结构 | `disco/model/disco.py` |
| PairformerBlock | `disco/model/modules/pairformer.py` |
| 扩散模块 | `disco/model/modules/diffusion.py` |
| 注意力机制基础 | `disco/model/modules/primitives.py` |
| Transformer 变体 | `disco/model/modules/transformer.py` |
| 语言模型集成 | `disco/model/modules/plm.py` |
| LigandMPNN 包装 | `disco/model/modules/lmpnn.py` |
| 推理主循环 | `disco/model/cogen_inference/cogen_generator.py` |
| 单步去噪 | `disco/model/cogen_inference/cogen_inference_loop_body_impl.py` |
| 序列采样策略 | `disco/model/cogen_inference/sequence_sampling_strategy.py` |
| 噪声调度 | `disco/model/utils.py` |
| 输入特征化 | `disco/data/featurizer.py` |
| JSON 解析 | `disco/data/json_parser.py` |
| 完整特征流水线 | `disco/data/json_to_feature.py` |
| 推理数据集 | `disco/data/infer_data_pipeline.py` |
| 掩码处理 | `disco/data/task_manager.py` |
| 所有常量 | `disco/data/constants.py` |
| CCD 访问 | `disco/data/ccd.py` |
| 推理主脚本 | `runner/inference.py` |
| 输出保存 | `runner/dumper.py` |
| 模型配置 | `configs/model/default.yaml` |
| 推理配置 | `configs/inference/default.yaml` |
| 实验配置 | `configs/experiment/designable.yaml` |

---

## 附录 B：快速上手命令

```bash
# 无条件蛋白质生成（最简单）
INPUT_JSON=input_jsons/unconditional_config.json \
DUMP_DIR=./output_unconditional \
bash scripts/run_inference.sh

# 配体条件生成（华法林）
INPUT_JSON=input_jsons/warfarin.json \
DUMP_DIR=./output_warfarin \
bash scripts/run_inference.sh

# RNA 条件生成
INPUT_JSON=input_jsons/6YMC_rna.json \
DUMP_DIR=./output_rna \
bash scripts/run_inference.sh

# 快速模式（减少步数）
DISCO_EFFORT=fast \
INPUT_JSON=input_jsons/PLP.json \
DUMP_DIR=./output_fast \
bash scripts/run_inference.sh

# 多样性模式（更多随机性）
DISCO_EXPERIMENT=diverse \
INPUT_JSON=input_jsons/heme_b.json \
DUMP_DIR=./output_diverse \
bash scripts/run_inference.sh

# 使用多个种子生成多个设计
DISCO_SEEDS="[0,1,2,3,4,5,6,7,8,9]" \
INPUT_JSON=input_jsons/NDI.json \
DUMP_DIR=./output_multi_seed \
bash scripts/run_inference.sh
```

---

*本文档基于 DISCO 仓库代码全面分析编写，覆盖了从数据处理到模型架构到推理流程的所有关键细节。如需了解某个具体组件的更多信息，请参考附录 A 中的文件索引。*
