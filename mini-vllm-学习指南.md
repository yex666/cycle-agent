# 从 mini-vLLM 到推理引擎专家：完整学习指南

> 本文档基于 [mini-vLLM](../mini-vllm/) 项目，一个用纯 NumPy 实现的可运行 vLLM 简化版。
> 读完后你将有能力：理解 vLLM 全部核心机制、阅读真实 vLLM 源码、自己造一个推理框架、部署 DeepSeek 等 MoE 模型。

---

## 目录

- [第 0 章 前置知识：你必须先搞懂的事](#第-0-章-前置知识你必须先搞懂的事)
- [第 1 章 为什么需要专门的推理引擎](#第-1-章-为什么需要专门的推理引擎)
- [第 2 章 PagedAttention：vLLM 的灵魂](#第-2-章-pagedattentionvllm-的灵魂)
- [第 3 章 连续批处理：高吞吐量的秘密](#第-3-章-连续批处理高吞吐量的秘密)
- [第 4 章 引擎的完整运转流程](#第-4-章-引擎的完整运转流程)
- [第 5 章 进阶机制：前缀缓存、抢占、分块 Prefill](#第-5-章-进阶机制前缀缓存抢占分块-prefill)
- [第 6 章 从 TinyGPT 到真实模型：架构差异](#第-6-章-从-tinygpt-到真实模型架构差异)
- [第 7 章 量化、投机解码与服务化](#第-7-章-量化投机解码与服务化)
- [第 8 章 动手实验：改代码加深理解](#第-8-章-动手实验改代码加深理解)
- [第 9 章 部署 DeepSeek / MoE 模型实战](#第-9-章-部署-deepseek--moe-模型实战)
- [第 10 章 造你自己的推理框架](#第-10-章-造你自己的推理框架)
- [附录 A：mini-vLLM 全模块速查表](#附录-amini-vllm-全模块速查表)
- [附录 B：术语表](#附录-b术语表)

---

## 第 0 章 前置知识：你必须先搞懂的事

### 0.1 Transformer 推理的两个阶段

大语言模型（LLM）推理分为两个截然不同的阶段：

```
用户输入: "vLLM is a fast library"

【Prefill 阶段】（处理整个 prompt）
  输入: ["vLLM", "is", "a", "fast", "library"]  ← 所有 token 一次性送入
  输出: 第 1 个生成 token（比如 "for"）
  副作用: 所有 5 个 token 的 Key/Value 被缓存

【Decode 阶段】（每次只处理 1 个 token）
  第 1 步: 输入 ["for"]         → 输出 "LLM"     → 缓存 "for" 的 KV
  第 2 步: 输入 ["LLM"]         → 输出 "inference" → 缓存 "LLM" 的 KV
  第 3 步: 输入 ["inference"]   → 输出 "and"      → 缓存 "inference" 的 KV
  ...
  直到输出 EOS（结束符）或达到最大长度
```

**关键区别**：
- Prefill 是**计算密集型**的——一次处理很多 token，GPU 算力被打满
- Decode 是**访存密集型**的——每次只处理 1 个 token，但要读取所有历史 KV，GPU 算力大量闲置

这就是为什么 decode 阶段是推理速度的瓶颈——不是算不快，而是**数据搬不动**。

### 0.2 KV Cache 是什么，为什么它吃显存

在注意力机制中，每个 token 位置会产生一个 Key 向量和一个 Value 向量。生成新 token 时，新 token 的 Query 要和**所有历史 token 的 Key** 做点积，然后用得到的注意力权重去加权所有历史 Value。

如果不缓存历史 KV，每生成一个 token 都要重新计算整个 prompt——复杂度是 O(n²)。所以必须缓存。

**KV Cache 大小估算**：

```
单个 token 的 KV 大小 = 2 (K和V) × n_layers × n_heads × head_dim × dtype_size

以 LLaMA-7B 为例:
  n_layers=32, n_heads=32, head_dim=128, dtype=float16(2字节)
  单 token KV = 2 × 32 × 32 × 128 × 2 = 524,288 字节 = 0.5 MB

  2048 长度的序列 KV = 0.5 MB × 2048 = 1 GB
  100 个并发请求 = 100 GB  ← 远超单张 GPU 的显存！
```

这就是推理引擎要解决的核心问题：**如何在有限的显存里，高效地为大量并发请求管理 KV Cache**。

### 0.3 计算机里有哪几种存储（先分清层次）

在看分页机制前，先把"计算机里到底有哪些存东西的地方"搞清楚。核心只有 4 类硬件 + 1 个 GPU 专属：

```
        ┌─────────────┐
        │    寄存器    │  CPU内部，KB级，最快最贵，断电丢
        ├─────────────┤
        │ L1/L2/L3缓存 │  CPU旁边，MB级，断电丢
        ├─────────────┤
        │  内存(RAM)   │  GB级，断电丢，程序运行的地方
        │   (主存)     │  ← 操作系统分页管的是这层
        ├─────────────┤
        │  磁盘(SSD)   │  TB级，断电不丢，文件长期存放处
        └─────────────┘

另外还有 GPU 专属的:
        ┌─────────────┐
        │  显存(VRAM)  │  GPU自己的内存，放模型权重和 KV Cache
        └─────────────┘
```

| 存储 | 容量 | 断电 | 干什么用 |
|------|------|------|---------|
| 寄存器 | KB 级 | 丢失 | CPU 运算时的操作数 |
| 缓存 Cache | MB 级 | 丢失 | CPU 常用数据的快速副本 |
| 内存 RAM | GB 级 | 丢失 | 程序运行时所在的"工作台" |
| 磁盘 | TB 级 | 保留 | 文件、程序的"长期存档" |
| 显存 VRAM | GB 级 | 丢失 | GPU 专属内存，LLM 推理主战场 |

三个关键事实：

1. **速度差巨大**：寄存器比磁盘快 10 万倍以上。离 CPU 越近越快，但越贵、容量越小。
2. **CPU 只能直接访问内存**（和缓存），不能直接访问磁盘。磁盘上的东西必须先搬到内存才能用。
3. **同一个程序有两份**：磁盘上一份（存档），内存里一份（运行拷贝）。

一个程序的完整旅程：

```
磁盘: xxx.py 文件躺在这（断电不丢）
  ↓ 双击启动
内存: 操作系统把文件读到 RAM
  ↓
CPU: 从内存取指令，常用数据过缓存，操作数进寄存器
  ↓ 程序关闭
内存: 这份拷贝销毁，磁盘原文件还在
```

**LLM 推理涉及三种存储，各司其职**：

```
磁盘   →  模型权重文件（几GB~几百GB），启动时一次性读入显存
显存   →  模型权重（常驻）+ KV Cache（动态增减） ← vLLM 管的就是这里
内存   →  SWAP 抢占时 KV Cache 的暂存地（显存不够时的兜底）
```

### 0.4 操作系统的分页机制（vLLM 的灵感来源）

#### 先纠正一个直觉：物理连续 ≠ 数据必须连续

"连续"有两层含义，必须分开：

1. **存储介质连续**：物理内存的格子是 0,1,2,3... 排下去的。这是对的，没法否认。
2. **程序的数据连续**：程序 D 的 10 格数据，是不是必须放在物理内存的 10 个相邻格子里？

分页机制回答的是第 2 层：**不需要**。

#### 朴素连续分配的问题

早期系统要求程序占用的内存必须是一段连续空间：

```
内存（8格）:
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ A0 │ A1 │ A2 │ A3 │ B0 │ B1 │    │    │
└────┴────┴────┴────┴────┴────┴────┴────┘

A 用 0~3，B 用 4~5。程序 C 需要 3 格 → 只剩 6~7 两格，装不下！
这就是"外部碎片"：空间总量够，但没有足够大的连续段。
```

#### 分页的核心思想（一句话）

> **把程序和内存都切成等大的小块，程序的一块可以放在内存的任何空闲位置，用一张"页表"记住放哪了。**

第一步：把物理内存切成等大的"页框"。第二步：把程序也切成同样大小的"页"。第三步：每页放进任意空闲页框，记录映射。

```
程序D的虚拟视角（连续）:          物理内存（实际散布）:
┌───────────┐
│ D页0 (4格) │────→ 放进页框 3   (物理地址 12~15)
├───────────┤
│ D页1 (4格) │────→ 放进页框 1   (物理地址 4~7)
├───────────┤
│ D页2 (2格) │────→ 放进页框 5   (物理地址 20~21)
└───────────┘

物理内存里，D 的格子被别的程序隔开，并不连续！

页表:
  页号0 → 页框3
  页号1 → 页框1
  页号2 → 页框5
```

程序运行时怎么"找"数据？以"访问我的第 5 格"为例：

```
逻辑地址 5 属于逻辑页 1（每页4格，5 = 4 + 1）
页表说: 逻辑页1 在 物理页框1
于是去物理地址: 页框1起点(4) + 页内偏移(1) = 物理地址 5

这一步由硬件(MMU)自动完成，程序毫无感知。
程序以为自己的数据是连续的 0~9，"连续"是程序视角的幻象，
"页表映射"是让它成真的机制。
```

#### 分页的三个好处

1. **不需要连续分配**——进程不需要一整块连续内存，外部碎片消失
2. **按需分配**——用到哪页才分配哪页
3. **共享内存**——两个进程的页表可指向同一个物理页框（如共享库）

#### 换页（swap）：内存不够时的招

程序太大放不下时，操作系统把不用的页搬到磁盘，要用再搬回来：

```
程序D的页2 暂时不用 → 搬到磁盘 → 页框5 空出来给别人用
程序D要用页2 → 从磁盘搬回，放进任意空闲页框，更新页表
```

#### 对照到 vLLM：逐词翻译

| 操作系统 | vLLM | 说明 |
|---------|------|------|
| 进程（程序） | 序列（Sequence） | 一个正在推理的请求 |
| 虚拟地址空间 | 逻辑 token 位置 | 第 0、1、2... 个 token |
| 页（Page） | 块（Block） | 定长单元，vLLM 默认 16 个 token |
| 物理页框 | 物理 KV 块 | 显存里存 K/V 的小块 |
| 页表（Page Table） | 块表（Block Table） | 逻辑位置 → 物理块的映射 |
| 共享物理页 | 共享 KV 块 | 前缀缓存时两个请求共享同一批块 |
| 写时复制 COW | COW | 共享块要写时先复制 |
| 换页到磁盘 | SWAP 抢占模式 | KV 从 GPU 搬到 CPU 内存 |

**注意**：OS 管的是"内存(RAM)"这层，vLLM 管的是"显存"这层——**机制相同，管理的存储层不同**。

回到 vLLM 的一个常见疑惑：`KVStore` 本身明明是 `np.zeros` 一次性分配的一块连续显存，为什么说数据可以不连续？因为"**显存介质连续**"和"**序列数据连续**"是两回事——物理块 3 和物理块 5 之间隔着的物理块 4 可能是别的序列的。block_table 管的就是后者，这正是第 2 章要讲的。

### 0.5 你需要的基础知识清单

| 知识点 | 需要掌握的程度 |
|--------|---------------|
| Python | 能读写 Python 代码 |
| NumPy | 理解 ndarray 基本操作（矩阵乘法、reshape） |
| Transformer 架构 | 知道 Self-Attention、MLP、LayerNorm |
| 矩阵运算 | 理解 einsum 或矩阵乘法的含义 |
| 操作系统分页 | 概念上理解页表即可 |

不需要你懂 CUDA、不需要你懂 PyTorch 的底层、不需要你懂分布式系统。

---

## 第 1 章 为什么需要专门的推理引擎

### 1.1 朴素推理的三大问题

假设你直接用 HuggingFace Transformers 做推理：

```python
# 朴素方式
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("...")
output = model.generate("Hello, I am")  # 每次只处理一个请求
```

这会面临三个致命问题：

#### 问题 1：显存浪费（内部碎片）

先澄清两个数字：上面代码里的 `"Hello, I am"` 只是**调用示例**，与后面的数字无关。下面假设一个**真实请求**：这个请求的 prompt 有 **500 个 token**。

朴素实现（HuggingFace Transformers）的 KV Cache 是**一次性按"模型最大上下文长度"分配的**——很多模型默认最大长度是 2048（GPT-2 的 `max_position_embeddings` 就是 2048），所以不管你的 prompt 多短，它先按 2048 长度把整块显存占住：

```
请求A: prompt 实际 500 tokens，但预分配 2048 长度的 KV 空间
  → 2048 - 500 = 1548 个 token 的空间白白浪费

实际数字（GPT-2，12层/12头/64维/float32）:
  单 token KV = 2 × 12 × 12 × 64 × 4 = 72 KB
  预分配 2048: 72 KB × 2048 ≈ 144 MB
  实际用 500:  72 KB × 500  ≈ 35 MB
  浪费 ≈ 109 MB（约 76%）

换成 LLaMA-7B（float16）更夸张:
  单 token KV = 2 × 32 × 32 × 128 × 2 = 512 KB / token
  一个请求就浪费约 750 MB！
```

vLLM 的做法：**按 block 动态分配**，用多少分多少，不提前占位：

```
vLLM（同样 500 token 的请求）:
  需要块数 = ceil(500/16) = 32 个物理块
  最后一块只用了 4/16 → 最多浪费 15 个 token 的空间（≤3%）
  解码过程中用到哪块才分配哪块
  生成结束，块立即回收给别的请求用
```

#### 问题 2：外部碎片

```
显存: [已用A][空闲200][已用B][空闲300][已用C][空闲500]
请求D 需要 400 连续空间 → 总空闲 1000 但放不下（碎片化）
```

#### 问题 3：无法批处理

```
请求1 正在 decode（每次只算 1 个 token，GPU 大量闲置）
请求2 刚到达（必须等请求1 完成才能开始）
  → GPU 利用率极低，用户等待时间长
```

### 1.2 vLLM 的解决方案

| 问题 | vLLM 的解法 | 对应代码 |
|------|------------|---------|
| 内部碎片 | KV Cache 按 block 分配，用多少分多少 | `BlockAllocator` |
| 外部碎片 | 不需要连续内存，block 可散布 | `block_table` |
| 无法批处理 | 连续批处理，每步混合 prefill + decode | `Scheduler` |
| 重复前缀计算 | 前缀缓存，共享 KV 块 | `PrefixCache` |
| 显存不足 | 抢占低优先级请求 | `_preempt_one()` |

### 1.3 mini-vLLM 的定位

mini-vLLM 是 vLLM 的**教学版**，它：

| 特性 | 真实 vLLM | mini-vLLM |
|------|-----------|-----------|
| 计算后端 | CUDA kernel + PyTorch | 纯 NumPy |
| 模型 | LLaMA/GPT/Qwen/DeepSeek... | TinyGPT（2-3层玩具模型） |
| 分布式 | Tensor/Pipeline Parallel | 无 |
| KV Cache 布局 | 完全一致 | 完全一致 ✅ |
| 调度逻辑 | 完全一致 | 完全一致 ✅ |
| 服务形态 | OpenAI 兼容 | OpenAI 兼容 ✅ |

**它保留了 vLLM 的全部"大脑"，只替换了"肌肉"（CUDA kernel → NumPy）。**

---

## 第 2 章 PagedAttention：vLLM 的灵魂

> 对应代码：`minivllm/kv_cache.py` + `minivllm/attention.py`

这是整个项目最核心的一章。如果你只读一个章节，就读这个。

### 2.1 物理存储：KVStore

KV Cache 的物理存储是一个 4 维数组：

```python
# kv_cache.py
class KVStore:
    # 每层的 KV cache 形状: (num_blocks, block_size, num_heads, head_dim)
    shape = (num_blocks, block_size, num_heads, head_dim)
    self.k_cache = [np.zeros(shape) for _ in range(num_layers)]
    self.v_cache = [np.zeros(shape) for _ in range(num_layers)]
```

用图来理解：

```
物理块编号:    0          1          2          3        ...
            ┌──────────┬──────────┬──────────┬──────────┐
  block 0   │ token0   │          │          │ token0   │
  block 1   │ token1   │ token0   │          │ token1   │
  block 2   │ token2   │ token1   │          │          │
  ...       │ ...      │ ...      │  (空闲)  │ ...      │
  block 15  │ token15  │ token15  │          │ token15  │
            └──────────┴──────────┴──────────┴──────────┘
              ↑ 物理块0    ↑ 物理块1   ↑ 空闲    ↑ 物理块3
              被 seq A 使用  被 seq B 使用            被 seq C 使用
```

每个物理块能存 `block_size` 个 token 的 KV。块和块之间不需要连续。

### 2.2 块表：逻辑→物理映射

每个序列持有一个 **block table**（块表），就像操作系统的页表：

```python
# 序列 A 的 block_table: [0, 3]
# 意思: A 的第 0~15 token 的 KV 在物理块 0
#        A 的第 16~31 token 的 KV 在物理块 3

# 序列 B 的 block_table: [1]
# 意思: B 的第 0~15 token 的 KV 在物理块 1

# 序列 C 的 block_table: [0, 3]  ← 和 A 共享！
# 意思: C 的前 16 token 和 A 共享物理块 0（前缀缓存）
#        C 的第 16~31 token 和 A 共享物理块 3
```

这就是为什么不需要连续内存——逻辑上连续的 token，物理上可以散布在任何空闲块里。

### 2.3 块分配器与引用计数

```python
# kv_cache.py
class BlockAllocator:
    free: List[int]           # 空闲块列表
    ref_count: List[int]     # 每个块的引用计数

    def allocate(self) -> int:
        """分配一个新块，引用计数设为 1"""
        block_id = self.free.pop()
        self.ref_count[block_id] = 1
        return block_id

    def free_block(self, block_id):
        """释放引用，引用计数归零才真正回收"""
        self.ref_count[block_id] -= 1
        if self.ref_count[block_id] == 0:
            self.free.append(block_id)

    def touch(self, block_id):
        """增加引用——用于前缀缓存共享"""
        self.ref_count[block_id] += 1
```

**为什么需要引用计数？**

当一个物理块被多个序列共享时（比如前缀缓存命中），不能任何一个序列结束就释放它。只有**所有使用者都释放后**才能回收：

```
物理块 0 被以下对象持有:
  - 序列 A (引用计数 +1)
  - 序列 C (引用计数 +1)
  - 前缀缓存 (引用计数 +1)
  总引用计数 = 3

序列 A 结束 → free_block(0) → ref_count 变成 2 → 块不回收
序列 C 结束 → free_block(0) → ref_count 变成 1 → 块不回收
前缀缓存淘汰 → free_block(0) → ref_count 变成 0 → 块回收 ✅
```

### 2.4 分页注意力：从散布的块中收集 KV

标准注意力计算：

```
Attention(Q, K, V) = softmax(Q × K^T / √d) × V
```

在标准实现中，K 和 V 是连续的矩阵。但在 PagedAttention 中，K/V 散布在不同物理块里，需要先**收集**：

```python
# attention.py
def gather_kv(k_cache, v_cache, block_table,
              logical_start, logical_end, ...):
    """从散布的物理块中收集逻辑位置 [start, end) 的 KV"""

    block_start = logical_start // block_size
    block_end = (logical_end + block_size - 1) // block_size

    ks, vs = [], []
    for bi in range(block_start, block_end):
        phys = block_table[bi]          # 查页表: 逻辑块 → 物理块
        lo = max(0, logical_start - bi * block_size)
        hi = min(block_size, logical_end - bi * block_size)
        ks.append(k_cache[phys, lo:hi])  # 从物理块切片
        vs.append(v_cache[phys, lo:hi])

    return np.concatenate(ks), np.concatenate(vs)  # 拼成连续数组
```

图解（三个视图 + 逐步执行）：

【视图一：物理块数组】—— KVStore 是连续分配的一块显存，但每个块装谁的 KV 是"乱"的

```
物理块:  [   块0   ][   块1   ][   块2   ][   块3   ][   块4   ][   块5   ][   块6   ][   块7   ]
          A的token   B的token   空闲       A的token   C的token   空闲       B的token续  空闲
          0~15                  ↑          16~31                          ↑
          └─────────────────────┘───────────────────────┘
             A 的物理块是 0 和 3，中间隔着块1(B的)、块2(空闲) —— 物理上不连续！
```

【视图二：A 的逻辑视角】—— 序列 A 以为自己是连续的

```
A 的逻辑位置:  0  1  2 ... 15 | 16  17 ... 31
              └── 逻辑块0 ──┘ └── 逻辑块1 ──┘
              （逻辑上就是连续的两块，它自己毫无感知）
```

【视图三：block_table 映射表】—— 把"逻辑连续"翻译成"物理散落"的关键

```
  逻辑块号    →    物理块号
    0        →      0
    1        →      3

问题: 逻辑位置 5 的 KV 去哪找？
  → 5 // 16 = 0 (属于逻辑块0) → 查表 → 物理块 0 → 块内偏移 5
问题: 逻辑位置 16 的 KV 去哪找？
  → 16 // 16 = 1 (属于逻辑块1) → 查表 → 物理块 3 → 块内偏移 0

没有 block_table，你根本不知道去哪找！
```

【执行 gather_kv：收集逻辑位置 [5, 20)】

要收集 A 的逻辑位置 5~19（共 15 个 token），它横跨逻辑块 0 和逻辑块 1，查两次表：

```
逻辑块 0 覆盖位置 0~15 → 查表: block_table[0] = 物理块 0
  lo = max(0, 5  - 0×16) = 5
  hi = min(16, 20 - 0×16) = 16
  → 从物理块 0 切 k_cache[0, 5:16]   (11个token: 逻辑位置5~15)

逻辑块 1 覆盖位置 16~31 → 查表: block_table[1] = 物理块 3
  lo = max(0, 5  - 1×16) = 0
  hi = min(16, 20 - 1×16) = 4
  → 从物理块 3 切 k_cache[3, 0:4]    (4个token: 逻辑位置16~19)

拼接 → 15 个 token 的连续 KV（逻辑位置 5~19 全齐，顺序不乱）
```

**代码逐行执行跟踪**（代入具体数字：block_size=16，block_table=[0,3]，收集逻辑位置 [5,20)）：

```
前提: 序列A的 block_table = [0, 3]
      要收集 A 的逻辑位置 5~19（15个token），横跨逻辑块0和逻辑块1

block_start = 5 // 16        = 0    # 起始逻辑块
block_end = (20 + 15) // 16  = 2    # 结束逻辑块(不含) → 只处理 bi=0 和 bi=1

# ── 第1次循环: bi = 0 ──
phys = block_table[0]        = 0    # 逻辑块0 → 物理块0
lo = max(0, 5 - 0*16)        = 5    # 块内从偏移5开始切
hi = min(16, 20 - 0*16)      = 16   # 切到块末尾
ks.append(k_cache[0, 5:16])         # 取物理块0的槽位[5,16) = 逻辑位置5~15 ✓

# ── 第2次循环: bi = 1 ──
phys = block_table[1]        = 3    # 逻辑块1 → 物理块3   ← 注意！0 跳到 3
lo = max(0, 5 - 1*16)        = 0    # 逻辑位置5不在块1里，从块开头切
hi = min(16, 20 - 1*16)      = 4    # 切到偏移4
ks.append(k_cache[3, 0:4])          # 取物理块3的槽位[0,4) = 逻辑位置16~19 ✓

k = np.concatenate(ks)              # [块0的11个] + [块3的4个] = 连续的15个 ✓
```

**"散布 / 映射 / 收集"分别体现在代码的哪里**：

| 你在找的东西 | 在代码的哪里 | 具体表现 |
|-------------|-------------|---------|
| 散布 | 两次循环取的是 `k_cache[0,...]` 和 `k_cache[3,...]` | 索引从 0 跳到 3，中间 1、2 是别人的数据 |
| 映射 | `phys = block_table[bi]` | 一行查表：逻辑块号 → 物理块号 |
| 收集 | `ks.append(...)` + `np.concatenate(ks)` | 从两个不相邻的块各切一段，拼回连续数组 |

**lo/hi 公式在算什么**：把"全局逻辑区间"翻译成"块内偏移区间"。

```
lo/hi = 逻辑区间[5, 20) 与"该块覆盖的逻辑范围"的重叠部分，减去该块的起点

| bi | 该块覆盖的逻辑位置 | 与[5,20)重叠的部分 | 块内偏移[lo,hi) |
|----|------------------|-------------------|----------------|
| 0 | [0, 16)  | [5, 16)  | [5, 16)  |
| 1 | [16, 32) | [16, 20) | [0, 4)   |

- logical_start - bi*16 = 重叠区起点 - 块起点 = 块内偏移（第一块可能从中间开始 → 5）
- 中间块完整覆盖时 lo=0, hi=16（全取）
- 最后一块只用到开头 → hi 被 min 截断到 4
```

**核心理解**：`gather_kv` 做的事和操作系统查页表一模一样——每个逻辑块对应哪个物理块，全靠 block_table 这一张表记住；没有它，物理块 0 和物理块 3 之间的"断点"根本无从跨越。代码里没直接写"散布"和"映射"四个字，因为它们是**数据本身**：**散布 = 0→3 那个跳跃，映射 = `block_table[bi]` 那一行，收集 = 拼接**。

收集到连续的 K/V 后，注意力计算就和标准实现完全一样了：

```python
# attention.py: paged_attention_batch()
scores = np.einsum("hit,jht->hij", q, k_full) / np.sqrt(head_dim)

# 因果掩码: query i 只能看 key 0..cached_len+i
key_positions = np.arange(total_len)[None, :]
query_positions = cached_len + np.arange(num_new)[:, None]
mask = key_positions > query_positions
scores = np.where(mask, -np.inf, scores)

probs = softmax(scores)
out = np.einsum("hij,jht->hit", probs, v_full)
```

**在真实 vLLM 中**，这个 gather + attention 被融合进一个 CUDA kernel (`vllm/attention/ops/paged_attn.py`)，避免额外的内存拷贝。但逻辑完全一样。

### 2.5 Copy-on-Write：写时复制

当两个序列共享一个物理块时，如果其中一个要往这个块写入新 KV，不能直接写（会污染另一个序列）：

```python
# kv_cache.py: write_kv()
def write_kv(self, layer, seq, logical_pos, k, v):
    block_index = logical_pos // self.block_size
    offset = logical_pos % self.block_size
    phys = table[block_index]

    if self.allocator.ref_count[phys] > 1:
        # ⚠️ 块被共享！必须先复制一份
        new_phys = self.allocator.allocate()                    # 分配新块
        self.kv_store.k_cache[layer][new_phys] = \
            self.kv_store.k_cache[layer][phys]                   # 复制旧K
        self.kv_store.v_cache[layer][new_phys] = \
            self.kv_store.v_cache[layer][phys]                   # 复制旧V
        self.allocator.free_block(phys)                          # 释放旧块引用
        table[block_index] = new_phys                            # 更新页表
        phys = new_phys

    # 现在可以安全写入
    self.kv_store.k_cache[layer][phys, offset] = k
    self.kv_store.v_cache[layer][phys, offset] = v
```

图解 COW：

```
写入前:
  序列A.block_table = [5]  ──→  物理块5 [KV0, KV1, KV2, ...]
  序列C.block_table = [5]  ──→  ref_count[5] = 2

序列C 要写入新 KV 到位置 3:
  1. ref_count[5] > 1 → 触发 COW
  2. 分配物理块 8
  3. 复制: 物理块5 的内容 → 物理块8
  4. 序列C.block_table = [8]  ← 更新
  5. free_block(5) → ref_count[5] = 1（序列A仍持有）
  6. 写入: 物理块8[3] = 新KV

写入后:
  序列A.block_table = [5]  ──→  物理块5 [KV0, KV1, KV2, ...]     ref=1
  序列C.block_table = [8]  ──→  物理块8 [KV0, KV1, KV2, 新KV]    ref=1
```

**这和操作系统的 COW（fork 后写时复制）完全一样。**

### 2.6 小结：PagedAttention 的三大好处

1. **几乎消除内部碎片**——每个序列只分配它实际使用的块数（最多浪费最后一个块的少量空间）
2. **完全消除外部碎片**——物理块不需要连续，任何空闲块都能用
3. **支持共享**——通过引用计数 + COW，多个序列可以安全共享相同前缀的 KV

---

## 第 3 章 连续批处理：高吞吐量的秘密

> 对应代码：`minivllm/scheduler.py`

### 3.1 传统批处理 vs 连续批处理

**传统批处理（Static Batching）**：

```
时间→  t=0    t=1    t=2    t=3    t=4    t=5    t=6
请求A: [prefill][dec1][dec2][dec3][done]  等......等
请求B: [prefill][dec1][done]              等......等
请求C:                              [等待B完成后才能开始]

问题: 请求C 必须等整个 batch 都完成才能开始，GPU 大量空闲
```

**连续批处理（Continuous Batching）**：

```
时间→  t=0    t=1    t=2    t=3    t=4    t=5    t=6
请求A: [prefill][dec1][dec2][dec3][done]
请求B: [prefill][dec1][done]
请求C:                   [prefill][dec1][dec2][dec3]

请求C 在请求B 完成后立刻插入，不需要等请求A完成
每个 engine step 可以混合: 多个decode + 1个prefill
```

### 3.2 调度器的核心数据结构

```python
# scheduler.py
class Scheduler:
    waiting: deque[Sequence]   # 等待prefill的请求队列
    running: List[Sequence]    # 正在处理（prefill中或decode中）的请求
```

每个请求的生命周期：

```
用户请求到达
    ↓
  WAITING (在waiting队列中排队)
    ↓  调度器接纳
  RUNNING (prefill → decode)
    ↓  生成完成 或 被抢占
  FINISHED (返回结果) 或 回到WAITING (被抢占)
```

### 3.3 一个 engine step 的本质：token 名额池

在进入三阶段之前，先搞清楚 scheduler 每步到底在分配什么。

**一个 engine step = 一次 GPU 前向传播**。GPU 一次前向传播能处理的 token 数是有限的，所以引擎把工作切成 step，每个 step 就是一次前向传播。scheduler 的职责只有一件事：

> 决定这一步：哪些序列参与？每个序列处理几个 token？

它要满足两个硬约束：
- 一个 step 处理的**总 token 数 ≤ max_num_batched_tokens**（比如 512）
- 同时参与的**序列数 ≤ max_num_seqs**（比如 8）

**"名额池子"类比**：把 `max_num_batched_tokens` 想成每步的 token 名额池子：

```
每步的名额池子: [名额][名额][名额][名额][名额][名额][名额][名额][名额][名额]  ← 共10个

Phase A: 正在 decode 的序列"必须"各占 1 个名额
  → 3 个 decode 序列 → 占掉 3 个名额，剩 7 个

Phase B: 剩下 7 个名额给 prefill（新请求 / 没 prefill 完的）
  → prompt 长就切块，用多少名额切多少
```

**为什么 decode 优先？** 因为正在 decode 的序列已经在生成中了，每步必须推进 1 个 token，否则就卡死了。而 prefill 可以等、可以切成小块。所以 decode 先占名额，剩下的给 prefill。

### 3.4 一个 schedule() 调用的三阶段

每次引擎调用 `step()` 时，调度器先执行 `schedule()` 决定这一步做什么：

```python
# scheduler.py: schedule()
def schedule(self) -> ScheduledStep:
    step = ScheduledStep()
    num_batched_tokens = 0
    max_tokens = self.config.max_num_batched_tokens  # 比如 512

    # ─── Phase A: 给所有正在 decode 的序列各分配 1 个 token ───
    for seq in self.running:
        if seq.in_prefill:
            continue  # 还在prefill中，跳过
        # 检查显存够不够
        if not self.block_manager.can_allocate(seq, seq.cached_len + 1):
            self._preempt_one(step)  # 不够→抢占别人
        self.block_manager.ensure_blocks(seq, seq.cached_len + 1)
        step.decode_items.append(seq)
        num_batched_tokens += 1  # decode消耗1个token预算

    # ─── Phase B1: 继续还没 prefill 完的序列（分块 prefill）───
    remaining = max_tokens - num_batched_tokens
    for seq in self.running:
        if not seq.in_prefill:
            continue
        chunk = self._chunk_for(seq, need, remaining)  # 能切多少
        self.block_manager.ensure_blocks(seq, seq.cached_len + chunk)
        step.prefill_items.append((seq, chunk))
        remaining -= chunk
        num_batched_tokens += chunk

    # ─── Phase B2: 从等待队列接纳新请求 ───
    while self.waiting and remaining > 0:
        seq = self.waiting[0]

        # 前缀缓存命中？
        prefix_len, blocks = self.block_manager.match_prefix(seq.prompt_ids)
        if prefix_len > 0:
            self.block_manager.attach_shared_blocks(seq, blocks)
            seq.cached_len = prefix_len  # 跳过已缓存的token！

        chunk = self._chunk_for(seq, prompt_remaining, remaining)
        self.waiting.popleft()
        self._activate(seq)
        step.prefill_items.append((seq, chunk))

    return step
```

**一次调度的结果**（ScheduledStep）是一个混合批次：

```
ScheduledStep:
  prefill_items: [(seq_X, 128 tokens), (seq_Y, 64 tokens)]  ← 两个正在prefill
  decode_items:  [seq_A, seq_B, seq_C]                       ← 三个正在decode
  preempted:     [seq_Z]                                     ← 一个被抢占
```

这个混合批次会在一次前向传播中一起执行——这就是"连续批处理"的"连续"之处。

### 3.5 为什么不是"先处理完 A 再处理 B"（连续批处理的核心）

刚接触连续批处理的人都会问：请求是依次到达的，为什么 scheduler 不先处理完 A 再处理 B，而是把 A、B、C 塞进同一步？答案是：**GPU 前向传播一次能装多个序列，一起算和单独算耗时几乎一样，但利用率天差地别**。

#### 你直觉里的"处理"是单线程思维

你以为的处理方式是（这是 CPU 的思维）：

```
先处理A:  [A的6个token prefill][A decode...]  → A完成
再处理B:  [B的3个token prefill][B decode...]  → B完成
再处理C:  [C的4个token prefill][C decode...]  → C完成
```

这样每一步（一次前向传播）GPU 只算了 1 个序列的 token。但 GPU 不是这样工作的——**一次前向传播可以把 A、B、C 的 token 拼在一起算，耗时几乎不变**。

#### 关键事实：前向传播的耗时由"批大小"决定，但不线性增长

```
前向传播一次:
  只算 A 的 6 个 token:  耗时 ≈ 50ms   ← 6个token用满50ms？浪费！
  只算 B 的 3 个 token:  耗时 ≈ 50ms
  只算 C 的 4 个 token:  耗时 ≈ 50ms
  合计: 150ms，处理了 13 个 token

  把 A(6) + B(3) + C(4) 拼成一批:  耗时 ≈ 50~60ms
  合计: 60ms，处理了 13 个 token
```

为什么？因为前向传播的开销大头是 kernel 启动、访存、指令执行这些**固定成本**——不管 batch 里是 6 个 token 还是 13 个 token，这些开销都在。batch 大一点，只是把本来闲置的算力填满。

```
"一个一个处理" = 每次前向都只算一点 → GPU 算力大量闲置 → 慢
"凑满一起处理" = 每次前向都尽量塞满 token → GPU 算力用满 → 快
```

#### 回到那个例子：Step 1 为什么不先处理 A？

Step 1 的池子有 10 个名额：

```
方案1（先处理A）:
  只接纳 A → A 占 6 个名额 → 池子剩 4 个名额白白浪费
  前向传播跑了 50ms，只算了 6 个 token → 利用率 60%

方案2（scheduler 实际做的，尽量塞满）:
  接纳 A(6) + B(3) + C(1) = 10 个名额 → 池子正好满
  前向传播跑了 50ms，算了 10 个 token → 利用率 100%
```

**同样的 50ms，方案2 白赚了 B 和 C 的进度。** 如果请求是源源不断来的（生产环境就是这样），每步都塞满池子，吞吐量直接翻倍甚至更多。

#### 那 C 为什么只切了 1 个 token？

因为池子满了：A 拿了 6 个，B 拿了 3 个，轮到 C 只剩 1 个名额。C 的 prompt 有 4 个 token，这一步只能 prefill 它的第 1 个（cached_len 1/4），下一轮池子腾出名额再 prefill 剩下的 3 个。

关键：**C 没有干等**。它提前开始算了一部分，而不是等 A、B 全部完成才轮到它。如果按"先处理完 A 再处理 B"的方案，C 要等 A 生成完几十个 token 才能开始——那个等待时间就太长了。这就是**分块 prefill** 的价值：长 prompt 不用一次性占满池子，可以一点一点推进，同时和其他序列共享 GPU。

> **一句话**：为什么不是先处理 A？因为 GPU 一次前向传播的成本是**固定的**，装 6 个 token 和装 10 个 token 耗时差不多。所以调度器每次尽量**塞满池子**，让一次前向传播同时推进尽量多的序列。

### 3.6 完整时间线：结合代码逐 Step 推演

#### 先分清两段代码：决定（schedule） vs 执行（execute）

```
代码 A = Scheduler.schedule()   ← 在 scheduler.py 里，属于"调度器"类
代码 B = LLMEngine._execute()   ← 在 engine.py 里，属于"引擎"类
```

它们在一个 step 内先后被调用：

```python
# engine.py
class LLMEngine:
    def step(self):                          # 引擎的每一步都调 step()
        scheduled = self.scheduler.schedule()   # ← 调代码A: 决定做什么
        if not scheduled.prefill_items and not scheduled.decode_items:
            return []                            # 无事可做
        return self._execute(scheduled)          # ← 调代码B: 按清单执行
```

```
engine.step() 每走一步:
  第1步: 代码A schedule()  ──→  只动"队列"和"表格"，不碰模型
         产出: 一张工作清单 ScheduledStep
               prefill_items = [(A,6), (B,3), (C,1)]
               decode_items  = []

  第2步: 代码B _execute() ──→  拿着清单，真正调 model.forward() 计算
         产出: List[RequestOutput]（每个序列这一步的最新输出）
```

**一句话类比**：

```
代码 A (schedule)  = 点菜：决定这一步炒哪些菜、每道菜用多少料
                     （只动 waiting/running 队列和 block_table，不算任何模型）
代码 B (execute)   = 炒菜：按清单下锅，真正调 model.forward() 算 logits
                     （真刀真枪跑前向传播）
```

**为什么拆成两个函数？**
1. **职责分离**：决定（调度）和执行（引擎）解耦。调度器只操心"这一步该让谁干多少活"，不关心模型怎么算。
2. **真实 vLLM 也是这样**：`Scheduler.schedule()` 在 CPU 上跑（纯逻辑，很快），算完把清单交给 GPU worker 执行。
3. **可测试性**：可以单独测调度逻辑，不用跑模型。

#### 推演前提

设 `max_num_batched_tokens=10`，三个请求依次到达：

```
请求A: prompt 6 token
请求B: prompt 3 token
请求C: prompt 4 token
```

**初始状态**：
```
waiting = [A(6), B(3), C(4)]   ← 括号里是 prompt 的 token 数
running = []
```

#### Step 1

**schedule() 执行（代码 A）**：
- running 为空，无清理
- Phase A：running 空，循环不执行 → `decode_items=[]`, `num_batched_tokens=0`
- `remaining = 10 - 0 = 10`
- Phase B1：running 空，跳过
- Phase B2 循环（waiting 非空且 remaining=10 > 0）：
  - **第1圈**：`seq=A`，`_chunk_for(A, need=6, remaining=10) = 6`，A 出队激活进 running，`prefill_items=[(A,6)]`，`remaining=4`
  - **第2圈**：`seq=B`，`_chunk_for(B, 3, 4) = 3`，B 出队激活，`prefill_items=[(A,6),(B,3)]`，`remaining=1`
  - **第3圈**：`seq=C`，`_chunk_for(C, 4, 1) = 1` ← **池子只剩1，C 只能切1块**，C 出队激活，`prefill_items=[(A,6),(B,3),(C,1)]`，`remaining=0`
  - `remaining > 0` 不满足 → 退出

**返回**：`prefill_items=[(A,6),(B,3),(C,1)]`, `decode_items=[]`

**execute() 执行（代码 B，先 prefill 后 decode）**：
- 处理 `(A,6)`：forward 6 个 token → 写KV → `cached_len=6` → `6==num_tokens(6)` → 转 DECODE → 采样 → **A.output_ids=[t1]**
- 处理 `(B,3)`：同理 → **B.output_ids=[t1]**
- 处理 `(C,1)`：forward 1 个 token → `cached_len=1` → `1 != 4` → **留在 PREFILL**

**Step 1 结束**：
```
running = [A: DECODE, cached_len=6, output=[t1]
           B: DECODE, cached_len=3, output=[t1]
           C: PREFILL, cached_len=1, output=[]]
```

#### Step 2

**schedule() 执行（代码 A）**：
- Phase A 遍历 running=[A,B,C]：
  - `A`：不是 in_prefill → 加入 `decode_items`，`num=1`
  - `B`：同 → `decode_items=[A,B]`, `num=2`
  - `C`：`in_prefill=True` → continue 跳过
- `remaining = 10 - 2 = 8`
- Phase B1 遍历 running：
  - `C`：`need = 4 - 1 = 3`，`_chunk_for(C, 3, 8) = 3` → `prefill_items=[(C,3)]`, `remaining=5`
- Phase B2：waiting 空，跳过

**返回**：`prefill_items=[(C,3)]`, `decode_items=[A,B]`

**execute() 执行（代码 B）**：
- 处理 `(C,3)`：forward C 的第2~4个token → `cached_len=1+3=4` → `4==num_tokens(4)` → 转 DECODE → 采样 → **C.output_ids=[t1]**
- 处理 decode `A`：`_sample_and_apply(A, cached_len=6)` → 处理 t1 → 写位置6的KV → `cached_len=7` → 采样 → **A.output_ids=[t1,t2]**
- 处理 decode `B`：同 → **B.output_ids=[t1,t2]**, cached_len=4

**Step 2 结束**：
```
running = [A: DECODE, cached_len=7, output=[t1,t2]
           B: DECODE, cached_len=4, output=[t1,t2]
           C: DECODE, cached_len=4, output=[t1]]
```

#### Step 3

**schedule() 执行（代码 A）**：
- Phase A：A、B、C 都不是 in_prefill → `decode_items=[A,B,C]`, `num=3`
- Phase B1/B2：无 prefill 可做、waiting 空

**execute() 执行（代码 B）**：
- A：decode → `output=[t1,t2,t3]` → `_apply_token` 检查 `len(output_ids)=3 >= max_tokens=3` → `stop_reason="length"` → **`_finish(A)`**：register_prefix（缓存前缀）→ release（释放KV块）→ state=FINISHED → 从 running 移除
- B：decode → `output=[t1,t2,t3]`
- C：decode → `output=[t1,t2]`

**Step 3 结束**：
```
running = [B: DECODE, output=[t1,t2,t3]
           C: DECODE, output=[t1,t2]]
```

#### Step 4 及以后

- B、C 继续 decode。假设 B 也达到 max_tokens → 完成、释放块。
- 如果此时新请求 D 到达：D 进 waiting → 下一个 Step 的 Phase B2 会接纳它（running 有位置、池子有名额）→ **这就是"连续批处理"的"连续"：完成一个，立刻补一个，不需要等整批结束。**

**贪心塞满的代价与收益**：代价是 C 的 prefill 被切成 1+3 两块（分块 prefill）；收益是 GPU 每一轮都跑满 10 个名额而不是只跑 6 个。C 在 Step 1 就起步，而不是干等到 A、B 全部完成。

### 3.7 为什么这能提高吞吐量

假设 GPU 一次前向传播的耗时：
- Prefill 128 tokens: 50ms（计算密集，算力被打满）
- Decode 1 token: 5ms（访存密集，算力大量闲置）

如果 8 个请求同时 decode，耗时还是约 5ms（因为 decode 的瓶颈是访存不是计算，批处理摊薄了访存开销）。

```
传统方式（一个一个来）:
  8个请求 × 各20个decode步 × 5ms/步 = 800ms

连续批处理（8个一起decode）:
  20个decode步 × 5ms/步 = 100ms  ← 8倍加速！
```

再加上 decode 时 GPU 算力闲置，可以同时插入 prefill，进一步利用算力。

---

## 第 4 章 引擎的完整运转流程

> 对应代码：`minivllm/engine.py`

### 4.1 引擎的组成

```python
# engine.py
class LLMEngine:
    def __init__(self, config, tokenizer, model):
        self.model = model                              # TinyGPT 模型
        self.tokenizer = tokenizer                      # 字符级tokenizer
        self.block_manager = BlockManager(...)          # KV Cache管理
        self.scheduler = Scheduler(...)                 # 调度器
        self.rng = np.random.default_rng(config.seed)   # 随机数生成器
        self.seqs = {}                                  # 所有序列的注册表
```

### 4.2 一个请求的完整生命周期（数据快照版）

用一个最小的例子走完全程：`prompt="hi"`，假设字符 tokenizer 把它编码成 `[7, 8]`，`max_tokens=3`。全程只需 **3 个 engine step**（1 步 prefill + 2 步 decode）。

先记住 Sequence 的全部字段（这就是"一个请求"的全部数据）：

```python
seq_id       # 请求编号
prompt_ids   # 输入的token列表
output_ids   # 生成的token列表
cached_len   # 已写入KV cache的token数 / decode时=下一步要处理的位置
phase        # "PREFILL" | "DECODE"
state        # WAITING | RUNNING | FINISHED
stop_reason  # None | "stop" | "length"
```

#### 阶段 0：add_request（请求出生）

```python
seq_id = engine.add_request("hi", SamplingParams(max_tokens=3))
```

add_request 内部：
```python
prompt_ids = tokenizer.encode("hi")   # → [7, 8]
seq = Sequence(seq_id=0, prompt_ids=[7, 8], sampling_params=..., priority=...)
self.seqs[0] = seq
scheduler.add_sequence(seq)           # waiting = [seq0]
```

**快照**：
```
seq0: prompt_ids=[7,8]  output_ids=[]  cached_len=0  phase=PREFILL  state=WAITING  stop=None
KV cache: (空)
```

#### 阶段 1：Step 1 —— prefill（第一次 step()）

**schedule()（只规划，不计算）**：
```
waiting=[seq0], running=[]
Phase A: running空 → 跳过
Phase B1: running空 → 跳过
Phase B2: 接纳seq0 → need = 2-0 = 2 → chunk = min(2, 512) = 2
  waiting=[]  running=[seq0]  prefill_items=[(seq0, 2)]
```

**execute()（真正计算）**：
```python
# 处理 prefill 项 (seq0, 2)
start    = seq0.cached_len = 0
tokens   = all_ids[0:2] = [7, 8]        # prompt 的全部 2 个 token
positions= [0, 1]
_run_forward → model.forward([7,8], [0,1])   # 写位置0、1的KV
seq0.cached_len = 0 + 2 = 2

# 检查: 2 == num_tokens(2) → prefill 完成！
seq0.phase = "DECODE"
seq0.cached_len = 2 - 1 = 1              # decode约定（见4.3）
_sample_and_apply(seq0, 1):
  token = all_ids[1] = 8                 # 处理最后一个prompt token
  forward([8], [1]) → 重写位置1的KV → seq0.cached_len = 1+1 = 2
  采样 → 42 → output_ids = [42]
  _apply_token: len(output_ids)=1 < max_tokens=3 → 不停止
```

**快照**：
```
seq0: prompt_ids=[7,8]  output_ids=[42]  cached_len=2  phase=DECODE  state=RUNNING  stop=None
KV cache: 位置0(7) ✅  位置1(8) ✅
```

#### 阶段 2：Step 2 —— 第一次 decode

**schedule()**：
```
running=[seq0]，seq0 已经是 DECODE
Phase A: seq0 进 decode_items → decode_items=[seq0]
```

**execute()**：
```python
_sample_and_apply(seq0, cached_len=2):     # 处理位置2的token
  token = all_ids[2] = 42                  # ← 位置2就是上一步采样出的42！
  forward([42], [2]) → 写位置2的KV → seq0.cached_len = 2+1 = 3
  采样 → 55 → output_ids = [42, 55]
  _apply_token: len=2 < 3 → 不停止
```

**快照**：
```
seq0: prompt_ids=[7,8]  output_ids=[42,55]  cached_len=3  phase=DECODE  state=RUNNING  stop=None
KV cache: 位置0(7) 位置1(8) 位置2(42) ✅
```

#### 阶段 3：Step 3 —— decode + 触发停止（请求死亡）

**execute()**：
```python
_sample_and_apply(seq0, cached_len=3):
  token = all_ids[3] = 55
  forward([55], [3]) → 写位置3的KV → seq0.cached_len = 4
  采样 → 9 → output_ids = [42, 55, 9]
  _apply_token 检查:
    len(output_ids)=3 >= max_tokens=3 → stop_reason = "length"
  _finish(seq0):
    register_prefix(seq0)   # 把prompt [7,8] 的KV块存入PrefixCache（供后续请求复用）
    release(seq0)           # 释放物理块，还给空闲列表
    state = FINISHED        # 从running移除
```

**快照**：
```
seq0: prompt_ids=[7,8]  output_ids=[42,55,9]  cached_len=4  phase=DECODE  state=FINISHED  stop="length"
KV cache: 已释放，块回到空闲列表
```

#### 收尾：generate() 的循环

`generate()` 就是不断 `step()` 直到 finished：

```python
def generate(self, prompt, sampling_params):
    seq_id = self.add_request(prompt, sampling_params)   # 阶段0
    seq = self.seqs[seq_id]
    while not seq.is_finished:
        self.step()                                      # 阶段1、2、3
    return RequestOutput.from_sequence(seq, self.tokenizer)
    # prompt="hi", text=decode([42,55,9]) ← 用户拿到的最终结果
```

#### 生命周期全景图

```
add_request ─→ WAITING ──→ RUNNING(PREFILL) ──→ RUNNING(DECODE) ──→ FINISHED
                 │              │                      │
                 │         Phase B2 接纳         每步decode 1个token
                 │         prefill全部/部分      直到触发停止条件
                 │              │                      │
                 │         cached_len ==            _finish():
                 │         num_tokens                注册前缀缓存
                 │         → 转DECODE+采样          释放KV块
                 │                                   置FINISHED
```

#### 六个关键点（对照上面看）

1. **每一步 step() = schedule()（规划） + execute()（执行）**：前者只动队列和表格，后者才跑模型。
2. **phase 只在 execute 里变**（PREFILL→DECODE），因为要等 forward 跑完才知道 `cached_len` 有没有凑满。
3. **cached_len 的含义**：PREFILL 时 = 已写 KV 的 token 数；DECODE 时 = 下一步要处理的位置。
4. **KV cache 的位置 = 序列的绝对位置**（prompt + 输出统一编号）：位置 0、1 是 prompt，位置 2 是第一个输出 token（42），位置 3 是 55……每个 token 的 KV 按它在整个序列中的位置存放。
5. **停止检查在 `_apply_token`**：EOS / max_model_len / max_tokens / 停止字符串，任一命中就置 stop_reason。
6. **死亡 = `_finish`**：前缀入缓存 → KV 块释放 → 状态置 FINISHED → 从 running 移除。块释放后物理空间立刻可以被其他请求使用。

### 4.3 prefill 和 decode 的统一处理

mini-vLLM 有一个巧妙的约定来统一这两条路径：

```python
# prefill完成后:
seq.cached_len -= 1  # 为什么减1？
```

#### 先记一条铁律

**`model.forward(x, pos)` 会把传入的每个 token 的 KV 写进 cache**（`model.py` 里 `write_kv`）。调一次 forward、传几个 token，就有几个 token 的 KV 被新写入。

#### A 的完整数值账

设 A 的 prompt = 6 个 token（位置 0~5），max_tokens = 3：

```
① prefill (A, 6):
     forward([p0..p5], [0..5]) → 写入位置0~5的KV → KV cache: [0~5] ✅
     cached_len = 0 + 6 = 6      ← 此时"已写KV的token数"= 6
     返回的 logits[0,-1] 是位置5的预测，它预测位置6是谁

② 检查: 6 == num_tokens(6) → prefill 完成
     phase = DECODE
     cached_len = 6 - 1 = 5      ← 减1！（原因见下）
     _sample_and_apply(A, 5):

③ _sample_and_apply(A, 5):
     处理 all_ids[5] = p5（最后一个prompt token）
     forward([p5], [5]) → 重新写位置5的KV → cached_len = 5 + 1 = 6
     采样 → t1 → A.output_ids = [t1]

④ decode: _sample_and_apply(A, 6):
     处理 all_ids[6] = t1 → forward 写位置6的KV → cached_len = 7
     采样 → t2 → A.output_ids = [t1, t2]

⑤ decode: _sample_and_apply(A, 7):
     处理 all_ids[7] = t2 → forward 写位置7的KV → cached_len = 8
     采样 → t3 → A.output_ids = [t1, t2, t3] → 达到max_tokens → FINISHED
```

| 步骤 | 动作 | KV cache 已有 | cached_len |
|------|------|--------------|-----------|
| ① prefill | forward 写位置0~5 | [0~5] ✅ | 0 → 6 |
| ② 完成检查 | prefill 完，转 DECODE | [0~5] ✅ | 6 → 5（减1）|
| ③ 第一次采样 | 处理 p5，重写位置5 | [0~5] ✅ | 5 → 6 |
| ④ decode | 处理 t1，写位置6 | [0~6] ✅ | 6 → 7 |
| ⑤ decode | 处理 t2，写位置7 | [0~7] ✅ | 7 → 8 |

#### 为什么非减 1 不可？

**核心原因：采样第一个输出 token 的"输入 token"是 p5，不是 t1。**

- prefill 的 forward 处理完 p0~p5 后，返回的 logits 最后一列是 **p5 的预测**（预测位置 6 的 token）
- 你**只能用 p5 的 logits 采样出 t1**；t1 是采样的"结果"，在采样之前它还不存在
- 所以 prefill 完成后"下一步要处理的 token"只能是 p5（位置 5）

`_sample_and_apply(seq, input_position)` 的约定是"处理 `all_ids[input_position]` 这个 token"。prefill 刚结束时 cached_len=6，如果**不减 1** 就调 `_sample_and_apply(seq, 6)`，会去访问 `all_ids[6]`——而 `all_ids[6]` 就是 t1，它还没出生，直接越界。

减 1 之后，`_sample_and_apply(seq, 5)` 处理 p5，和后续 decode 的调用形式完全一样，统一了代码路径：

```
① prefill完成 → _sample_and_apply(seq, cached_len)   ← 处理p5 → 采样t1
② decode 每步 → _sample_and_apply(seq, cached_len)   ← 处理最新token → 采样下一个
```

#### KV 数据没丢

减 1 动的只是 cached_len 这个"指针"，KV 数据原地不动：

```
减 1 前 KV cache: [p0][p1][p2][p3][p4][p5]   ← 6个都在
减 1 后 KV cache: [p0][p1][p2][p3][p4][p5]   ← 还是这6个
```

#### 代价与回报

- **代价**：位置 5 的 KV 被重复写了一遍（① 写一次，③ 又写一次）。内容相同，无害冗余。
- **回报**：prefill 与 decode 的采样路径 100% 统一，引擎只有一个 `_sample_and_apply(seq, cached_len)` 处理所有"生成 token"的逻辑。

#### 一句话总结

> prefill 结束时 KV 里存了 6 个 token（位置0~5），但你要采样第一个输出 token，必须用位置5的 logits——所以把 cached_len 退到5，然后像 decode 一样"处理p5 → 写KV → +1 → 采样"。**减 1 不是数学需要，是代码统一的需要**：cached_len 在 prefill 阶段表示"已写KV的token数"，在 decode 阶段表示"下一步要处理的位置"，减 1 就是切换语义。

```python
# _sample_and_apply: decode 的核心
def _sample_and_apply(self, seq, input_position):
    token_id = seq.all_ids[input_position]       # 要处理的token
    logits = self.model.forward(
        x=[[token_id]], positions=[[input_position]],
        seqs=[seq], block_manager=self.block_manager
    )[0, -1, :]                                   # 取最后一个位置的logits
    seq.cached_len += 1                           # 这个token的KV已写入
    token, logprob = sample_token(logits, params, rng)
    seq.append_token(token)                       # 加入输出
```

### 4.4 停止条件检查

```python
# engine.py: _apply_token()
def _apply_token(self, seq, token_id, logprob):
    seq.append_token(token_id)
    seq.cumulative_logprob += logprob

    # 检查停止条件（顺序和vLLM一致）
    if not seq.sampling_params.ignore_eos and token_id == self.tokenizer.eos_token_id:
        seq.stop_reason = "stop"           # 遇到EOS
    elif seq.num_tokens >= max_model_len:
        seq.stop_reason = "length"         # 超过模型最大长度
    elif len(seq.output_ids) >= seq.sampling_params.max_tokens:
        seq.stop_reason = "length"         # 超过请求的max_tokens
    else:
        # 检查停止字符串
        text = self.tokenizer.decode(seq.output_ids)
        for s in seq.sampling_params.stop or []:
            if s and text.endswith(s):
                seq.stop_reason = "stop"   # 遇到停止字符串
                break

    if seq.stop_reason is not None:
        self._finish(seq)
```

### 4.5 生成 API

```python
# 非流式: 一次性生成
output = engine.generate("vLLM is", SamplingParams(max_tokens=32))

# 流式: 逐token返回
for output in engine.generate_stream("vLLM is", SamplingParams(max_tokens=32)):
    print(output.outputs[0].text, end="", flush=True)
```

`generate` 是阻塞的：内部循环 `step()` 直到序列完成。
`generate_stream` 是生成器：每次 `step()` 后 yield 当前输出。

---

## 第 5 章 进阶机制：前缀缓存、抢占、分块 Prefill

### 5.1 前缀缓存 (Prefix Caching)

> 对应代码：`kv_cache.py: PrefixCache`

**问题**：多个请求有相同的前缀（比如系统提示词），每个请求都重新计算 prefill 是浪费。

**解决**：把已完成请求的 prompt 前 N 个 block 的 KV 缓存起来，新请求如果前缀匹配，直接共享这些物理块。

```python
# kv_cache.py: PrefixCache
class PrefixCache:
    """LRU缓存: token前缀 → 物理块列表"""

    _cache: Dict[tuple, List[int]]  # 前缀token元组 → 物理块ID列表
    _lru: OrderedDict               # LRU淘汰

    def get(self, prefix_tokens: tuple):
        """查找是否有缓存的前缀"""
        if prefix_tokens in self._cache:
            self._lru.move_to_end(prefix_tokens)  # 更新LRU
            return list(self._cache[prefix_tokens])
        return None

    def put(self, prefix_tokens: tuple, block_ids: List[int]):
        """请求完成后，把prompt的KV块存入缓存"""
        self._cache[prefix_tokens] = list(block_ids)
        for b in block_ids:
            self.allocator.touch(b)  # 缓存持有引用，防止被回收
        # LRU淘汰
        while self._size > self.max_cached_blocks:
            self._evict_one()
```

**匹配过程**（最长前缀匹配）：

```python
# kv_cache.py: match_prefix()
def match_prefix(self, prompt_ids: List[int]):
    """找最长的已缓存前缀"""
    for L in range(len(prompt_ids), 0, -1):  # 从长到短试
        blocks = self.prefix_cache.get(tuple(prompt_ids[:L]))
        if blocks is not None:
            return L, blocks  # 命中！返回前缀长度和物理块
    return 0, []  # 没命中
```

**命中后的效果**（最长前缀匹配）：

```
请求A已完成: prompt = "You are a helpful assistant. Tell me about..."
  → register_prefix(A): A 的【整个】prompt 都入缓存（token序列 → KV块列表）

前缀缓存: {(A的完整prompt token序列): [block_3, block_7]}

请求B到来: prompt = "You are a helpful assistant. What is vLLM?"
  match_prefix 做最长前缀匹配:
    B 的前 27 个 token（"You are a helpful assistant."）与 A 相同 → 命中
  → attach_shared_blocks: seq_B 复用 block_3, block_7
  → seq_B.cached_len = 27  ← 跳过27个token的prefill！
  → 只需 prefill 剩余的 "What is vLLM?" 部分

注意: 不是"只缓存前半句"。A 的整个 prompt 都在缓存里，
     能共享多少取决于两个请求的公共前缀有多长。
     如果 B 的 prompt 与 A 完全相同 → prompt_remaining==0
     → 完全跳过 prefill，直接进 decode（见 scheduler 的对应分支）。
```

**注意**：只有 prompt 部分的块被缓存（不是生成内容），因为不同请求的生成内容不同，不应共享。

#### 为什么需要前缀缓存：KV 块是动态的，会被覆盖

容易误解的点："prompt 的 KV 不是本来就在 KV cache 里么，还要前缀缓存干嘛？"

关键：**KV cache 的块是动态的——序列结束后就被释放，马上会被其他序列覆盖。** 不做前缀缓存，prompt 的 KV 很快就不存在了：

```
没有前缀缓存:
  A 完成 → release(A) → A的块(含prompt KV)回空闲列表
  B 到来 → allocate 拿到这个块 → 写入B的数据 → A的prompt KV 被覆盖！没了
  C 用相同prompt到来 → 找不到 → 只能重新prefill（重新算一遍）

有前缀缓存:
  A 完成 → register_prefix: prompt部分的块 → 转给PrefixCache（引用+1，不被回收）
          release: 序列放手（引用-1，但缓存仍持有）
  块活着，数据还是A的prompt KV
  C 用相同prompt到来 → match_prefix 命中 → 直接复用 → 跳过prefill
```

**前缀缓存不是"再复制一份 KV 数据"，而是用引用计数延长 KV 块的寿命。**

#### 释放 ≠ 清数据：所有权转移

`_finish` 的顺序（engine.py）：

```python
def _finish(self, seq):
    if self.config.enable_prefix_caching and not seq.is_preempted:
        self.block_manager.register_prefix(seq)   # ① 先登记前缀
    self.block_manager.release(seq)               # ② 再释放
    seq.state = FINISHED
```

用 4.2 的例子（seq0 的 prompt [7,8] 占物理块 5）：

```
register_prefix(seq0):
  prefix_cache.put((7,8), [5])
  allocator.touch(5)      # ref_count[5]: 1 → 2   （序列 + 缓存各持1）

release(seq0):
  allocator.free_block(5) # ref_count[5]: 2 → 1   （序列放手，缓存接手）
  → ref != 0 → 块不回收！

结果: 物理块5的数据还在（KV数值原封不动），owner 变成 PrefixCache
```

**"清除"的是序列的所有权，不是数据**。数据只在下次 allocate 分配该块、写入新 KV 时才被覆盖。系统从不清空内存（清内存很贵）。

#### 运行时 KV Cache vs 前缀缓存

| | 运行时 KV Cache (KVStore) | 前缀缓存 (PrefixCache) |
|---|---|---|
| 存什么 | **输入 + 全部生成**的 KV | 只存 **prompt（输入）** 部分的块 |
| 给谁用 | 当前序列自己（decode 注意力要用） | 未来的新请求（跳过 prefill） |
| 生命周期 | 序列存活期间持续增长 | 序列结束后仍保留（LRU 淘汰） |
| 为什么生成部分也在 | decode 必须能看到自己生成过的 token | 生成是随机的，无跨请求复用价值 |

#### 块的一生（生命周期全景）

```
allocate（分配，ref=1）
   → 序列用（写入KV）
   → 序列结束
       ├─ 若是 prompt 部分的块:
       │    register_prefix: 前缀缓存 +1 → release: 序列 -1 → ref=1（活着）
       │    → 未来请求 match_prefix 命中 → touch +1 → 新序列用 → 结束 → release -1
       │    → 一直活在缓存里，直到 LRU 淘汰 → ref=0 → 回空闲列表
       └─ 若是生成部分的块:
            release: ref 1→0 → 回空闲列表
            → 下次 allocate 分配出去，新数据覆盖旧数据
```

#### 与厂商的"缓存命中"是同一个东西

DeepSeek 的 Context Caching、Anthropic 的 Prompt Caching、OpenAI 的 Prompt Caching，本质就是 vLLM 的前缀缓存：

- 相同前缀的 prompt → KV 直接复用 → 不重新算 prefill
- 厂商对"命中缓存的输入 token"打折计费（DeepSeek 的 cache hit 输入价格约为 miss 的 1/10）
- 你用同一个 system prompt（2000 token）发 10000 个请求——第一个请求算一次 prefill，后面 9999 个全部命中复用，省的是 GPU 计算

#### 一句话总结

> KV Cache 全程缓存所有 token（含生成），但块是动态的，序列结束就会被释放、被覆盖。前缀缓存 = 在序列结束的瞬间，用引用计数把 prompt 部分的块"救下来"，让它们活过所属序列、留给未来请求复用。它不加数据、不复制数据，只是改变 KV 块的所有权和生命周期——这正是厂商"缓存命中"的底层实现。

### 5.2 抢占 (Preemption)

> 对应代码：`scheduler.py: _preempt_one()`

**问题**：KV Cache 满了，但又来了新请求，怎么办？

**解决**：驱逐最低优先级的运行中序列，释放它的 KV 块给更需要的人。

```python
# scheduler.py: _preempt_one()
def _preempt_one(self, step):
    # 从running列表末尾往前找（最低优先级）
    for i in range(len(self.running) - 1, -1, -1):
        seq = self.running[i]
        if seq.seq_id in self._newly_admitted:
            continue  # 刚接纳的不驱逐（否则白接纳了）

        # 释放它的所有KV块
        self.block_manager.release(seq)

        # 重置缓存状态
        seq.cached_len = 0          # KV全没了
        seq.phase = "PREFILL"      # 需要重新prefill
        seq.is_preempted = True
        seq.state = WAITING

        # 放回等待队列头部（优先重新调度）
        self.waiting.appendleft(seq)

        # 从当前step的计划中移除它（如果已计划）
        step.prefill_items = [(s,c) for s,c in step.prefill_items
                              if s.seq_id != seq.seq_id]
        step.decode_items = [s for s in step.decode_items
                             if s.seq_id != seq.seq_id]
        return True
    return False
```

**关键点**：被抢占的序列**已生成的 token 不会丢失**！

```python
# sequence.py: all_ids 属性
@property
def all_ids(self) -> List[int]:
    """prompt + 已生成的输出"""
    return self.prompt_ids + self.output_ids
```

重新调度时，`all_ids` 包含了之前生成的 token，会一起重新 prefill。虽然要重算 KV，但不会丢失已有输出。

**RECOMPUTE vs SWAP**：

| 模式 | 做法 | 优点 | 缺点 |
|------|------|------|------|
| RECOMPUTE | 丢弃KV，重算 | 实现简单，不占CPU内存 | 浪费计算 |
| SWAP | KV从GPU搬到CPU | 不浪费计算 | 需要CPU内存，搬运有延迟 |

mini-vLLM 只实现了 RECOMPUTE 模式。真实 vLLM 两种都支持。

#### 抢占只是兜底：vLLM 的"解题顺序"

先明确：**vLLM 解决显存问题的主要手段不是"搬迁"，而是"少占、共享、复用"**。搬迁要拷贝数据，非常贵，是最后手段。完整解题顺序（从便宜到贵）：

```
显存不够装所有请求的 KV Cache
  ↓
1. 分页分配 ── 用多少分多少，不浪费          ← 便宜，常用
2. 前缀共享 ── 相同前缀的KV只算一份          ← 便宜，常用
3. 前缀复用 ── 缓存prompt前缀，跳过prefill   ← 便宜，常用
  ↓ 还不够？
4. 抢占 RECOMPUTE ── 丢KV重算              ← 中等成本，兜底
5. 抢占 SWAP ── 显存↔内存搬迁              ← 最贵，最后手段
```

前三步让 KV Cache 在显存里本身用得极少，大多数情况下根本轮不到抢占和搬迁。

#### SWAP 搬去哪：显存 ↔ CPU 内存（不是磁盘）

三层存储速度排序：**显存 (GPU) > 内存 (CPU RAM) > 磁盘 (SSD)**

注意 vLLM 的 SWAP 慢层是 **CPU 内存**而不是磁盘，这和 OS 的 swap 有个重要差别：

| | OS 的 swap | vLLM 的 SWAP |
|---|---|---|
| 快层 | 内存 (RAM) | 显存 (VRAM) |
| 慢层 | 磁盘 (SSD) | CPU 内存 (RAM) |
| 为什么 | 磁盘便宜容量大 | CPU 内存速度可接受，放磁盘就太慢了 |

vLLM 的 KV Cache **永远不会直接进磁盘**。

#### mini-vLLM 为什么只实现 RECOMPUTE

```python
# config.py
@dataclass
class SchedulerConfig:
    preemption_mode: str = "recompute"  # "recompute" | "swap"
    # 注释: only recompute implemented
```

原因：RECOMPUTE 实现简单（丢掉KV，重算），代码只有 20 行；SWAP 要管理 CPU 侧内存、拷贝逻辑、同步，复杂得多。教学项目优先展示机制。真实 vLLM 两者都支持，且很多生产部署直接禁用 preemption（`--swap-space 0`），靠前三步撑住吞吐量。

**测试验证**（`test_preemption_recovers`）：

```python
# 故意把KV cache设得很小（20个block × 8 = 160个token位置）
# 6个长prompt（每个~60 token）无法同时驻留
# 验证所有请求最终都能完成
cfg = EngineConfig(
    cache=CacheConfig(block_size=8, num_gpu_blocks=20),
    scheduler=SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=64),
    enable_prefix_caching=False,  # 关掉前缀缓存，确保抢占真的发生
)
# ... 添加6个请求 ...
while e.has_pending() and steps < 20000:
    e.step()
assert not e.has_pending()  # 全部完成！
```

### 5.3 分块 Prefill (Chunked Prefill)

> 对应代码：`scheduler.py: _chunk_for()`

**问题**：一个很长的 prompt（比如 4096 tokens），如果必须一次性 prefill，会：
1. 超过 `max_num_batched_tokens` 预算
2. 需要大量连续空闲块（可能不够）

**解决**：把长 prompt 分成多个 chunk，每个 step 处理一部分。

```python
# scheduler.py: _chunk_for()
def _chunk_for(self, seq, need, remaining):
    """计算本轮能处理多少token，受双重限制"""
    block_size = self.block_manager.block_size
    have_blocks = len(self.block_manager.get_block_table(seq))
    max_tokens_by_blocks = (have_blocks + self.block_manager.num_free_blocks) * block_size
    extra_by_blocks = max(0, max_tokens_by_blocks - seq.cached_len)
    return max(0, min(need, remaining, extra_by_blocks))
    #           ↑     ↑          ↑
    #        需要的  token预算   空闲块能放多少
```

**图解**：

```
prompt = 1000 tokens, max_num_batched_tokens = 256, block_size = 16

Step 1: chunk = min(1000, 256, 可用块×16) = 256
  → prefill token 0~255, cached_len = 256

Step 2: chunk = min(744, 256, 可用块×16) = 256
  → prefill token 256~511, cached_len = 512

Step 3: chunk = min(488, 256, 可用块×16) = 256
  → prefill token 512~767, cached_len = 768

Step 4: chunk = min(232, 256, 可用块×16) = 232
  → prefill token 768~999, cached_len = 1000 → prefill完成！
```

**等价性保证**（`test_chunked_prefill_matches_full`）：

```python
# 256 token预算（一次prefill完）
full_engine = make_engine(max_batched_tokens=256)
_, ids_full = gen_text(full_engine, prompt)

# 6 token预算（极端分块，分很多步）
chunked_engine = make_engine(max_batched_tokens=6)
_, ids_chunked = gen_text(chunked_engine, prompt)

assert ids_full == ids_chunked  # 结果完全一样！
```

**为什么结果一样？** 因为注意力是因果的——每个 token 只看它之前的 token，不管你是一步算还是分步算，最终的 KV Cache 内容是一样的。

---

## 第 6 章 从 TinyGPT 到真实模型：架构差异

### 6.1 TinyGPT 的架构

```python
# model.py: TinyGPT 的一个 Transformer 层
def forward(self, x, positions, seqs, block_manager):
    h = wte[x] + wpe[positions]  # 嵌入

    for layer in range(n_layer):
        # 1. LayerNorm + QKV
        ln1 = _layernorm(h, ln1_w[layer], ln1_b[layer])
        qkv = ln1 @ c_attn_w[layer] + c_attn_b[layer]
        q, k, v = np.split(qkv, 3, axis=-1)

        # 2. 写KV到分页cache + paged attention
        block_manager.write_kv(layer, seq, pos, k, v)  # 写
        attn = paged_attention_batch(q, k, v, ...)     # 读+算

        # 3. 残差
        h = h + attn @ c_proj_w[layer]

        # 4. MLP (两层全连接 + GELU)
        ln2 = _layernorm(h, ln2_w[layer], ln2_b[layer])
        mlp = _gelu(ln2 @ c_mlp_w[layer] + c_mlp_b[layer])
        h = h + mlp @ c_mlp_p_w[layer]

    logits = _layernorm(h, ln_f_w, ln_f_b) @ wte.T  # tied lm_head
```

### 6.2 真实模型的架构差异

| 组件 | TinyGPT (GPT-2风格) | LLaMA | DeepSeek-V3 |
|------|-------|-------|-------------|
| **位置编码** | 可学习绝对位置 | RoPE（旋转位置编码） | RoPE |
| **归一化** | LayerNorm | RMSNorm | RMSNorm |
| **注意力** | 标准多头注意力(MHA) | GQA（分组查询注意力） | MLA（多头潜在注意力） |
| **激活函数** | GELU | SwiGLU | SwiGLU |
| **前馈网络** | 2层MLP | SwiGLU (3层带门控) | MoE（混合专家） |
| **lm_head** | 与嵌入权重绑定 | 独立权重 | 独立权重 |

下面逐一解释。

### 6.3 GQA (Grouped Query Attention)

```
标准多头注意力 (MHA):
  n_heads 个 Query 头, n_heads 个 Key 头, n_heads 个 Value 头
  → KV Cache 大小 ∝ n_heads

分组查询注意力 (GQA):
  n_heads 个 Query 头, n_kv_groups 个 KV 头 (n_kv_groups < n_heads)
  多个 Query 头共享一对 KV 头
  → KV Cache 大小 ∝ n_kv_groups （更小！）

极端情况 = MQA (Multi-Query Attention):
  n_heads 个 Query 头, 1 个 KV 头
  → KV Cache 最小，但质量可能下降

LLaMA-3-8B: n_heads=32, n_kv_groups=8  → KV Cache 减少 4 倍
```

在引擎层面，唯一变化是 `KVStore` 的 `num_heads` 参数变成 `n_kv_groups`，attention 计算时多个 query 头共享同一组 KV。

### 6.4 MLA (Multi-head Latent Attention) — DeepSeek 的创新

DeepSeek-V2/V3 使用 MLA 进一步压缩 KV Cache：

```
标准注意力:
  K = x @ W_K  → shape (seq, n_heads, head_dim)  ← 存入KV Cache
  V = x @ W_V  → shape (seq, n_heads, head_dim)

MLA:
  c = x @ W_D  → shape (seq, d_c)  ← 低秩压缩，d_c << n_heads × head_dim
  存入KV Cache 的只是压缩向量 c（很小！）
  推理时: K = c @ W_UK, V = c @ W_UV  ← 解压回完整K/V
```

**对引擎的影响**：
- KV Cache 的 shape 从 `(blocks, block_size, n_heads, head_dim)` 变成 `(blocks, block_size, d_c)`
- Attention 计算时需要先解压
- 但**调度器、block table、前缀缓存等机制完全不变**

### 6.5 RoPE (Rotary Position Embedding)

TinyGPT 用可学习的绝对位置编码：`h = wte[x] + wpe[positions]`

LLaMA/DeepSeek 用 RoPE：不显式加位置编码，而是在 attention 计算时对 Q 和 K 做旋转：

```
q_rotated = q * cos(θ) + rotate_half(q) * sin(θ)
k_rotated = k * cos(θ) + rotate_half(k) * sin(θ)
```

其中 θ 是位置相关的角度。

**对引擎的影响**：在 `model.py` 的 forward 中，QKV 投影后对 Q 和 K 做旋转即可。其他不变。

### 6.6 RMSNorm

```python
# LayerNorm (TinyGPT)
def _layernorm(x, w, b):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b

# RMSNorm (LLaMA/DeepSeek)
def _rmsnorm(x, w):
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return x / rms * w  # 没有减均值，没有偏置b
```

更简单、更快、效果差不多。

### 6.7 SwiGLU

```python
# TinyGPT 的 MLP (2层 + GELU):
mlp = gelu(x @ W1 + b1) @ W2 + b2

# LLaMA 的 SwiGLU (3层，带门控):
gate = silu(x @ W_gate)       # 门控信号
value = x @ W_value            # 值
mlp = (gate * value) @ W_down  # 门控 × 值 → 降维
```

### 6.8 MoE (Mixture of Experts)

DeepSeek-V3 的 FFN 被替换成 MoE：

```
标准 FFN:
  每个 token 都过同一个 MLP

MoE:
  每个 token 先过 Router (一个小的线性层)
  Router 输出每个专家的权重
  选 Top-K 个专家 (DeepSeek-V3: 256选8)
  token 只被选中的专家处理
  最终输出 = 加权平均(各专家输出)
```

**对推理引擎的影响**：

1. **模型权重结构变化**：不再是简单的 `W1, W2` 矩阵，而是 256 组专家权重 + router 权重
2. **计算量变化**：每个 token 只激活 8/256 = 3.1% 的参数，但权重总量很大
3. **显存需求**：虽然计算少，但所有专家的权重都要加载到显存
4. **调度器不变**：连续批处理、PagedAttention 等机制完全不受影响

---

## 第 7 章 量化、投机解码与服务化

### 7.1 量化 (Quantization)

> 对应代码：`minivllm/quantize.py`

**目的**：减少权重（和 KV Cache）的内存占用。

```python
# quantize.py: 对称 int8 量化
def quantize_matrix(w):
    """每列一个 scale，使 max(|w|/scale) <= 127"""
    scale = np.max(np.abs(w), axis=0) / 127.0
    q = np.round(w / scale).astype(np.int8)  # float32 → int8
    return q, scale  # 存 int8 权重 + float32 scale
```

**效果**：
```
float32 权重: 4 字节/参数
int8 权重:    1 字节/参数 + scale(4字节/列)
对于大矩阵: 压缩比 ≈ 4x

例: 7B 模型 float32 → 28 GB
              int8  → ~7 GB  (能放进单张 8GB 显卡)
```

**真实 vLLM 的量化方案**：

| 方案 | 原理 | 特点 |
|------|------|------|
| GPTQ | 训练后量化，逐层校准 | 精度高，需要校准数据 |
| AWQ | 激活感知量化 | 保护重要通道 |
| FP8 | 8位浮点 | 硬件原生支持，速度快 |
| GGUF | llama.cpp 格式 | CPU/GPU 混合推理 |

### 7.2 投机解码 (Speculative Decoding)

> 对应代码：`minivllm/speculative.py`

**核心思想**：用一个小模型（draft）猜 K 个 token，大模型（target）一次验证全部 K 个。

```
正常 decode (生成 K 个 token):
  大模型前向 × K 次  ← 每次只生成1个token，K次串行

投机 decode:
  1. 小模型猜 K 个 token  (很快)
  2. 大模型一次前向验证 K 个  (1次前向，并行验证)
  3. 接受匹配的前缀 + 1 个 bonus token
  
  如果猜对了 m 个: 实际得到 m+1 个 token，只做了 1 次大模型前向
  如果全猜错: 得到 1 个 token（bonus），浪费了小模型的开销
```

**mini-vLLM 的实现**：

```python
# speculative.py
# 草稿模型: 字符 bigram (统计模型，极快)
class BigramDraftModel:
    def draft(self, context, num_tokens):
        """贪心预测 num_tokens 个token"""
        tokens = []
        cur = context[-1]
        for _ in range(num_tokens):
            cur = int(np.argmax(self.log_probs[cur]))
            tokens.append(cur)
        return tokens

# 投机生成
def speculative_generate(model, draft_model, prompt_ids, ...):
    # 1. prefill prompt
    # 2. 循环:
    while not done:
        # draft: 猜K个
        draft = draft_model.draft(seq.all_ids, K)

        # verify: 大模型一次前向验证
        vlogits = model.forward(draft_tokens, ...)
        preds = [argmax(last_logits)] + [argmax(vlogits[j]) for j in range(K-1)]

        # accept: 找最长匹配
        m = 0
        while m < K and draft[m] == preds[m]:
            m += 1

        # 保留接受的 + bonus
        kept = draft[:m] + [preds[m]]  # 至少得到1个token
```

**为什么无损**：最终输出由大模型的 argmax 决定，草稿模型只影响速度不影响正确性。

**真实 vLLM 的实现**：
- `EagleWorker`: 用小 Transformer 做草稿模型
- `MedusaWorker`: 多头并行预测
- `ngram`: 统计模型
- 都使用相同的 paged KV cache 做验证

### 7.3 服务化

> 对应代码：`minivllm/async_engine.py` + `minivllm/api_server.py`

**架构**：

```
HTTP 请求 ──→ FastAPI (异步)
                ↓
            AsyncLLMEngine
                ↓
         后台线程 (单线程)
                ↓
          LLMEngine.step()
                ↓
         Queue (每个请求一个)
                ↓
          HTTP 响应 (流式)
```

```python
# async_engine.py
class AsyncLLMEngine:
    def __init__(self, config, tokenizer, model):
        self._engine = LLMEngine(...)        # 同步引擎
        self._streams = {}                   # seq_id → Queue
        self._thread = Thread(target=self._run_loop)  # 后台线程
        self._thread.start()

    def _run_loop(self):
        """引擎主循环: 不断step"""
        while not self._stop:
            if self._engine.has_pending():
                outputs = self._engine.step()       # 执行一步
                for o in outputs:
                    self._streams[o.request_id].put(o)  # 推给对应请求
            else:
                time.sleep(0.001)  # 空闲时小睡

    async def stream(self, prompt, params):
        """异步流式生成"""
        q = queue.Queue()
        seq_id = self._engine.add_request(prompt, params)
        self._streams[seq_id] = q
        while True:
            o = await asyncio.to_thread(q.get)  # 异步等输出
            yield o
            if o.finished:
                break
```

**关键设计**：
- 引擎是**单线程**的（GPU 只有一块，不需要并发执行）
- HTTP 层是**异步**的（可以同时接收多个请求）
- 通过 Queue 解耦：HTTP 请求和引擎执行互不阻塞

**OpenAI 兼容 API**：

```python
# api_server.py
@app.post("/v1/completions")
async def completions(body: dict):
    params = _sampling_params_from_body(body)
    if body.get("stream"):
        # 流式: SSE
        return StreamingResponse(
            _completion_stream(engine, prompt, params),
            media_type="text/event-stream"
        )
    else:
        # 非流式: 等完整输出
        output = await engine.complete(prompt, params)
        return _completion_response(output, params)
```

这意味着你可以用任何 OpenAI 客户端（`openai` Python 库、curl、LangChain 等）直接调用 mini-vLLM 服务。

---

## 第 8 章 动手实验：改代码加深理解

### 实验 1：观察 block 分配

```python
# 在 engine.py 的 _execute() 开头加打印
def _execute(self, scheduled):
    print(f"  free_blocks={self.block_manager.num_free_blocks}/"
          f"{self.block_manager.num_total_blocks}")
    print(f"  prefill={len(scheduled.prefill_items)}, "
          f"decode={len(scheduled.decode_items)}")
    ...
```

然后跑 demo：
```bash
python -m minivllm demo --model artifacts/tinygpt
```

观察每个 step 有多少空闲块、多少序列在 prefill vs decode。

### 实验 2：调整 block_size

```python
# 把 block_size 从 16 改成 4
cfg = CacheConfig(block_size=4, num_gpu_blocks=256)
# 总KV容量一样 (4×256=1024 tokens vs 16×64=1024 tokens)
# 但小块 → 更细粒度分配 → 更少碎片 但 block_table 更长
```

对比生成结果是否一致（应该一致，只是性能不同）。

### 实验 3：关闭前缀缓存

```python
# 对比
cfg.enable_prefix_caching = True   # 开
cfg.enable_prefix_caching = False  # 关

# 用相同前缀的prompt跑100次，对比引擎step数
```

开前缀缓存时，相同前缀的后续请求应该跳过大部分 prefill step。

### 实验 4：触发抢占

```python
# 故意把 KV cache 设极小
cfg = CacheConfig(block_size=4, num_gpu_blocks=8)  # 只能放32个token
# 添加多个长prompt → 必然触发抢占
# 观察输出: "seq X was preempted" → "seq X re-prefilling"
```

### 实验 5：修改采样参数

```python
# 贪心解码 (temperature=0): 每次输出完全一样
SamplingParams(temperature=0)

# 高温度 (temperature=2.0): 输出更随机
SamplingParams(temperature=2.0)

# top-k=1: 只从概率最高的1个token中选（等同于贪心）
SamplingParams(top_k=1)

# top-p=0.9: 从累计概率达90%的token集合中选
SamplingParams(top_p=0.9)
```

### 实验 6：理解分块 prefill 的等价性

```python
# 同一个prompt，不同的token预算
e1 = make_engine(max_batched_tokens=256)  # 一步prefill完
e2 = make_engine(max_batched_tokens=6)   # 分很多步prefill

# 固定seed，对比输出
params = SamplingParams(seed=42, temperature=0.8, max_tokens=20)
out1 = e1.generate("vLLM is", params)
out2 = e2.generate("vLLM is", params)
assert out1.outputs[0].token_ids == out2.outputs[0].token_ids  # 应该一样
```

---

## 第 9 章 部署 DeepSeek / MoE 模型实战

### 9.1 从 mini-vLLM 到真实 vLLM

mini-vLLM 教会你的所有机制，在真实 vLLM 中完全对应：

| 你在 mini-vLLM 学到的 | 真实 vLLM 的对应 |
|---|---|
| `KVStore` 的 4D shape | 完全一样，只是分配在 CUDA memory |
| `BlockAllocator` 引用计数 | 完全一样 |
| `block_table` 逻辑→物理映射 | 完全一样 |
| `gather_kv` 从块收集 KV | 融合在 paged_attn CUDA kernel 中 |
| `Scheduler` 三阶段调度 | 完全一样，多了优先级和策略 |
| `PrefixCache` LRU | 完全一样，多了 hash-based 匹配 |
| `_preempt_one` recompute | 完全一样，还有 swap 模式 |
| `_chunk_for` 分块prefill | 完全一样 |
| `AsyncLLMEngine` | 完全一样 |

**你阅读 mini-vLLM 源码获得的知识，100% 可迁移到真实 vLLM。**

### 9.2 部署 DeepSeek-V3

```bash
# 1. 安装 vLLM
pip install vllm

# 2. 单卡部署（如果显存够大）
vllm serve deepseek-ai/DeepSeek-V3 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.9

# 3. 多卡部署（Tensor Parallel，模型太大一块GPU放不下）
vllm serve deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 8 \
  --max-model-len 32768

# 4. 用 OpenAI 客户端调用
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3",
    messages=[{"role": "user", "content": "解释PagedAttention"}]
)
```

### 9.3 DeepSeek-V3 的特殊性及引擎适配

#### MLA 对引擎的影响

```
标准模型 KV Cache per token:
  2 × n_layers × n_heads × head_dim

DeepSeek-V3 MLA KV Cache per token:
  2 × n_layers × d_c  (d_c = 512, 远小于 n_heads × head_dim)

例: DeepSeek-V3 (61层, 128头, 128维)
  标准: 2 × 61 × 128 × 128 = 2,002,944 字节/token (float16)
  MLA:  2 × 61 × 512    =    62,464 字节/token (float16)
  → KV Cache 减少 32 倍！
```

在 vLLM 中，MLA 的 KV Cache shape 不同，但 block manager 和 scheduler 的逻辑完全不变。

#### MoE 对引擎的影响

```
DeepSeek-V3 MoE:
  总参数: 671B
  每个 token 激活: 37B (256选8)
  → 计算量小，但权重总量大

引擎适配:
  - 权重加载: 要加载256组专家权重
  - 前向传播: router → 选专家 → 只算被选中的
  - 调度器: 完全不变
  - KV Cache: 不受MoE影响（MoE只替换FFN部分）
```

### 9.4 部署其他模型

```bash
# LLaMA
vllm serve meta-llama/Llama-3.1-8B-Instruct

# Qwen (带MoE的版本)
vllm serve Qwen/Qwen2.5-72B-Instruct --tensor-parallel-size 4

# Qwen MoE
vllm serve Qwen/Qwen2.5-MoE-72B --tensor-parallel-size 4

# Mistral
vllm serve mistralai/Mistral-7B-v0.3

# 量化模型
vllm serve TheBloke/Llama-2-13B-AWQ --quantization awq
vllm serve TheBloke/Llama-2-13B-GPTQ --quantization gptq
```

### 9.5 关键启动参数

这些参数你在 mini-vLLM 中都见过对应的：

| vLLM 参数 | 含义 | mini-vLLM 对应 |
|---|---|---|
| `--block-size` | KV Cache 块大小 | `CacheConfig.block_size` |
| `--gpu-memory-utilization` | GPU 显存使用比例 | `CacheConfig.num_gpu_blocks` |
| `--max-num-seqs` | 最大并发序列数 | `SchedulerConfig.max_num_seqs` |
| `--max-num-batched-tokens` | 每步 token 预算 | `SchedulerConfig.max_num_batched_tokens` |
| `--enable-chunked-prefill` | 分块 prefill | `SchedulerConfig.enable_chunked_prefill` |
| `--enable-prefix-caching` | 前缀缓存 | `EngineConfig.enable_prefix_caching` |
| `--tensor-parallel-size` | 张量并行度 | 无（mini版不支持分布式） |
| `--quantization` | 量化方案 | `EngineConfig.quantize` |
| `--max-model-len` | 最大上下文长度 | `EngineConfig.max_model_len` |

---

## 第 10 章 造你自己的推理框架

如果你想自己造一个推理引擎，mini-vLLM 就是你的骨架。以下是从零到可用的路线图。

### 10.1 技术栈选择

| 层次 | 选择 | 理由 |
|------|------|------|
| 语言 | Python | 生态最好 |
| 计算后端 | PyTorch (GPU) / NumPy (CPU) | PyTorch 有 CUDA 支持 |
| 模型定义 | 直接写或用 HuggingFace | HF 模型可以直接加载 |
| Web 框架 | FastAPI | 异步、轻量、OpenAI 兼容 |
| KV Cache | 自己实现 PagedAttention | 这是核心壁垒 |

### 10.2 实现路线图

#### 阶段 1: 单请求推理（无 KV Cache）
```python
# 最简单的: 不缓存KV, 每次重新计算
def generate_naive(model, prompt, max_tokens):
    tokens = tokenizer.encode(prompt)
    for _ in range(max_tokens):
        logits = model.forward(tokens)  # 每次处理整个序列
        next_token = sample(logits[-1])
        tokens.append(next_token)
        if next_token == eos:
            break
    return tokens
```

#### 阶段 2: 加 KV Cache（连续分配）
```python
# 简单 KV Cache: 每个序列分配一段连续内存
class SimpleKVCache:
    def __init__(self, max_len, n_layers, n_heads, head_dim):
        self.k = np.zeros((max_len, n_layers, n_heads, head_dim))
        self.v = np.zeros((max_len, n_layers, n_heads, head_dim))
        self.len = 0

    def append(self, k_new, v_new):
        self.k[self.len] = k_new  # 追加到末尾
        self.v[self.len] = v_new
        self.len += 1
```

#### 阶段 3: 加 PagedAttention（分块分配）
- 这是 mini-vLLM 的 `kv_cache.py`
- 加 `BlockAllocator`, `block_table`, 引用计数

#### 阶段 4: 加调度器（连续批处理）
- 这是 mini-vLLM 的 `scheduler.py`
- 加 waiting/running 队列, 三阶段调度

#### 阶段 5: 加进阶机制
- 前缀缓存 (PrefixCache)
- 分块 prefill (_chunk_for)
- 抢占 (_preempt_one)

#### 阶段 6: 加服务化
- AsyncEngine (后台线程)
- FastAPI OpenAI 兼容 API

#### 阶段 7: 替换为真实模型
- 把 TinyGPT 换成 HuggingFace 模型
- 处理权重格式转换
- 支持 RoPE, RMSNorm, SwiGLU, GQA

#### 阶段 8: GPU 加速
- NumPy → PyTorch
- paged_attention → Triton kernel 或 vLLM 的 CUDA kernel

#### 阶段 9: 分布式
- Tensor Parallel (模型按维度切分到多卡)
- Pipeline Parallel (模型按层切分)

#### 阶段 10: 高级特性
- 投机解码
- 量化 (GPTQ/AWQ/FP8)
- 多模态 (图片+文本)

### 10.3 核心代码结构建议

```
my_inference_engine/
├── config.py          # 配置
├── tokenizer.py       # 用 HuggingFace tokenizer
├── models/
│   ├── base.py        # 模型基类
│   ├── llama.py       # LLaMA 架构
│   ├── deepseek.py     # DeepSeek (MLA + MoE)
│   └── qwen.py        # Qwen 架构
├── attention/
│   ├── paged_attn.py  # 分页注意力 (PyTorch)
│   └── flash_attn.py  # FlashAttention (可选)
├── kv_cache.py        # BlockManager (和mini-vLLM一样)
├── scheduler.py       # 连续批处理 (和mini-vLLM一样)
├── sampler.py         # 采样 (和mini-vLLM一样)
├── engine.py          # LLMEngine (和mini-vLLM一样)
├── async_engine.py    # AsyncEngine (和mini-vLLM一样)
├── api_server.py      # OpenAI API (和mini-vLLM一样)
└── distributed/
    ├── tensor_parallel.py  # 张量并行
    └── pipeline_parallel.py # 流水线并行
```

**你会发现**：`kv_cache.py`, `scheduler.py`, `sampler.py`, `engine.py`, `async_engine.py`, `api_server.py` 这 6 个文件可以几乎原封不动地从 mini-vLLM 复用——因为它们和模型架构无关，只管引擎机制。

---

## 附录 A：mini-vLLM 全模块速查表

| 文件 | 行数 | 核心功能 | 对应真实 vLLM |
|------|------|---------|-------------|
| `config.py` | 178 | EngineConfig/ModelConfig/CacheConfig/SchedulerConfig/SamplingParams | `vllm/engine/arg_utils.py` |
| `tokenizer.py` | 92 | 字符级 tokenizer (模仿 HF 接口) | HuggingFace tokenizers |
| `model.py` | 156 | TinyGPT: 嵌入→N层(LN+Attn+残差+LN+MLP+残差)→LN→lm_head | `vllm/model_executor/models/` |
| `attention.py` | 113 | gather_kv + paged_attention_batch (因果注意力) | `vllm/attention/ops/paged_attn.py` |
| `kv_cache.py` | 269 | KVStore + BlockAllocator + PrefixCache + BlockManager(COW) | `vllm/core/block_manager.py` |
| `scheduler.py` | 207 | 三阶段调度 + 分块prefill + recompute抢占 | `vllm/core/scheduler.py` |
| `sampler.py` | 92 | 温度→top-k→top-p→softmax→采样 | `vllm/model_executor/layers/sampler.py` |
| `sequence.py` | 118 | Sequence状态机 + CompletionOutput + RequestOutput | `vllm/sequence.py` |
| `engine.py` | 195 | LLMEngine: add_request/step/execute/generate/generate_stream | `vllm/engine/llm_engine.py` |
| `async_engine.py` | 80 | 后台线程 + Queue + 异步迭代器 | `vllm/engine/async_llm_engine.py` |
| `api_server.py` | 221 | FastAPI OpenAI兼容API + SSE流式 | `vllm/entrypoints/openai/` |
| `cli.py` | 202 | serve/chat/demo/spec/quant 子命令 | `vllm/entrypoints/cli/` |
| `quantize.py` | 96 | int8对称量化 + 反量化 | GPTQ/AWQ/FP8 |
| `speculative.py` | 146 | bigram草稿 + 验证 + 接受/reject | `vllm/spec_decode/` |
| `training.py` | 260 | 纯NumPy反向传播 (教育用途) | (vLLM不训练模型) |
| `checkpoint.py` | 76 | save/load model+config+tokenizer | `transformers` 的 save_pretrained |
| `data.py` | 35 | 训练语料 (关于vLLM的短文本) | (vLLM不需要) |

---

## 附录 B：术语表

| 术语 | 解释 |
|------|------|
| **KV Cache** | 缓存历史token的Key和Value向量，避免重复计算 |
| **PagedAttention** | 把KV Cache存在定长物理块中，通过block table映射 |
| **Block Table** | 序列的"页表"：逻辑位置→物理块的映射 |
| **Block Allocator** | 管理物理块的分配/回收/引用计数 |
| **Reference Count** | 一个物理块被多少对象引用，归零才能回收 |
| **COW (Copy-on-Write)** | 共享块要写入时先复制，避免污染其他使用者 |
| **Prefill** | 处理整个prompt，计算并缓存所有token的KV |
| **Decode** | 每次处理1个token，利用缓存的KV做注意力 |
| **Continuous Batching** | 每步混合prefill和decode，完成一个立刻接纳新的 |
| **Chunked Prefill** | 长 prompt 分多步 prefill |
| **Preemption** | KV cache满时驱逐低优先级序列 |
| **RECOMPUTE** | 抢占模式: 丢弃KV，之后重算 |
| **SWAP** | 抢占模式: KV从GPU搬到CPU |
| **Prefix Caching** | 缓存公共prompt前缀的KV块，新请求可复用 |
| **GQA** | 分组查询注意力: 多个Q头共享KV头，减少KV cache |
| **MLA** | 多头潜在注意力: 低秩压缩KV，DeepSeek创新 |
| **MoE** | 混合专家: 每个token只激活部分专家网络 |
| **RoPE** | 旋转位置编码: 对Q/K做旋转编码位置信息 |
| **RMSNorm** | 均方根归一化: LayerNorm的简化版 |
| **SwiGLU** | 带门控的MLP激活函数 |
| **Speculative Decoding** | 投机解码: 小模型猜+大模型验证 |
| **Tensor Parallel** | 张量并行: 把模型权重按维度切分到多卡 |
| **TPS** | Tokens Per Second: 每秒生成token数 |
| **Throughput** | 吞吐量: 单位时间处理的请求/token总数 |
| **Latency** | 延迟: 单个请求从提交到完成的耗时 |

---

## 学习建议

1. **先跑通**：安装 mini-vLLM，训练模型，跑测试和 demo
2. **读代码**：按本文章节顺序，对照源码逐个模块理解
3. **改代码**：做第 8 章的实验，加打印，调参数
4. **读真实 vLLM**：对照速查表，看真实 vLLM 对应文件
5. **部署模型**：用真实 vLLM 部署一个 LLaMA 或 DeepSeek
6. **造框架**：按第 10 章的路线图，从阶段 1 开始逐步实现

mini-vLLM 的全部代码不到 2000 行，但它包含了 vLLM 的完整灵魂。理解了它，你就从一个 LLM 推理的门外汉，变成了能看懂源码、能改架构、能造轮子的专家。
