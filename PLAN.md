# RVV Multi-Lane TLP Extension Plan

## Context

在 RISC-V 向量处理器上，AI workload（softmax、rmsnorm、rope 等）存在大量独立的长依赖链计算任务。当前单 lane 执行时，发射槽利用率仅 31%，大量 cycle 浪费在等待依赖上。

本方案通过在 RVV ISA 中新增 **Multi-Lane** 机制，让一条向量指令自动广播到多个独立 lane（各拥有独立寄存器堆和记分牌），共享 decode 和执行单元，用最小硬件代价榨取 TLP。模拟显示 2 lane 即可提升 41% 性能。

---

## 1. ISA 规范（体系结构）

### 1.1 vtype 扩展

```
vtype CSR 扩展:
  [9:8]  vlane   00=1lane, 01=2lanes, 10=4lanes, 11=8lanes
  [7:0]  现有字段不变 (vsew, vlmul, vta, vma)
```

- vlane=0 时行为与标准 RVV 完全一致（向后兼容）
- vlane>0 时，后续向量指令广播到 vlane 个 lane

### 1.2 vlane stride CSR

新增 4 个 CSR：

```
CSR 0x??0: vlane0  [31:0] lane_stride (bytes)  ← stride=0 表示共享访问
CSR 0x??1: vlane1  [31:0] lane_stride
CSR 0x??2: vlane2  [31:0] lane_stride
CSR 0x??3: vlane3  [31:0] lane_stride
```

- load/store 指令通过 2-bit vlane ctx 选择器关联对应 CSR
- 实际地址 = rs1(base) + lane_id × vlaneN_stride
- stride=0 时所有 lane 访问同一地址（共享数据，如 gamma/beta）

### 1.3 Load/Store 指令编码

利用 unit-stride load/store 的 `lumop` 预留位：

```
vle16.v    v4, (a0)    # lumop=00000 → vlane0（标准编码，向后兼容）
vle16.l1.v v4, (a0)    # lumop=00001 → vlane1
vle16.l2.v v4, (a0)    # lumop=00010 → vlane2
vle16.l3.v v4, (a0)    # lumop=00011 → vlane3
```

store 指令同理（vs 的 sumop 对应位）。

### 1.4 编程模型

```asm
# 1. 设置 stride（kernel 入口前，一次性）
csrw   vlane0, t_in_stride     # input stride
csrw   vlane1, t_out_stride    # output stride
csrw   vlane2, zero            # shared (gamma/beta)

# 2. 开启 multi-lane
vsetvli a0, a1, e16, m1, lanes=4

# 3. kernel body — 标准 RVV 代码，load/store 自动加 stride
vle16.v    v4, (a0)             # per-lane input (vlane0)
vle16.l2.v v8, (a2)             # shared gamma (vlane2, stride=0)
vfmul.vv   v4, v4, v8           # 各 lane 各自计算
vse16.l1.v v4, (a1)             # per-lane output (vlane1)
```

---

## 2. 微结构设计

### 2.1 整体结构

```
                    ┌───────────┐
                    │  Decoder  │  读 vtype.vlane，广播指令
                    └─────┬─────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │  Lane 0  │   │  Lane 1  │   │  Lane N  │
    │ VRF[32]  │   │ VRF[32]  │   │ VRF[32]  │  每lane独立
    │ Scoreboard│  │ Scoreboard│  │ Scoreboard│
    │ Issue Q  │   │ Issue Q  │   │ Issue Q  │
    └────┬─────┘   └────┬─────┘   └────┬─────┘
         │               │               │
         └───────────────┼───────────────┘
                         ▼
                  ┌──────────────┐
                  │   Arbiter    │  round-robin 仲裁
                  └──────┬───────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │ Load/    │   │ FMA/     │   │ Reduce   │  共享执行单元
   │ Store    │   │ EXP2     │   │ Unit     │
   │ Unit     │   │ Unit     │   │          │
   └──────────┘   └──────────┘   └──────────┘
```

### 2.2 地址计算

load/store 单元内新增：

```
effective_addr = rs1 + lane_id[2:0] × vlane_ctx_stride[31:0]
```

- lane_id 来自当前 lane 的编号
- vlane_ctx_stride 来自 load/store 指令编码的 2-bit ctx 选择对应的 CSR
- 硬件：一个 32-bit × 3-bit 乘法器 + 64-bit 加法器

### 2.3 面积开销估算

```
新增项                    面积（相对向量寄存器堆）
─────────────────────────────────────────────
N-1 套向量寄存器堆         (N-1) × 32 × VLEN bit
N-1 套记分牌               (N-1) × 少量 SRAM
4 个 stride CSR            4 × 32 bit = 128 bit
地址计算加法器/乘法器       可忽略
Decoder 广播逻辑            可忽略
Arbiter                    可忽略
```

以 4 lane、VLEN=2048 为例：
- 额外 3 套向量寄存器堆 = 3 × 32 × 256B = 24KB
- 不需要复制执行单元、标量流水线、cache

---

## 3. ISA 模拟器（增强 softmax_sim）

### 3.1 目标

在现有 softmax_simulator.py 基础上，增加 ISA 级建模，支持多 kernel 类型的微架构性能探索。

### 3.2 改动清单

**文件**: `/root/opencute/opencute_github/softmax_sim/softmax_simulator.py`

1. **ProcessorConfig 新增字段**:
   - `lane_strides: List[int]` — 4 个 stride 值（对应 vlane0-vlane3）

2. **MicroOp 新增字段**:
   - `vlane_ctx: int` — 该 μop 使用哪个 vlane CSR（0-3）

3. **地址计算修改**:
   - `_split_memory_instruction` 中，μop 的地址不再仅由 cache_bandwidth 拆分
   - 增加 lane 偏移：每个 context 的 load/store 地址 = base + context_id × lane_strides[vlane_ctx]

4. **新增 kernel 生成函数**:
   - `create_rmsnorm_instruction_stream()` — rmsnorm 指令流
   - `create_rope_instruction_stream()` — rope 指令流
   - `create_silu_instruction_stream()` — silu 指令流
   - 统一接口，接受 `num_contexts`、`lane_strides` 参数

5. **Load/Store 指令增加 vlane ctx 标注**:
   - LoadInstruction/StoreInstruction 增加 `vlane_ctx` 参数
   - 对应 vlane0-vlane3 的 stride

6. **CLI 参数**:
   - `--kernel {softmax,rmsnorm,rope,silu}` — 选择 kernel
   - `--lane-stride-0 N` ... `--lane-stride-3 N` — 设置各 vlane ctx 的 stride

### 3.3 验证

- 各 kernel 单 lane 结果与已有 softmax 结果一致
- 多 lane 性能提升数据（cycle、利用率）
- stride=0 时共享数据行为正确

---

## 4. QEMU 功能模拟器

### 4.1 基线

基于 `/root/opencute/CUTE/cute-sdk/cuteqemu/`（T-Head 版本），已有 RVV 支持。

### 4.2 改动清单

**目标文件**: `target/riscv/` 下的相关文件

1. **CSR 注册** (`csr.c`):
   - 注册 vlane0-vlane3 四个自定义 CSR
   - 读写函数：直接访问 CPU 状态中的对应字段

2. **CPU 状态扩展** (`cpu.h`):
   ```c
   // 在 CPURISCVState 中新增:
   uint64_t vlane_stride[4];  // vlane0-vlane3
   ```

3. **vtype 处理** (`vector_helper.c` 或 `translate.c`):
   - 解析 vtype[9:8] 为 vlane 数
   - 向量指令执行时检查 vlane，若 >1 则循环执行 N 次

4. **Load/Store 地址计算** (`vector_helper.c`):
   - 解析 lumop[1:0] 获取 vlane ctx (0-3)
   - 地址计算加入 lane 偏移：
     ```c
     addr = base + lane_id * env->vlane_stride[vlane_ctx];
     ```
   - 每个 lane 独立处理自己的向量寄存器组

5. **向量寄存器堆多 lane** (`vector_helper.c`):
   - 当 vlane > 1 时，使用 `vreg[lane_id * 32 + reg_id]` 映射
   - 逻辑寄存器到物理寄存器的映射

6. **指令译码** (`translate.c`):
   - 识别 lumop=00001/00010/00011 作为 vlane ctx 选择
   - 传递 ctx 信息到 helper 函数

### 4.3 测试

- 编写简单 RVV 汇编程序，使用 vlane CSR + vsetvli lanes
- 验证地址计算正确性（各 lane 读到不同数据）
- 验证 stride=0 时共享访问行为
- 与 softmax_sim 的性能数据交叉验证

---

## 5. 编译器支持（LLVM/Clang）

### 5.1 目标

在 LLVM 的 RVV 后端中支持 `#pragma simt`，自动生成 vlane CSR 配置和带 vlane ctx 的 load/store。

### 5.2 实现步骤

1. **Pragma 解析** (`clang/Lex/` + `clang/Parse/`):
   ```c
   #pragma simt(lanes=N, stride_in=expr, stride_out=expr, stride_shared=0)
   ```
   - 解析 pragma 参数到 AST 节点
   - lanes、stride 参数绑定到变量

2. **AST -> LLVM IR** (`clang/CodeGen/`):
   - 在 pragma 作用域的 kernel 函数入口前插入：
     - `csrw vlane0, stride_in`
     - `csrw vlane1, stride_out`
     - `csrw vlane2, 0`（shared）
   - 修改 vsetvli intrinsic 调用，加入 lanes 参数

3. **Load/Store 指令选择** (`llvm/lib/Target/RISCV/`):
   - 分析 kernel 内的 load/store 对应哪个指针参数
   - 根据 pragma 的 stride 映射关系，选择对应的 vlane ctx
   - 生成 `vle16.l0.v`、`vse16.l1.v` 等（lumop 编码）

4. **vtype 编码** (`llvm/lib/Target/RISCV/RISCVISelLowering.cpp`):
   - vsetvli 的 vtype immediate 加入 vlane[9:8] 编码

### 5.3 第一版简化方案

先不写 LLVM pass，用 **intrinsic + 汇编宏** 包装：

```c
// cute_simt.h — 第一版
#define SIMT_BEGIN(lanes, sin, sout) \
    asm volatile("csrw vlane0, %0" :: "r"(sin)); \
    asm volatile("csrw vlane1, %0" :: "r"(sout)); \
    asm volatile("csrw vlane2, zero");

#define SIMT_VSETVL(rd, rs1, eew, lmul, lanes) \
    asm volatile("vsetvli %0, %1, " #eew ", " #lmul ", lanes=" #lanes \
                 : "=r"(rd) : "r"(rs1));

// load/store 用 inline asm wrapper
#define VL_F16_LANE0(vd, rs1) \
    asm volatile("vle16.v %0, (%1)" : "=vr"(vd) : "r"(rs1));
#define VL_F16_LANE2(vd, rs1) \
    asm volatile("vle16.l2.v %0, (%1)" : "=vr"(vd) : "r"(rs1));
```

### 5.4 验证

- 编译生成 .s 文件，检查 CSR 写入和 lumop 编码
- 在 QEMU 上跑编译出的程序验证功能
- 对比 softmax_sim 的性能数据

---

## 6. 实施顺序

```
阶段 0: ISA 规范文档
  ↓
阶段 1: softmax_sim 增强（微架构性能验证）
  ├── 新增 rmsnorm/rope/silu kernel 生成
  ├── lane stride 地址计算
  └── 多 kernel 性能对比报告
  ↓
阶段 2: QEMU 功能模拟器
  ├── CSR 注册
  ├── vtype vlane 解析
  ├── 多 lane 寄存器映射
  └── load/store 地址计算
  ↓
阶段 3: 编译器支持
  ├── 第一版：intrinsic + 汇编宏包装
  ├── 在 QEMU 上验证
  └── 第二版：LLVM pragma pass（按需）
  ↓
阶段 4: 端到端验证
  ├── CUTE SDK kernel 用 pragma 改写
  ├── 编译 → QEMU 运行 → 结果验证
  └── 性能数据与 softmax_sim 交叉验证
```

---

## 7. 关键文件路径

```
模拟器:
  /root/opencute/opencute_github/softmax_sim/softmax_simulator.py

QEMU:
  /root/opencute/CUTE/cute-sdk/cuteqemu/target/riscv/
  ├── csr.c                    ← CSR 注册
  ├── cpu.h                    ← CPU 状态扩展
  ├── vector_helper.c          ← 向量指令执行逻辑
  ├── vector_internals.h       ← 向量内部接口
  └── translate.c              ← 指令译码

CUTE SDK kernels:
  /root/opencute/CUTE/cute-sdk/cutelib/primitive/include/
  ├── cute_sequence.h          ← softmax, rope
  ├── cute_quant.h             ← rmsnorm, smoothquant
  ├── cute_elementwise.h       ← silu, hadamard
  ├── cute_vec_math.h          ← exp, sin/cos, sqrt
  ├── cute_convert.h           ← dequant, f16/f32 转换
  └── cute_vector_fusion.h     ← 融合算子

编译器:
  LLVM/Clang (系统安装)
  riscv64-linux-gnu-gcc (系统安装)
```
