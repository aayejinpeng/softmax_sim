# Plan 0: Multi-Lane RVV ISA 规范

## 目标

定义 RVV Multi-Lane TLP Extension 的完整 ISA 规范，包括编码、CSR、指令语义。

## 交付物

文件: `softmax_sim/isa_spec.md`（ISA 规范文档）

## ISA 规范

### 1. vtype 扩展

```
vtype CSR:
  bit [9:8]  vlane    R/W, 复位=0
    00 = 1 lane（标准 RVV，向后兼容）
    01 = 2 lanes
    10 = 4 lanes
    11 = 8 lanes
  bit [7:0]  现有字段（vsew, vlmul, vta, vma），不变
```

### 2. 新增 CSR: vlane0-vlane3

| CSR 地址 | 名称 | 位宽 | 含义 |
|----------|------|------|------|
| 0x7C0 | vlane0 | [31:0] | Lane stride 0 (bytes) |
| 0x7C1 | vlane1 | [31:0] | Lane stride 1 (bytes) |
| 0x7C2 | vlane2 | [31:0] | Lane stride 2 (bytes) |
| 0x7C3 | vlane3 | [31:0] | Lane stride 3 (bytes) |

- stride=0 时所有 lane 访问同一地址（共享数据）
- 复位值 = 0

### 3. Load/Store 编码

利用 unit-stride load 的 lumop[4:0] 预留值：

```
lumop=00000: vle (标准 load，vlane0) — 向后兼容
lumop=00001: vle.l1 (vlane1)
lumop=00010: vle.l2 (vlane2)
lumop=00011: vle.l3 (vlane3)

store (vse) 的 sumop 同理：
sumop=00000: vse (标准 store，vlane0)
sumop=00001: vse.l1 (vlane1)
sumop=00010: vse.l2 (vlane2)
sumop=00011: vse.l3 (vlane3)
```

### 4. 地址计算语义

```
effective_addr = base(rs1) + lane_id × vlane_ctx_stride[31:0]

其中:
  base(rs1) = 标量寄存器 rs1 的值（所有 lane 共享）
  lane_id = 当前 lane 编号 (0 ~ vlane-1)
  vlane_ctx_stride = CSR vlane[ctx] 的值，ctx 由 lumop/sumop 低 2 位选择
```

### 5. 向量寄存器映射

```
物理寄存器编号 = logical_reg + lane_id × 32

例: 4 lanes 时
  Lane 0: v0-v31 → 物理 0-31
  Lane 1: v0-v31 → 物理 32-63
  Lane 2: v0-v31 → 物理 64-95
  Lane 3: v0-v31 → 物理 96-127
```

### 6. 编程模型

```asm
# 设置
csrw vlane0, t0           # input stride
csrw vlane1, t1           # output stride
csrw vlane2, zero          # shared (weight/gamma)
vsetvli a0, a1, e32, m4, lanes=4

# kernel body — 标准 RVV 代码
vle32.v    v4, (a0)        # per-lane input
vle32.l2.v v8, (a2)        # shared weight
vfmul.vv   v4, v4, v8      # 各 lane 各自计算
vse32.l1.v v4, (a1)        # per-lane output
```

### 7. 约束

- vlane > 1 时，物理向量寄存器数量 = 32 × vlane，需要实现支持
- 非 load/store 的向量指令（算术、reduce 等）不受 stride 影响，仅按 lane_id 映射寄存器
- vsetvli 设置的 vl 对所有 lane 相同
- vlane > 1 时 vlmul 受限：确保 32 × vlane × lmul ≤ 物理寄存器总数

## 验证

- 文档完整性检查
- 编码表无冲突
- 向后兼容性：vlane=0 时所有行为与标准 RVV 一致
