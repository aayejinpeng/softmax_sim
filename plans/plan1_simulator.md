# Plan 1: 多 Kernel ISA 模拟器

## 目标

在现有 softmax_simulator.py 基础上，增加 rmsnorm、rope、silu 等 kernel 的指令流生成和模拟，支持 lane stride 地址计算，产出多 kernel 多配置的性能数据。

## 改动文件

- `softmax_sim/softmax_simulator.py` — 主要改动
- `softmax_sim/plans/benchmark_results.md` — 性能数据输出

## 具体步骤

### Step 1: ProcessorConfig 增加 lane_strides

在 `ProcessorConfig` 中加入:

```python
lane_strides: Tuple[int, int, int, int] = (0, 0, 0, 0)  # vlane0-vlane3 stride (bytes)
```

### Step 2: Load/Store 指令增加 vlane_ctx

在 `LoadInstruction` 和 `StoreInstruction` 中加入 `vlane_ctx: int = 0` 参数。
`MicroOp` 增加 `vlane_ctx: int = 0` 字段，从 Instruction 继承。

### Step 3: 地址计算修改

在 `_split_memory_instruction` 中，μop 的地址需要考虑 lane stride：
- 当前按 cache_bandwidth 拆分 μop，不变
- 每个 μop 的"逻辑地址" = base + chunk_offset + context_id × lane_strides[vlane_ctx]
- 这只影响依赖分析和发射时机，实际 cycle 不变（访存延迟是建模参数）

### Step 4: 新增 rmsnorm 指令流

参考 `cute_quant.h` 中 `cute_rmsnorm` 的实现：

```python
def create_rmsnorm_instruction_stream(reg_width, num_rows, num_contexts=1):
    """
    rmsnorm 每行的操作序列:
    1. LOAD input (per-row)
    2. FMA square (vfmul.vv) — 多个 chunk
    3. REDUCE sum (vfredusum)
    4. FMA scale by rms (vfmul.vf) — 多个 chunk
    5. LOAD weight (shared, stride=0)
    6. FMA multiply weight (vfmul.vv) — 多个 chunk
    7. STORE output (per-row)
    """
```

依赖链:
```
LOAD(input) → FMA(square) → REDUCE(sum) → FMA(scale) → LOAD(weight) → FMA(mul_weight) → STORE
```

stride 配置:
- vlane0: input row stride (row_bytes)
- vlane1: output row stride (row_bytes)
- vlane2: 0 (weight is shared)

### Step 5: 新增 silu 指令流

参考 `cute_elementwise.h` 中 `cute_silu_out_tile`：

```python
def create_silu_instruction_stream(reg_width, num_rows, num_contexts=1):
    """
    silu 每行的操作序列:
    1. LOAD input
    2. FMA negate (vfneg)
    3. EXP2 (vec_exp, 多个内部步骤)
    4. FMA add 1.0 (vfadd)
    5. FMA div (vfdiv) or recip + mul (fast version)
    6. STORE output
    """
```

依赖链:
```
LOAD → FMA(neg) → EXP2 → FMA(add1) → FMA(div) → STORE
```

### Step 6: 新增 rope 指令流（简化版）

参考 `cute_sequence.h` 中 `cute_rope_f16_tile`：

```python
def create_rope_instruction_stream(reg_width, num_rows, num_contexts=1):
    """
    rope 每行的操作序列（简化，忽略 sin/cos 的复杂多项式）:
    1. LOAD theta (shared)
    2. FMA multiply position (angle = theta * pos)
    3. LOAD input real/imag (strided load, 2*sizeof(float))
    4. FMA real_out = real*cos - imag*sin
    5. FMA imag_out = real*sin + imag*cos
    6. STORE output
    """
```

依赖链:
```
LOAD(theta) → FMA(angle) → LOAD(input) → FMA(real) → FMA(imag) → STORE
```

### Step 7: CLI 参数更新

```python
parser.add_argument('--kernel', choices=['softmax', 'rmsnorm', 'silu', 'rope'], default='softmax')
parser.add_argument('--lane-stride-0', type=int, default=0)
parser.add_argument('--lane-stride-1', type=int, default=0)
parser.add_argument('--lane-stride-2', type=int, default=0)
parser.add_argument('--lane-stride-3', type=int, default=0)
```

### Step 8: main() 中根据 kernel 选择生成函数

```python
if args.kernel == 'softmax':
    instructions = create_softmax_instruction_stream(...)
elif args.kernel == 'rmsnorm':
    instructions = create_rmsnorm_instruction_stream(...)
elif args.kernel == 'silu':
    instructions = create_silu_instruction_stream(...)
elif args.kernel == 'rope':
    instructions = create_rope_instruction_stream(...)
```

## 性能数据目标

对每个 kernel 生成：

```
kernel       | 1 lane | 2 lanes | 4 lanes | issue_width=1 | issue_width=2
softmax      | xxx cy | xxx cy  | xxx cy  | ...
rmsnorm      | xxx cy | xxx cy  | xxx cy  | ...
silu         | xxx cy | xxx cy  | xxx cy  | ...
rope         | xxx cy | xxx cy  | xxx cy  | ...
```

加上各 kernel 的：
- 发射槽利用率
- 各执行单元利用率
- 多 lane 性能提升百分比

## 验证

- 单 lane 结果合理性检查（依赖链长度 × 延迟 ≥ 总 cycle）
- 多 lane 收益符合预期（长链 kernel 收益大，短链收益小）
- stride=0 时共享数据行为正确
