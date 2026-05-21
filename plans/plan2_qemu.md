# Plan 2: QEMU Multi-Lane 功能模拟器

## 目标

在现有 T-Head QEMU (`/root/opencute/CUTE/cute-sdk/cuteqemu/`) 上实现 Multi-Lane TLP Extension 的功能级模拟，能运行带 vlane CSR 和 lumop 编码的 RVV 程序并验证正确性。

## 基线

- QEMU 已有完整的 RVV 支持（T-Head 版本）
- VRF: `uint64_t vreg[32 * RV_VLEN_MAX / 64]`（32 个向量寄存器，每个 VLEN bit）
- Load/Store: `vext_ldst_us()` / `vext_ldst_stride()` 计算地址
- CSR: `csr.c` 中的 `csr_ops[]` 表管理所有 CSR

## 改动文件

所有文件在 `/root/opencute/CUTE/cute-sdk/cuteqemu/target/riscv/` 下。

### Step 1: CPU 状态扩展 — `cpu.h`

在 `CPURISCVState` 中新增字段：

```c
// Multi-lane state
target_ulong vlane_stride[4];  // vlane0-vlane3 CSR values
```

### Step 2: CSR 注册 — `csr.c`

在 `csr_ops[]` 表中注册 4 个新 CSR：

```c
// vlane0-vlane3: 0x7C0-0x7C3
[CSR_VLANE0] = { "vlane0", vsr, read_vlane, write_vlane, NULL, NULL,
                  (target_ulong)0 },
[CSR_VLANE1] = { "vlane1", vsr, read_vlane, write_vlane, NULL, NULL,
                  (target_ulong)1 },
...
```

新增 helper：
```c
static RISCVException read_vlane(CPURISCVState *env, int csrno,
                                  target_ulong *val) {
    *val = env->vlane_stride[csrno - CSR_VLANE0];
    return RISCV_EXCP_NONE;
}

static RISCVException write_vlane(CPURISCVState *env, int csrno,
                                   target_ulong val) {
    env->vlane_stride[csrno - CSR_VLANE0] = val;
    return RISCV_EXCP_NONE;
}
```

### Step 3: vtype 解码支持 vlane — `cpu.h` / `vector_helper.c`

新增字段提取宏：

```c
#define VTYPE_VLANE  FIELD_EX64(env->vtype, VTYPE, VLANE)
// 或直接: (env->vtype >> 8) & 0x3
```

在 `vsetvli` helper 中保存 vlane 到 env 状态。

### Step 4: 向量指令广播执行 — `vector_helper.c`

核心改动：当 vlane > 1 时，向量指令循环执行 N 次（每个 lane 一次）。

在所有向量 helper 函数的入口处，包裹一个 lane 循环：

```c
// 伪代码 - 在 helper 入口
int vlane = (env->vtype >> 8) & 0x3;  // 提取 vlane
if (vlane == 0) vlane = 1;             // vlane=0 means 1 lane

for (int lane = 0; lane < vlane; lane++) {
    // 执行向量操作，使用 lane 偏移访问 VRF
    // VRF[lane][reg] = vreg[(lane * 32 + reg) * VLEN/64]
}
```

实际实现方式：在 `vext_vcommon()` 或每个 helper 的循环中，调整 VRF 偏移：

```c
// 原来：直接访问 vd
// 现在：加上 lane 偏移
uint64_t *vd_lane = vreg + (vd_off + lane * 32 * VLEN/64);
```

### Step 5: Load/Store 地址计算 — `vector_helper.c`

修改 `vext_ldst_us()` 函数：

```c
// 原来的地址计算：
// addr = base + (i << log2_esz)

// 新增 lane stride：
// 从指令 descriptor 中提取 vlane_ctx (lumop 低 2 位)
// addr = base + (i << log2_esz) + lane_id * env->vlane_stride[vlane_ctx]
```

需要从翻译阶段传递 `vlane_ctx` 信息到 helper。在 `TCGv_i32 desc` 中预留 bit 传递。

### Step 6: 指令译码 — `insn_trans/trans_rvv.c.inc`

在 unit-stride load/store 的译码函数中：

1. 检测 `lumop` 值：
   - `lumop == 0b00001/00010/00011` → 这是 multi-lane load
   - 提取 `lumop & 0x3` 作为 `vlane_ctx`

2. 将 `vlane_ctx` 编码到 helper 的 `desc` 参数中（使用预留 bit）

3. 生成 TCG 代码调用 helper 时传入 desc

### Step 7: VRF 布局扩展

当前 VRF 大小：`vreg[32 * RV_VLEN_MAX / 64]`（32 个寄存器）

Multi-lane 需要：`vreg[32 * max_lanes * RV_VLEN_MAX / 64]`

两种实现方式：
- **方案 A（简单）**：直接扩大 VRF 数组，`max_lanes=8` → 数组扩大 8 倍
- **方案 B（动态）**：运行时根据 vlane 分配，但 QEMU 中静态分配更简单

建议用方案 A，改 `cpu.h`：
```c
#define RV_MAX_LANES 8
uint64_t vreg[32 * RV_MAX_LANES * RV_VLEN_MAX / 64];
```

### Step 8: 测试程序

编写测试程序（用 riscv64-linux-gnu-gcc 编译）：

```c
// test_vlane.c
#include <riscv_vector.h>

int main() {
    // 设置 stride
    asm volatile("csrw 0x7C0, %0" :: "r"(256));  // vlane0 = 256 (input stride)
    asm volatile("csrw 0x7C1, %0" :: "r"(256));  // vlane1 = 256 (output stride)
    asm volatile("csrw 0x7C2, %0" :: "r"(0));     // vlane2 = 0 (shared)

    // 用 inline asm 设置 vsetvli with lanes=2
    // ... 执行 load + compute + store ...

    // 验证结果
    return 0;
}
```

## 实施顺序

```
Step 1: cpu.h 状态扩展 + VRF 扩大
  ↓
Step 2: csr.c 注册 vlane CSR
  ↓
Step 3: vtype 解码 vlane
  ↓
Step 4: vector_helper.c 地址计算（load/store）
  ↓
Step 5: vector_helper.c VRF lane 偏移
  ↓
Step 6: trans_rvv.c.inc lumop 译码
  ↓
Step 7: 测试程序编写和验证
```

## 验证

1. CSR 读写：`csrw vlane0, t0` + `csrr t1, vlane0` → t0 == t1
2. 单 lane 向后兼容：vlane=0 时所有现有 RVV 测试通过
3. 多 lane load 地址：4 lane 各读到不同数据
4. stride=0 共享：所有 lane 读到同一地址的数据
5. 简单 kernel（如 memcpy）在多 lane 模式下结果正确
