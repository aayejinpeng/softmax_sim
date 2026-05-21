# Plan 3: 编译器支持（LLVM/Clang）

## 目标

为 Multi-Lane TLP Extension 提供 LLVM/Clang 编译器支持，分两阶段：
1. **第一版**：intrinsic + 汇编宏包装（快速可用）
2. **第二版**：`#pragma simt` 自动代码生成（完整方案）

## 基线

- 系统已安装 LLVM/Clang，支持 RISC-V 后端
- LLVM 已有 RVV intrinsic 支持（`__riscv_vle32_v_f32m4` 等）
- 系统已安装 `riscv64-linux-gnu-gcc`

---

## 第一版：Intrinsic + 汇编宏

### 交付物

文件: `softmax_sim/compiler/cute_simt.h`

### 实现

```c
// cute_simt.h — Multi-Lane TLP Extension 第一版支持

#ifndef CUTE_SIMT_H
#define CUTE_SIMT_H

#include <riscv_vector.h>
#include <stdint.h>

// ---- CSR 操作 ----

#define SIMT_SET_STRIDE(ctx, val) \
    asm volatile("csrw 0x7C" #ctx ", %0" :: "r"((uint32_t)(val)))

#define SIMT_SET_STRIDE_SHARED(ctx) \
    asm volatile("csrw 0x7C" #ctx ", zero")

// ---- vsetvli with lanes ----

// vtype 编码: vlane 在 bit[9:8]
// 00=1lane, 01=2lanes, 10=4lanes, 11=8lanes
// vtype = (vlane << 8) | (vsew | vlmul | vta | vma)
// 这里简化处理，直接构造 vtype immediate

#define SIMT_VTYPE(lmul, sew, lanes) \
    ((lanes << 8) | sew | lmul)  // 简化，需要根据实际编码调整

// ---- Multi-lane load/store ----
// 利用 lumop 编码:
//   lumop=00000 → vlane0 (标准 vle，向后兼容)
//   lumop=00001 → vlane1
//   lumop=00010 → vlane2
//   lumop=00011 → vlane3

// 第一版用 inline asm 实现，因为标准 intrinsic 不支持 lumop 扩展

#define SIMT_VLE32_VLANE0(vd, rs1, vl) \
    __riscv_vle32_v_f32m4(rs1, vl)  // 标准 intrinsic, vlane0

// vlane1-vlane3 需要自定义 asm，因为 lumop 编码不同
// 用 .insn 伪指令构造自定义编码
// 具体编码需要根据 RVV load 指令格式计算

#define SIMT_VLE32_VLANE1(vd_ptr, rs1, vl) \
    asm volatile(".insn r 0x07, 0x0, 0x1, %0, %1, x0" \
                 : "=vr"(*(vd_ptr)) : "r"(rs1))  // lumop=00001

// store 同理
#define SIMT_VSE32_VLANE0(vs, rs1, vl) \
    __riscv_vse32_v_f32m4(rs1, vs, vl)  // 标准 intrinsic, vlane0

#define SIMT_VSE32_VLANE1(vs, rs1, vl) \
    asm volatile(".insn r 0x27, 0x0, 0x1, x0, %0, %1" \
                 :: "vr"(vs), "r"(rs1))  // sumop=00001

// ---- Kernel 模板 ----

#define SIMT_KERNEL_BEGIN(lanes, stride0, stride1, stride2) \
    SIMT_SET_STRIDE(0, stride0); \
    SIMT_SET_STRIDE(1, stride1); \
    SIMT_SET_STRIDE_SHARED(2);

#define SIMT_KERNEL_END() \
    /* reset strides to 0 */ \
    SIMT_SET_STRIDE_SHARED(0); \
    SIMT_SET_STRIDE_SHARED(1); \
    SIMT_SET_STRIDE_SHARED(2);

#endif // CUTE_SIMT_H
```

### 使用示例

```c
#include "cute_simt.h"

void rmsnorm_kernel(const float *input, float *output,
                    const float *weight, int hidden_dim) {
    int row_bytes = hidden_dim * sizeof(float);

    SIMT_KERNEL_BEGIN(4, row_bytes, row_bytes, 0);

    size_t vl = __riscv_vsetvl_e32m4(hidden_dim);
    // ... 加上 lanes 参数的 vsetvli ...

    vfloat32m4_t v = SIMT_VLE32_VLANE0(input, vl);    // per-lane
    vfloat32m4_t w = __riscv_vle32_v_f32m4(weight, vl); // shared (standard vle)
    vfloat32m4_t out = __riscv_vfmul_vv_f32m4(v, w, vl);
    SIMT_VSE32_VLANE1(out, output, vl);                 // per-lane output

    SIMT_KERNEL_END();
}
```

### 注意事项

- `.insn` 伪指令的具体编码需要根据 RVV load/store 指令的 bit layout 精确计算
- lumop 在 unit-stride load 的编码位置是 bits[19:15]（对应 rs2 字段）
- 可能需要用 `.long` 直接写 32-bit 指令编码作为备选方案

---

## 第二版：`#pragma simt` LLVM Pass

### 目标

在 LLVM 的 RISC-V 后端添加 pragma 支持，自动生成 CSR 写入和带 vlane ctx 的 load/store。

### 涉及的 LLVM 模块

```
clang/lib/Parse/          — pragma 解析
clang/lib/Sema/           — pragma 语义检查
clang/lib/CodeGen/        — AST → LLVM IR 生成
llvm/lib/Target/RISCV/    — RISC-V 后端指令选择
```

### Step 1: Pragma 解析 — `clang/lib/Parse/ParsePragma.cpp`

注册 `#pragma simt`，解析参数：

```c++
// #pragma simt(lanes=N, stride0=expr, stride1=expr, stride2=expr, stride3=expr)
// 解析为 SimtPragmaAttr AST 节点
```

### Step 2: AST 属性 — `clang/include/clang/Basic/Attr.td`

```c++
def SimtPragma : InheritableAttr {
  let Args = [UnsignedArgument<"Lanes">, ...];
}
```

### Step 3: CodeGen — `clang/lib/CodeGen/CGStmt.cpp`

在 pragma 作用域的函数入口前插入：

```llvm
; CSR writes
call void @llvm.riscv.csrw(i32 0x7C0, i32 %stride0)
call void @llvm.riscv.csrw(i32 0x7C1, i32 %stride1)
call void @llvm.riscv.csrw(i32 0x7C2, i32 0)
; vsetvli with lanes
call i64 @llvm.riscv.vsetvli(i64 %avl, i64 %vtype_with_lanes)
```

### Step 4: Load/Store 指令选择 — `llvm/lib/Target/RISCV/RISCVISelDAGToDAG.cpp`

分析 pragma 作用域内的 load/store：
- 根据指针参数 → vlane ctx 映射（从 pragma 参数推导）
- 选择对应的 lumop 编码
- 生成带 vlane ctx 的 machine instruction

### Step 5: 验证

- 编译生成 `.s` 文件，检查 CSR 写入和 lumop 编码
- 在 QEMU 模拟器上运行编译出的程序
- 结果正确性验证

---

## 实施顺序

```
第一版（1-2 天）:
  1. 编写 cute_simt.h 汇编宏头文件
  2. 验证 CSR 读写（在 QEMU 上跑简单测试）
  3. 验证 load/store 的 lumop 编码
  4. 用汇编宏改写一个 CUTE kernel（如 rmsnorm）

第二版（2-4 周，按需）:
  1. LLVM pragma 解析
  2. CodeGen CSR 写入
  3. RISC-V 后端指令选择
  4. 多 kernel 端到端验证
```
