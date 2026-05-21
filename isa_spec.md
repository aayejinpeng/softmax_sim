# RVV Multi-Lane TLP Extension ISA Specification

## 1. Scope

This document defines an experimental RISC-V Vector (RVV) extension for
broadcasting one vector instruction stream across multiple independent vector
lanes. Each lane has an independent logical vector register file and scoreboard,
while decode, scalar operands, and execution units may be shared by the
implementation.

The extension is intended for workloads with many independent rows or heads,
such as softmax, RMSNorm, RoPE, and elementwise activation kernels. It is
designed so that `vlane = 0` preserves standard RVV behavior.

## 2. Architectural State

### 2.1 `vtype.vlane`

The extension allocates bits `[9:8]` of `vtype` for the lane count selector.

| Bits | Name | Access | Reset | Meaning |
|------|------|--------|-------|---------|
| `[9:8]` | `vlane` | R/W | `0` | Multi-lane execution count selector |
| `[7:0]` | standard RVV fields | standard | standard | `vsew`, `vlmul`, `vta`, `vma` |

`vlane` encoding:

| `vlane` | Active lanes |
|---------|--------------|
| `00` | 1 lane |
| `01` | 2 lanes |
| `10` | 4 lanes |
| `11` | 8 lanes |

When `vlane = 00`, all instructions execute exactly as standard RVV
instructions. When `vlane != 00`, each subsequent vector instruction is
architecturally broadcast to all active lanes.

The scalar `vl` value produced by `vsetvli` is shared by all active lanes.

### 2.2 Lane Stride CSRs

The extension defines four machine-visible lane stride CSRs. The CSR values are
byte strides used by vector unit-stride load/store instructions.

| CSR address | Name | Width | Reset | Meaning |
|-------------|------|-------|-------|---------|
| `0x7C0` | `vlane0` | 32 bits | `0` | Lane stride context 0, in bytes |
| `0x7C1` | `vlane1` | 32 bits | `0` | Lane stride context 1, in bytes |
| `0x7C2` | `vlane2` | 32 bits | `0` | Lane stride context 2, in bytes |
| `0x7C3` | `vlane3` | 32 bits | `0` | Lane stride context 3, in bytes |

A stride value of zero means every active lane uses the same base address for
that memory instruction. This is the expected encoding for shared data such as
normalization weights, gamma/beta vectors, or RoPE theta tables.

## 3. Load/Store Encoding

The extension uses the low reserved values of unit-stride vector load `lumop`
and vector store `sumop` to select one of the four lane stride CSRs.

### 3.1 Unit-Stride Loads

| `lumop[4:0]` | Mnemonic suffix | Stride CSR | Compatibility |
|--------------|-----------------|------------|---------------|
| `00000` | none, for example `vle32.v` | `vlane0` | standard RVV encoding |
| `00001` | `.l1`, for example `vle32.l1.v` | `vlane1` | extension |
| `00010` | `.l2`, for example `vle32.l2.v` | `vlane2` | extension |
| `00011` | `.l3`, for example `vle32.l3.v` | `vlane3` | extension |

### 3.2 Unit-Stride Stores

| `sumop[4:0]` | Mnemonic suffix | Stride CSR | Compatibility |
|--------------|-----------------|------------|---------------|
| `00000` | none, for example `vse32.v` | `vlane0` | standard RVV encoding |
| `00001` | `.l1`, for example `vse32.l1.v` | `vlane1` | extension |
| `00010` | `.l2`, for example `vse32.l2.v` | `vlane2` | extension |
| `00011` | `.l3`, for example `vse32.l3.v` | `vlane3` | extension |

Other `lumop` and `sumop` values retain their standard or reserved meanings.
This extension only defines the unit-stride forms listed above.

## 4. Instruction Semantics

### 4.1 Lane Broadcast

For any vector instruction `I` executed with `active_lanes > 1`, the
architectural effect is equivalent to executing `I` once for each `lane_id` in
`[0, active_lanes)`, using the lane-specific physical vector register mapping
defined in Section 5.

Scalar operands, immediate operands, `vl`, `vtype`, mask policy, and tail policy
are shared across all lanes.

### 4.2 Memory Effective Address

For unit-stride vector loads and stores, each lane computes:

```text
effective_addr = x[rs1] + element_byte_offset
               + lane_id * vlane_stride[vlane_ctx]
```

Where:

- `x[rs1]` is the scalar base address.
- `element_byte_offset` is the normal unit-stride byte offset within the vector
  memory operation.
- `lane_id` is the current lane number.
- `vlane_ctx` is selected by `lumop[1:0]` for loads and `sumop[1:0]` for stores.
- `vlane_stride[vlane_ctx]` is one of `vlane0` through `vlane3`.

If the selected stride is zero, all lanes access the same address range. If the
selected stride is nonzero, lane `n` accesses a byte range offset by
`n * stride`.

### 4.3 Non-Memory Vector Instructions

Arithmetic, logical, permutation, and reduction vector instructions do not use
lane stride CSRs. They only use the lane-specific vector register mapping.

Reductions reduce within each lane independently. No implicit cross-lane
reduction is performed by this extension.

## 5. Vector Register Mapping

Logical vector register numbers are mapped to physical vector registers by lane:

```text
physical_vreg = logical_vreg + lane_id * 32
```

For four lanes:

| Lane | Logical register range | Physical register range |
|------|------------------------|-------------------------|
| 0 | `v0` - `v31` | `0` - `31` |
| 1 | `v0` - `v31` | `32` - `63` |
| 2 | `v0` - `v31` | `64` - `95` |
| 3 | `v0` - `v31` | `96` - `127` |

An implementation that supports `N` active lanes must provide `32 * N`
physical vector registers, or an equivalent architectural mapping.

## 6. Programming Model

Example RMSNorm-style setup:

```asm
# t0 = input row stride, t1 = output row stride
csrw vlane0, t0
csrw vlane1, t1
csrw vlane2, zero

vsetvli a0, a1, e16, m1, ta, ma, lanes=4

vle16.v    v4, (a0)      # input, per lane via vlane0
vle16.l2.v v8, (a2)      # shared weight via vlane2, stride 0
vfmul.vv   v4, v4, v8    # independent per-lane computation
vse16.l1.v v4, (a1)      # output, per lane via vlane1
```

Recommended stride contexts:

| Context | Typical use |
|---------|-------------|
| `vlane0` | input rows, activations, or per-lane source tensors |
| `vlane1` | output rows or per-lane destination tensors |
| `vlane2` | shared weights, gamma/beta, theta, constants |
| `vlane3` | scratch, temporary streams, or kernel-specific data |

## 7. Constraints

- `vlane = 00` must be backward compatible with standard RVV.
- `vl` is identical for all lanes.
- Mask and tail policy are identical for all lanes.
- `vlmul` is legal only if the implementation can map all logical register
  groups for all active lanes. A conforming implementation may reject a
  `vtype` setting when the required physical vector registers are unavailable.
- Memory ordering follows the normal ordering rules for the equivalent sequence
  of per-lane vector memory instructions.
- The extension does not define cross-lane communication, cross-lane
  reductions, or lane predication.

## 8. Compatibility Checklist

- Standard encodings with `vlane = 00` execute as standard RVV.
- Standard unit-stride `vle*.v` and `vse*.v` use `vlane0`; with reset
  `vlane0 = 0`, every lane sees the original scalar base address.
- New `.l1`, `.l2`, and `.l3` suffixes only select the address stride context;
  they do not change element width, `vl`, masking, or register semantics.

