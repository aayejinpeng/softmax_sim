# 开题报告第二点：面向 AICPU 的 SIMTD 向量执行机制

## 研究定位

本课题的第二个研究点，聚焦于在 Saturn / Titan-I 这类有限乱序向量单元上，引入一种面向 AI workload 的 SIMTD 执行机制。这里的目标不是把 AICPU 做成传统高性能 CPU 的小型版本，也不是单纯追求更大的乱序窗口，而是利用 AI 算子中普遍存在的 task-level parallelism，把多个 head、row、token 或后处理片段组织成可交错推进的独立执行流，让有限硬件在较低代价下获得更高吞吐。

SIMTD 的核心思想是：编译器和 ISA 显式暴露多个结构相同但数据独立的任务流，硬件以轻量 scoreboard、有限窗口调度、lane/context 映射和必要的寄存器版本管理来交错执行这些任务。这样做可以用更简单的微结构榨取 AI workload 中天然存在的并行性，而不是依赖昂贵的大核式乱序后端去动态“猜”并行性。

## 问题背景

传统高性能 CPU 为了最大化通用单线程性能，通常采用向量寄存器重命名、大容量向量物理寄存器堆、深调度队列和复杂旁路网络。这类方案能够提供强大的向量乱序执行能力，但面积、功耗和验证复杂度都很高。对于 AICPU，更关键的不是支持任意程序上的强乱序，而是高效执行 AI 推理中反复出现的结构化算子。

这些算子的一个共同特点是：单个 head 或 row 内部往往存在较长依赖链，例如 softmax 的 reduce、exp、sum、div，rmsnorm 的 reduce 和 scale，silu 的指数近似链，以及 rope 的位置编码计算；但在不同 head、row、token 或融合后处理片段之间，又存在大量相互独立的 TLP。如果硬件只顺序推进单个任务流，就会被局部依赖链和访存延迟拖住；如果直接复制大核式 OOO，又会为 AICPU 付出过高代价。

因此，本课题希望回答的问题是：能否通过 SIMTD 方式，把 AI workload 中常见的 TLP 组织成硬件容易消费的形式，让一个有限乱序向量单元在不显著增加复杂度的前提下，稳定隐藏局部延迟并提升吞吐。

## 实验动机：张量单元之外的向量瓶颈

本研究的直接动机来自 AICPU 研制过程中的一个现象：张量单元能够提供很高的矩阵计算吞吐，但完整 AI 算子的执行并不是只由矩阵乘决定。softmax、rmsnorm、silu、rope、量化/反量化、数据搬运和融合后处理等步骤，都需要向量单元与张量单元配合完成。即使从峰值算力估算看，向量单元的计算能力已经能够覆盖这些操作，实际执行中仍然可能因为任务组织、依赖隐藏和访存/计算交错不足，导致张量单元前后出现等待。

因此，这里的问题不是简单地继续增加向量计算宽度，而是要回答：向量单元怎样以较低代价持续供给和消化张量单元周围的非矩阵计算。已有 workload benchmark 和微结构建模表明，AI workload 中存在大量规则的 TLP：多 head attention 中不同 head 独立，norm/activation 中不同行独立，rope 中不同 token 或位置块独立，融合后处理函数中也常出现多个结构相同、数据互不依赖的片段。当前实验的意义，就是把这些 TLP 用可控的硬件参数建模出来，说明 SIMTD 机制为什么值得做、应该在哪里做、需要编译器和 ISA 暴露哪些信息。

从建模结果看，AI workload 呈现出一种非常适合 SIMTD 的结构：单个任务流内部依赖链较深，但任务流之间独立性强。以 softmax 为例，单个 head 内部存在 reduce、exp 近似、sum 和 div 的串行链；以 rmsnorm 为例，单行内部需要 reduce 和 scale；以 silu 和 rope 为例，函数近似、旋转和访存之间也存在局部依赖。但这些链在不同 head、row、token 之间可以并行推进。如果硬件只能看到单个任务流，就会浪费向量单元可用吞吐；如果让编译器和 ISA 把多个任务流显式暴露给硬件，有限窗口和轻量 scoreboard 就能在等待一个任务时推进另一个任务，从而更稳定地利用向量执行资源。

## 已有基础与设计切入点

目前已经形成了三层基础。

第一是微结构建模基础。现有模拟器已经能够描述不同 lane/context 数量、issue width、issue queue window、scheduler window、寄存器重命名、循环展开和逻辑寄存器数量对 softmax、rmsnorm、silu、rope 等 kernel 的影响，并输出 cycle 与各类部件利用率。这些实验的核心作用，是刻画 AI workload 中 TLP 被硬件消费的条件：需要多少可见任务流、多少逻辑寄存器预算、怎样的窗口和发射组织，才能把向量单元从局部依赖链中释放出来。

第二是 ISA 设计基础。已有方案已经引入面向多 lane 的执行语义，包括 lane 数量、lane stride、load/store 访问上下文和逻辑寄存器映射等机制。它们为 SIMTD 提供了体系结构接口：编译器可以把多个 head、row、token 或融合后处理片段映射到不同 lane/context，硬件则不需要从普通指令流中复杂地推断这些并行性。

第三是编译流程基础。现有 benchmark 生成和指令流建模已经能够表达不同 kernel 的依赖链，并支持在逻辑寄存器预算内进行循环展开。后续编译器应进一步负责识别 AI 算子中的规则 TLP，选择展开粒度，分配逻辑寄存器银行，设置 lane stride，并生成适合 SIMTD 硬件消费的指令顺序。

## 拟开展的三方向设计

微结构方向将围绕轻量 SIMTD 向量后端展开。每个 lane/context 表示一个独立任务流，具备独立的依赖跟踪和逻辑寄存器视图；多个 lane 共享向量执行单元、访存单元和发射资源。硬件侧重点不是构建大容量通用乱序后端，而是通过有限 issue queue、轻量 scheduler、scoreboard、lane/context 仲裁和必要的版本化映射，在多个显式任务流之间选择 ready 操作执行。这样可以把硬件复杂度集中投入到 AI workload 确实需要的 TLP 消费能力上。

ISA 方向将定义 SIMTD 对软件可见的执行抽象。核心包括 lane 数量配置、lane stride CSR、load/store 的访问上下文、共享数据访问语义、每个 lane 的逻辑寄存器映射，以及与现有 RVV 编程模型的兼容关系。ISA 的目标是让编译器能够显式表达“这些任务结构相同、数据独立、可以交错执行”，同时让硬件保持简单清晰的执行边界。

编译流程方向将负责把 AI kernel 中的 TLP 转换成硬件可消费的 SIMTD 任务流。编译器需要识别 head/row/token 级并行，分析单任务依赖链和逻辑寄存器需求，在不超过逻辑寄存器数量的前提下自动展开到合适规模，并为输入、输出、共享参数和临时 scratch 数据生成 lane stride 与访存上下文。对于 softmax、rmsnorm、silu、rope 以及更复杂的融合后处理函数，编译器还应根据依赖链深度和访存模式选择不同的交错策略。

## 实现与验证路径

第一阶段继续完善现有 workload benchmark 和微结构模拟器。重点不是只比较某一种调度策略，而是用多 kernel、多硬件配置的结果说明：AI workload 中哪些 TLP 可以被 lane/context、循环展开、逻辑寄存器分配和轻量调度有效利用，哪些瓶颈来自访存、reduce、函数单元或寄存器组织。

第二阶段在 ISA 和功能模拟层验证 SIMTD 编程模型。基于现有 RVV 扩展方案，补充 lane stride、load/store context 和多 lane 寄存器映射的功能模拟，使编译出的 kernel 能够在功能层验证执行语义正确性。

第三阶段在 Titan-I 和 Saturn 这样的有限乱序向量单元上推进实现。这里的实现目标不是孤立地增强一个向量核，而是服务于 AICPU 的整体执行：向量单元需要与第一个工作中的 tensorcore 结合，在 RTL/SoC 级仿真中评估矩阵计算、向量前后处理、访存和融合后处理之间的端到端协同效果。最终希望验证 SIMTD 向量执行机制能否在真实微结构约束下提升 AI kernel 的整体吞吐和硬件利用率。

## 预期贡献

本研究预期形成一套面向 AICPU 的 SIMTD 向量执行机制。它把 AI workload 中常见的 head、row、token 和融合后处理级 TLP 显式组织起来，使有限乱序向量单元能够在较低硬件代价下持续推进多个独立任务流。

本研究还将形成微结构、ISA 和编译流程协同设计方法。微结构提供轻量 lane/context 调度和依赖跟踪，ISA 提供可编程的 SIMTD 执行语义，编译器负责识别和展开规则 TLP。三者共同解决“向量算力理论上够用，但实际难以稳定转化为 AI 算子吞吐”的问题。

最后，本研究将给出可落地的验证路径：从 workload benchmark 和微结构模型出发，经 ISA/功能模拟验证，再到 Titan-I/Saturn 上的实现，并与 tensorcore 设计共同进入 RTL/SoC 级仿真。这样可以把第二个研究点放在完整 AICPU 研制链条中，而不是停留在单独的模拟器优化上。

## 可直接放入开题报告的表述

在 AICPU 研制过程中，我们发现张量单元虽然能够提供较高的矩阵计算吞吐，但完整 AI 算子的执行仍然强依赖向量单元完成 softmax、normalization、activation、RoPE、数据搬运以及融合后处理等工作。理论上，向量单元的峰值算力已经可以满足这些计算需求；但 workload 分析和微结构建模表明，现有向量单元在任务流组织、依赖隐藏、逻辑寄存器分配以及访存/计算交错方面仍存在瓶颈，难以稳定榨取 AI workload 中普遍存在的 head/row/token 级 TLP。

因此，本课题拟研究一种面向 AICPU 的 SIMTD 向量执行机制。该机制通过 ISA 和编译器显式暴露多个结构相同但数据独立的任务流，并由硬件以 lane/context、轻量 scoreboard、有限窗口调度和必要的寄存器版本管理进行交错执行。围绕这一机制，本课题将展开微结构、ISA 体系结构和编译流程三个方向的协同设计，并计划在 Titan-I、Saturn 等有限乱序向量单元上实现验证。后续还将与 tensorcore 工作结合，进行 RTL/SoC 级仿真评估，从端到端 AI kernel 执行角度验证该设计对 AICPU 向量/张量协同效率的提升。

## 参考来源

- `PLAN.md`
- `benchmark_result.md`
- `plans/benchmark_results.md`
- `plans/benchmark_config.example.yaml`
- `plans/plan1_simulator.md`
- `plans/plan2_qemu.md`
- `plans/plan3_compiler.md`
- `isa_spec.md`
