# art3\_SGEMM CUDA 优化：从 Naive 到 cuBLAS 级别性能的逐步工作日志

# 1\. 导读 — 从零开始优化 SGEMM

**TL;DR：**本文通过 12 个递进式 CUDA kernel 优化步骤，将单精度矩阵乘法（SGEMM）性能从 309 GFLOPs（cuBLAS 的 1\.3%）提升至 \~22 TFLOPs（cuBLAS 的 93\.7%）。核心优化链：合并访存 → 共享内存分块 → 1D/2D Block Tiling → 向量化访存 → Bank Conflict 消除 → 自动调优 → Warp Tiling → 双缓冲。所有代码基于本工程 `src/kernels/` 实现，参考 siboehm\.com 的 CUDA\-MMM 工作日志。

**阅读导航：**本文按照性能瓶颈的递进发现链组织——每章解决前一章暴露的新瓶颈。先理解 GPU 内存层次（第 2 章）→ 优化全局内存访问（第 3 章）→ 利用共享内存缓存（第 4 章）→ 提高算术强度（第 5 章）→ 向量化与 Bank Conflict（第 6\-7 章）→ 自动调优与 Warp Tiling（第 8\-9 章）→ 计算与访存重叠（第 10 章），最终以总结收尾（第 11 章）。

**带着这三个问题阅读本文：**



1. 如何通过逐层 Tiling（Block → Warp → Thread → Register）最大化算术强度（Arithmetic Intensity）？

2. GPU 各层级内存（Global、L2、Shared、Register）的带宽与延迟特征如何决定优化策略？

3. 从 309 GFLOPs 到 22 TFLOPs，每个优化步骤各自解除了什么瓶颈、带来了多少收益？

---

---

# 2\. 理论基础 — GPU 性能模型与内存层次

**TL;DR：**理解 GPU 的内存层次和 Roofline 模型是后续所有优化的理论基础。共享内存带宽是全局内存的 \~16 倍（12 TB/s vs 750 GB/s），而寄存器比共享内存更快。优化目标：最大化算术强度（FLOPs/Byte），让 kernel 从 Memory\-Bound 跨越到 Compute\-Bound。

**本章递进链：**先理解 GPU 的四级内存层次（§2\.1）→ 用 Roofline 模型定位瓶颈类型（§2\.2）→ 量化瓶颈用算术强度（§2\.3）→ 评估延迟隐藏能力用占用率（§2\.4）→ 最后理解 Warp 调度如何将高占用率转化为实际吞吐（§2\.5）。

## 2\.1 GPU 内存层次结构（GPU Memory Hierarchy）

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=N2YyYTVlMDNlZTliYjhjNjc3ZTBlYWFjNjI1MGUyM2RfMDRlNWQ4Y2MzNmM1MTMyNTJiMDBhMzk5Y2RlNjA5YTJfSUQ6NzY1ODU4NzMyODYwNDI2MTYyMl8xNzg0ODk5ODI4OjE3ODQ5ODYyMjhfVjM)

GPU 的内存系统分为四个层级，从上到下容量递增、带宽递减、延迟递增：

**全局内存（GMEM/HBM）** — 片外显存，容量最大（8\-80 GB），带宽 \~400\-768 GB/s，延迟 \~200\-800 cycles。NVIDIA 广告中的"80GB 显存、1TB/s 带宽"指的就是这一层。

**L2 Cache** — 片上全局缓存，\~6\-8 MB，所有 SM 共享，带宽 \~2\-3 TB/s。

**共享内存（SMEM）\+ L1 Cache** — 片上每 SM 共 100KB（可配置 SMEM/L1 比例），SMEM 带宽 \~12 TB/s（约 GMEM 的 16 倍），延迟 \~20\-30 cycles。Block 内所有线程共享——这是 Kernel 3 及之后所有优化的核心。

**寄存器（Register File）** — 每 SM 256KB（65536×4B），每线程最多 255 个，延迟接近 0 cycles，线程私有。优化最深的 Kernel 将大部分数据放在寄存器中计算。

**核心认知：**优化 SGEMM 的本质是**把数据从慢内存逐级搬运到快内存，并在最快的内存中尽可能多地复用。**从 Kernel 1 到 Kernel 12 的每一步，都是在减少 GMEM 访问、增加 SMEM/Register 复用。

## 2\.2 Roofline 模型（Roofline Model）

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZGY2YWY5ODg4NDk1OTJkZmU5YjIwMDgzZWM0M2UwNzdfMWI2MTE5MTUyNDIyZTU1ZGMxNmJjM2I1NDJlNWM4MDVfSUQ6NzY1ODU4NzgzNTIxMzQzNDAzN18xNzg0ODk5ODI4OjE3ODQ5ODYyMjhfVjM)

**Roofline 模型**（Williams et al\., 2009）用一个简单的约束方程描述 Kernel 性能上限：**Attainable Performance = min\(Peak FLOPs, Peak BW × Arithmetic Intensity\)**。在 XY 图上，X 轴为算术强度（FLOPs/Byte），Y 轴为可达到的性能（FLOPs/s）。水平线 = 峰值算力天花板，斜线 = 带宽天花板。两条线的交点称为**"拐点"（Ridge Point）**，AI\<sub\>ridge\</sub\> = Peak FLOPs / Peak BW。

对于 A6000（30 TFLOPs / 728 GB/s），拐点 ≈ **41\.2**** FLOPs/Byte**。低于此值：**Memory\-Bound** — 性能受带宽限制，增加 FLOPs 无法提升速度；高于此值：**Compute\-Bound** — 性能受算力限制，优化访存无济于事。SGEMM naive kernel 仅 \~15\.4 FLOPs/Byte（严重 Memory\-Bound），而 cuBLAS 做到了 \~245 FLOPs/Byte（Compute\-Bound）。整个优化过程就是**沿 X 轴向右移动。**

## 2\.3 算术强度（Arithmetic Intensity）

**算术强度（Arithmetic Intensity, AI）** = 总浮点运算数（FLOPs）/ 总数据传输量（Bytes）。AI 是 Roofline 模型的 X 轴，衡量每个 Byte 的数据搬运"做了多少有用功"。

对 SGEMM（M×N×K）：总 FLOPs ≈ **2×M×N×K**（每个 C 元素需要 K 次乘法 \+ K 次加法 = 2K FLOPs）。总数据传输取决于实现：**Naive kernel**（每个线程独立从 GMEM 加载）→ 理论最小传输 \~268 MB，但实际因零缓存导致 \~548 GB → AI ≈ **0\.25 FLOPs/Byte** → 极低；**cuBLAS**（高度优化的 tiling）→ 实际 GMEM 传输仅 \~500 MB → AI ≈ **274 FLOPs/Byte** → 远超拐点 41\.2，完全 Compute\-Bound。整个优化的核心目标：**最大化 AI，让每次数据加载后的计算尽可能多。**

## 2\.4 占用率（Occupancy）

**占用率（Occupancy）** = 每 SM 活跃 Warp 数 / 每 SM 最大 Warp 数。高占用率意味着有更多 Warp 可供调度器在等待内存时切换，从而更好地隐藏延迟。

占用率受三种资源限制（取最紧张者）：**寄存器**（Register\-per\-thread × threads/block ≤ 65536/SM）、**共享内存**（SMEM\-per\-block ≤ 100KB/SM）、**线程数**（threads/block ≤ 1536/SM）。

**计算实例（Kernel 3, A6000）：**37 regs/thread → 37×32 = 1184 regs/warp → 按 256 粒度上取整 = 1280 → 1280×32 warps/block = 40960 regs/block，65536/40960 ≈ 1\.6 → 只够 1 Block/SM。SMEM: 9216B/Block → 100KB/9216 ≈ 11 Blocks。Threads: 1024/1536 → 1 Block。寄存器成为瓶颈 → 最终 Occupancy = 32/48 = **66%**。

⚠️ 高占用率并非总是必要：当 AI 极高或极低时，低占用率也能达到峰值性能（Volkov's "cusp behavior"）。

## 2\.5 延迟隐藏与 Warp 调度（Latency Hiding \& Warp Scheduling）

GPU 隐藏内存延迟的核心机制是**Warp 级并发切换**：当 Warp A 发出内存请求后需要等待 \~200\-800 cycles，Warp Scheduler 立即切换到 Warp B 执行已就绪的指令。只要有足够多的活跃 Warp，计算单元就不会空闲。

一个 SM 内有 **4 个 Warp Scheduler**，每个可调度最多 12 个 Warp（A6000: 48 Warps/SM max）。FMA 指令延迟仅约 **4 cycles**（\~2\.6ns @1\.5GHz），远低于内存延迟。高 Occupancy 的价值在于提供更大的"Warp 池"，让 Scheduler 总有可切换的候选。

**通俗类比：**CPU 用大 L1/L2/L3 Cache 减少延迟；GPU 用海量线程 \+ 快速上下文切换来**容忍**延迟。前者是"避免等待"，后者是"等待时做别的事"。

## 2\.6 理论基础检查清单

||检查项|参考|
|---|---|---|
|□|能否画出 GPU 内存层次图并标注各级带宽/延迟？|§2\.1|
|□|能否用 Roofline 模型判断 kernel 是 Memory\-Bound 还是 Compute\-Bound？|§2\.2|
|□|能否计算给定 kernel 的算术强度？|§2\.3|
|□|是否能根据 Register/SMEM 使用量计算 Occupancy？|§2\.4|
|□|能否解释 Warp 调度如何通过并发隐藏内存延迟？|§2\.5|

---

# 3\. 基础优化 — 全局内存合并访存（Global Memory Coalescing）

**TL;DR：**Naive kernel 仅 309 GFLOPs（cuBLAS 的 1\.3%），访存带宽仅 15 GB/s。根本原因是同一 Warp 内线程访问非连续地址，导致每次访存拆分为 32 次 32B 事务。通过调整 threadIdx 到 C 矩阵位置的映射，使得同一 Warp 内线程访问连续地址，访存带宽提升至 110 GB/s，性能跃升至 1986 GFLOPs（6\.4x 提升）。

本章递进链：Warp 执行模型 → Naive kernel 分析 → 合并访存原理 → Kernel 2 实现。

## 3\.1 线程层次与 Warp 执行模型（Thread Hierarchy \& Warp）

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YjBlMWZmMTEyMzk1ODQxZTg2Nzg2NjAzZDQyZjhhZTBfMTc5ODA1OWQzNjVkYjhlNzE5MGFkNGRkYjM4MmM0ZDVfSUQ6NzY1ODU4ODA4MDAzOTIzNDc3MV8xNzg0ODk5ODI4OjE3ODQ5ODYyMjhfVjM)

CUDA 的线程组织分为三层：**Grid**（网格）包含多个 **Block**（线程块），每个 Block 包含多个 **Thread**（线程）。此外还有一个硬件层面的关键概念——**Warp**（线程束）。

Warp 是 GPU 的**最小调度单元**，固定包含 **32 个线程**。线程按 threadId 连续分组：threadId = threadIdx\.x \+ blockDim\.x × \(threadIdx\.y \+ blockDim\.y × threadIdx\.z\)。这意味着 **threadIdx\.x 连续 → 同一 Warp**——后续所有合并访存优化都基于这个事实。

每个 SM 内有 **4 个 Warp Scheduler**，每个周期从就绪 Warp 中选择一个发射指令。Warp 内的 32 个线程以 SIMT（Single Instruction, Multiple Threads）方式执行：所有线程执行同一条指令，但各自操作不同的数据。理解 Warp 是理解合并访存、Bank Conflict 和 Warp Tiling 的前提。

## 3\.2 Kernel 1：Naive 实现分析（Kernel 1: Naive）

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YTRmOTEzYzhkOTRlNjVmMGU0YjJiMmY1ODQ1MjgxYjVfMjhhZGVjNmZlYTc4ZDU4NDAxNDllZWExNWVmN2YxZmRfSUQ6NzY1ODU4ODQwNjQ0MzkyMDU4N18xNzg0ODk5ODI4OjE3ODQ5ODYyMjhfVjM)

Naive kernel 的设计思路非常直观：用 2D Grid \+ 2D Block 将每个线程映射到 C 矩阵的一个元素，每个线程独立计算一行 A 与一列 B 的点积。

**启动配置：**Block 大小 32×32 = 1024 threads，Grid 覆盖 ceil\(M/32\)×ceil\(N/32\) 个 Block。每个线程执行 2K 次 FLOPs（K 次乘法 \+ K 次加法），总计 2×M×N×K FLOPs。

**性能分析（A6000, 4096²）：**实测 **309 GFLOPs**（cuBLAS 的 1\.3%），GMEM 吞吐量仅 **15 GB/s**（峰值 768 GB/s 的 2%）。问题根源在于**访存模式**：thread \(x,y\) 加载 A\[x\*K \+ i\]，同一 Warp 内 threadIdx\.x 连续的线程访问的是 A 的不同行 → 地址不连续 → 无法合并。

**理论访存量：**若假设零缓存，每个线程加载 2K\+1 个 float = 32KB，4096² 线程总共需 \~548 GB 内存流量——远超 201MB 的绝对最小值。即使有 L2 Cache，Naive kernel 的访存模式也无法有效利用。

```cuda
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];  // ❌ 同行 Warp 线程访问非连续地址
    }
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}
```

## 3\.3 Kernel 2：合并访存实现（Kernel 2: Global Memory Coalescing）

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=Njc2NDJiNmU0ZjAxMjVlZTQ2MTJmMmYyNzE1ZWExMDBfYWE0NjUyODAwOGZhOWE3YjIzMjc1YTcxODJhMjAzMjFfSUQ6NzY1ODU4ODUxNTM3ODY5NTQwNF8xNzg0ODk5ODI4OjE3ODQ5ODYyMjhfVjM)

解决 Naive kernel 访存问题的关键是**合并访存（Global Memory Coalescing）**：同一 Warp 的 32 个线程同时访问 **连续的 128 字节** 时，GPU 硬件将其合并为一次 128B 的内存事务。这是利用率从 2% 跳到 15% 的唯一改动。

**核心变换：**将 blockDim 从 2D \(32,32\) 改为 1D \(1024\)，用除法和取模重新映射：cRow = blockIdx\.x × BLOCKSIZE \+ **\(threadIdx\.x / BLOCKSIZE\)**；cCol = blockIdx\.y × BLOCKSIZE \+ **\(threadIdx\.x % BLOCKSIZE\)**。这样 threadIdx\.x 连续的 32 个线程访问 C 的同一行、连续的 32 列 → 它们加载 B 的同一行但连续的列地址 → **完美合并。**

**为什么只改了两行代码？**因为合并访存完全由 GPU 硬件在运行时完成——编译器无法在编译时保证矩阵指针的对齐和连续性，所以 SASS 指令看起来和 Naive 基本一样（Godbolt 可验证）。硬件检测到同一 Warp 访问连续地址时自动合并为 128B 事务。

**结果（A6000, 4096²）：**GMEM 吞吐量从 **15 → 110 GB/s**（7\.3×），性能从 **309 → 1986 GFLOPs**（6\.4×，cuBLAS 的 8\.5%）。仅此一步就超越了 2015 年优化 CPU BLAS 的性能。

```cuda
template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
  const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
  // ✅ threadIdx.x 连续 → 同一 Warp 内线程访问连续 GMEM 地址

  if (cRow < M && cCol < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[cRow * K + i] * B[i * N + cCol];
    }
    C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
  }
}
```

## 3\.4 合并访存检查清单

||检查项|参考|
|---|---|---|
|□|同一 Warp 内线程是否访问连续全局内存地址？|§3\.1|
|□|访存是否对齐到 128B 边界？|§3\.3|
|□|是否使用 ncu 验证 GMEM throughput 接近峰值带宽？|§3\.2|
|□|threadIdx 映射方式是否让 x 维连续？|§3\.3|

---

# 4\. 共享内存缓存分块（Shared Memory Cache\-Blocking）

**TL;DR：**Kernel 2 每次迭代仍从全局内存重复加载相同数据。利用共享内存（On\-Chip、延迟 \~20\-30 cycles、带宽 \~12 TB/s），每个 Block 将 A 和 B 的当前 Tile 加载到 SMEM，然后在内循环中复用这些数据。性能从 1986 GFLOPs 提升至 2980 GFLOPs（1\.5x），但 SMEM 访存成为新瓶颈。

本章递进链：共享内存物理特性 → Kernel 3 SMEM 分块设计 → \_\_syncthreads 同步 → 占用率分析。

## 4\.1 共享内存物理特性（Shared Memory）

共享内存（SMEM）是 GPU 片上与 L1 Cache 共享的 SRAM 区域。物理上每 SM 一块（A6000: 100KB 可配置划分），逻辑上按 Block 分区（每 Block 最多 48KB \+ 1KB CUDA Runtime 开销）。

**关键性能指标（Volta 实测，Ampere 类似）：**全局内存带宽 \~750 GB/s vs 共享内存带宽 **\~12,080 GB/s（\~16×）**。延迟方面：GMEM \~200\-800 cycles，SMEM **\~20\-30 cycles**。这意味着 SMEM 不仅是"更快的内存"，更是一个可以**精确控制**的软件管理 Cache——程序员决定何时加载、何时淘汰，无 Cache Miss 的不确定性。

**Block 内共享：**同一 Block 的所有线程可以读写同一块 SMEM，这是线程间协作和计算分块的物理基础。后续所有 Tiling 优化都依赖 SMEM 作为 GMEM 和 Register 之间的中间层。

## 4\.2 Kernel 3：SMEM 缓存分块实现（Kernel 3: SMEM Cache\-Blocking）

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=OTg3YmU2N2UyMDJjNGJkOGFkYmE4OTMzMWZhY2U5YjNfNTk3NzI4ZDYzZjNiMjRjMDRmMzE4ZDRlNGU2OWQ5MDRfSUQ6NzY1ODU4NTc3MjgyODA4NTQ4NF8xNzg0ODk5ODI4OjE3ODQ5ODYyMjhfVjM)

```cuda
__shared__ float As[BLOCKSIZE * BLOCKSIZE];  // ✅ 片上高速缓存
__shared__ float Bs[BLOCKSIZE * BLOCKSIZE];
float tmp = 0.0;
for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
  // ① 协作加载 A/B 子块到 SMEM（合并访存）
  As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
  Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
  __syncthreads();  // ② 确保加载完成
  A += BLOCKSIZE; B += BLOCKSIZE * N;
  // ③ 对 SMEM 子块做点积
  for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx)
    tmp += As[threadRow * BLOCKSIZE + dotIdx] *
           Bs[dotIdx * BLOCKSIZE + threadCol];
  __syncthreads();  // ④ 防止快线程提前覆盖 SMEM
}
```

Kernel 3 的核心思想：**每次从 GMEM 加载 A 和 B 的一个子块（BLOCKSIZE×BLOCKSIZE）到 SMEM，对整个子块做完部分点积后再加载下一个子块。**

**分块策略：**外层循环沿 K 维步进（bkIdx \+= BLOCKSIZE），每步：① 所有线程协作加载 A 子块和 B 子块到 SMEM；② **\_\_syncthreads\(\)** 确保加载完成；③ 对该子块做 dotIdx 内层循环计算部分和；④ **\_\_syncthreads\(\)** 确保计算完成再覆盖 SMEM。两次 barrier 是关键——缺少会导致数据竞争。

**加载模式：**每个线程加载一对元素（As\[threadRow\*BK\+threadCol\] 和 Bs\[threadRow\*BN\+threadCol\]），threadCol = threadIdx\.x % BLOCKSIZE 为连续索引——保证了加载时依然享受 **合并访存**。

**SMEM 使用量：**BLOCKSIZE=32 时：2×32×32×4B = **8KB**。远低于 48KB/Block 上限。编译器输出确认：37 registers, 8192 bytes smem。

## 4\.3 占用率计算实例（Occupancy Calculation）

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=OTQ3MmNkMGQ0YjM0NGU2OGQzNjhmZGE5NDY3MGIwMGJfOTJlZTZlYWYwZDc3OTg3ZmViZTc1OWUxNTE0MzI5NTNfSUQ6NzY0ODEwNTU1NjAzMjg4Mzg5MV8xNzg0ODk5ODI4OjE3ODQ5ODYyMjhfVjM)

Kernel 3 的资源使用（A6000, BLOCKSIZE=32, 1024 threads/block）：37 regs/thread, 8192B SMEM/block。三种限制分别计算：

**寄存器限制：**37 regs × 32 threads/warp = 1184 regs/warp → 按 256 粒度取整 → 1280 regs/warp × 32 warps/block = **40960 regs/block**。SM 共 65536 regs → 只能容纳 **1 Block/SM**。

**SMEM 限制：**8192 \+ 1024\(Runtime\) = 9216 B/block。100KB / 9216 ≈ **10\.8 → 10 Blocks**。

**线程限制：**1024/1536 → **1 Block**。

最紧张的资源是寄存器和线程，最终 Occupancy = 32 active warps / 48 max warps = **66%**。

**性能结果：**\~2980 GFLOPs（cuBLAS 的 12\.8%），比 Kernel 2 提升 1\.5×。但 profiler 显示 Stall MIO Throttle 严重——大部分周期卡在等待 SMEM 加载（LDS 指令），而非执行 FMA。真正的问题是**每个线程只算 1 个结果，SMEM 访问/FMA 比例过高**。解决方向：让每个线程算更多结果 → 第 5 章 Block Tiling。

## 4\.4 共享内存优化检查清单

||检查项|参考|
|---|---|---|
|□|是否用 SMEM 缓存了可复用的全局内存数据？|§4\.2|
|□|\_\_syncthreads 是否成对出现（加载后\+计算后）？|§4\.2|
|□|是否计算了当前 kernel 的 Occupancy？|§4\.3|
|□|SMEM 使用量是否限制 Block/SM 数量？|§4\.1|

---

# 5\. 提高算术强度 — Block Tiling（Block Tiling）

**TL;DR：**Kernel 3 每个线程只计算 1 个 C 元素，SMEM 访存/FMA 比例过高。让每个线程计算更多结果（TM 个/线程），复用 SMEM 加载的数据，提升算术强度。1D Blocktiling（TM=8）达到 8474 GFLOPs（2\.8x）；2D Blocktiling（TM=TN=8）每线程计算 64 个结果，达到 15971 GFLOPs（1\.9x），算术强度从 \~32 提升至 \~128 FLOPs/Byte。

本章递进链：每个线程算 1 个结果的瓶颈 → 1D Blocktiling 让线程沿 M 维算多个 → 2D Blocktiling 让线程沿 M×N 维算矩阵块 → 编译器自动优化的神奇效果。

## 5\.1 Kernel 4：1D Blocktiling（Kernel 4: 1D Blocktiling）

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZDRiNzE1NmUzNTVlYzhkY2FmNWU5M2MzNjgyNGIxY2FfODkzNDNmOGJmMjVhMTc3M2M3ZDBjYzVkNTBhZTBmNzJfSUQ6NzY1ODYwMjk3MDI2Njg5NzM2Nl8xNzg0ODk5ODI4OjE3ODQ5ODYyMjhfVjM)

```cuda
// 参数: BM=64, BN=64, BK=8, TM=8
// 启动: dim3 blockDim((BM*BN)/TM);  // 每个线程算 TM 个结果

// 线程寄存器缓存 — 替代每次从 SMEM 重新加载
float threadResults[TM] = {0.0};

// 外层循环：沿 K 维遍历 Block Tile
for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
  // 协作加载 A/B 子块到 SMEM（与 Kernel 3 相同）
  As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
  Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
  __syncthreads();

  A += BK; B += BK * N;

  // ✅ dotIdx 外循环 — 让 Btmp 可被 TM 次 FMA 复用
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    float Btmp = Bs[dotIdx * BN + threadCol];  // 寄存器缓存 Bs 元素
    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
      threadResults[resIdx] +=
          As[(threadRow * TM + resIdx) * BK + dotIdx] * Btmp;
    }
  }
  __syncthreads();
}
// 写回 TM 个结果到 C
```

Kernel 3 的瓶颈在 SMEM 访存：每个线程只算 1 个 C 元素，但要在内循环中反复加载 As 和 Bs。1D Blocktiling 的思路：**让每个线程沿 M 维计算 TM=8 个结果，把 Bs 的同一元素复用 TM 次。**

**核心代码变化：**线程现在缓存 threadResults\[TM\] 在寄存器中；dotIdx 内循环中，先将 Bs\[dotIdx\*BN \+ threadCol\] 缓存到 **float Btmp**（寄存器），然后对 TM 个结果分别做 FMA——Btmp 被复用 TM 次，As 加载也从 1 次变成了 TM 次但因 BM×BK 更大而有更多复用。

**访存量对比（per result）：**Kernel 3: K/16 GMEM \+ K×2 SMEM。Kernel 4: K/32 GMEM \+ K×9/8 SMEM。SMEM 访问量从 2K 降至 **1\.125K**（降低 44%），这就是性能提升的来源。

**性能：**8474 GFLOPs（cuBLAS 的 36\.5%），相比 Kernel 3 提升 **2\.8×**。参数：BM=64, BN=64, BK=8, TM=8。

## 5\.2 Kernel 5：2D Blocktiling（Kernel 5: 2D Blocktiling）

```cuda
// 参数: BM=BN=128, BK=8, TM=TN=8
// 启动: dim3 blockDim((BM*BN)/(TM*TN));

// ✅ stride 多行加载 — 每个线程负责多行，保证 Tile 全覆盖
const uint strideA = numThreadsBlocktile / BK;
const uint strideB = numThreadsBlocktile / BN;

float threadResults[TM * TN] = {0.0};
float regM[TM] = {0.0};  // 寄存器缓存 As 的 TM 个元素
float regN[TN] = {0.0};  // 寄存器缓存 Bs 的 TN 个元素

for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
  // 多行加载 A → SMEM
  for (uint offset = 0; offset < BM; offset += strideA)
    As[(innerRowA + offset) * BK + innerColA] =
        A[(innerRowA + offset) * K + innerColA];
  // 多行加载 B → SMEM
  for (uint offset = 0; offset < BK; offset += strideB)
    Bs[(innerRowB + offset) * BN + innerColB] =
        B[(innerRowB + offset) * N + innerColB];
  __syncthreads();

  A += BK; B += BK * N;

  // ✅ dotIdx 外循环 → regM/regN 寄存器缓存 → TM×TN 次 FMA
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    for (uint i = 0; i < TM; ++i)
      regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
    for (uint i = 0; i < TN; ++i)
      regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
    for (uint resM = 0; resM < TM; ++resM)
      for (uint resN = 0; resN < TN; ++resN)
        threadResults[resM * TN + resN] += regM[resM] * regN[resN];
  }
  __syncthreads();
}
```

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NmY0MmIzZjk2MjM5ZjE0MjY1ZTU0ZWFkMzFmZTZiMjJfZjEyYjFlZGQxOGMwYjZiYjFlMTM4YzczNjcwOTY2MDZfSUQ6NzY1ODU4OTY0MTU4MjcwOTk5OV8xNzg0ODk5ODI4OjE3ODQ5ODYyMjhfVjM)

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YWQ2ODU3MjM4ZjFmZmU4MzM5MTJjNzBkNDIxZjI4ZmVfMTZkYjY0ZTNiZmJiODE2NjA0OTY5MTk2MzUxNDQ2NDhfSUQ6NzY1ODU4OTcwMTAzNjk4NTU3NV8xNzg0ODk5ODI4OjE3ODQ5ODYyMjhfVjM)

1D Blocktiling 只沿 M 维复用，2D Blocktiling **同时沿 M 和 N 维扩展**，每线程计算 TM×TN=8×8=**64 个结果**，进一步提升算术强度。

**SMEM 加载方式变化：**不再是一个线程加载一对元素，而是每个线程循环加载多行（strideA/strideB 步进），确保所有 BM×BK 和 BK×BN 元素都被覆盖。

**计算核心：**dotIdx 外循环 → 每步先将 A 的 TM 个值和 B 的 TN 个值加载到 **regM\[TM\] 和 regN\[TN\]** 寄存器 → 然后做 TM×TN 次外积 FMA。这使得 SMEM 访问量从 LDS\+FMA 交替变成了 **"加载寄存器 → 大量 FMA → 下一个 dotIdx"** 的模式，极大地减少了 LDS 指令占比。

**访存量对比（per result）：**Kernel 5: K/64 GMEM \+ K/4 SMEM。相比 Kernel 3 的 2K SMEM，降至 0\.25K（**降低 87\.5%**）。

**性能：**15971 GFLOPs（cuBLAS 的 68\.7%），比 Kernel 4 提升 1\.9×。参数：BM=BN=128, BK=8, TM=TN=8。

## 5\.3 编译器自动优化（Compiler Optimizations）

一个有趣的发现：如果按照直觉写双重嵌套（内层 dotIdx → 外层 resIdx），PTX 会显示 TM×BK×2 = 128 次 LDS 指令。但**NVCC 编译器会自动做两件事**：① 将已知迭代次数的循环**完全展开**（BK=TM=TN=8 均为编译时常量）；② **消除重复的 SMEM loads**——Bs 的同一元素只加载一次，在 TM 次 FMA 中复用。最终实际的 LDS 次数和手动优化过的版本完全一样。

**SASS 层面的自动向量化：**Bs 的 SMEM 加载被合并为 **LDS\.128**（128\-bit 一次加载 4 个 float），As 的加载仍是 32\-bit **LDS**。这是因为 Bs 在 SMEM 中是连续存储（dotIdx×BN \+ threadCol），而 As 当前布局不连续。下一章将通过**转置 As**解决这个问题。

参考：[Godbolt 编译器探索](https://godbolt.org/)可验证 NVCC 的循环展开和 LDS 消除行为。

## 5\.4 Block Tiling 检查清单

||检查项|参考|
|---|---|---|
|□|每个线程是否计算了多个 C 元素（TM \> 1）？|§5\.1|
|□|是否使用 2D Tiling（TM 和 TN 均 \> 1）最大化算术强度？|§5\.2|
|□|dotIdx 循环是否放在最外层以促进寄存器复用？|§5\.2|
|□|是否验证了 PTX/SASS 中循环被展开且 SMEM loads 被合并？|§5\.3|

---

# 6\. 向量化内存访问（Vectorized Memory Access）

**TL;DR：**Kernel 5 的 GMEM 加载仍使用 32\-bit 指令（LDG\.E），每个线程发 4 次独立的 load。通过 float4（128\-bit）向量化加载，将 GMEM 事务合并为 LDG\.E\.128，同时将 As 转置存储使 SMEM 读取也自动向量化为 LDS\.128。性能从 15971 GFLOPs 提升至 18237 GFLOPs（14%）。

本章递进链：向量化加载的必要性 → As 转置对齐 → float4 cast 的编译器语义 → GMEM Store 也向量化。

## 6\.1 float4 向量化加载（float4 Vectorized Load）

从 Kernel 5 的 SASS 可以看出：Bs 的 SMEM 加载是 **LDS\.128**（128\-bit 向量化），但 As 仍是 **LDS**（32\-bit），GMEM 加载是 **LDG\.E**（32\-bit）。Kernel 6 的核心目标：**把 GMEM 加载也变成 LDG\.E\.128，把 As 的 SMEM 加载也变成 LDS\.128。**

**实现方式：**使用 **float4** 类型做 **reinterpret\_cast**——将 float\* 指针强制转换然后一次解引用加载 4 个 float。编译器的关键在于：**reinterpret\_cast\<float4\*\> 承诺了 128\-bit 对齐**，使得编译器可以安全生成 LDG\.E\.128 指令。手动展开（写 4 行独立赋值）则编译器无法保证对齐，不会合并。

**为什么对齐这么重要？**GPU 的 LDG\.E\.128 指令要求源地址 128\-bit（16B）对齐。矩阵的行首地址（A \+ row\*K）不一定对齐，但在大多数情况下实际是对齐的。reinterpret\_cast 相当于告诉编译器："我保证对齐，用 128\-bit 指令"。

## 6\.2 As 转置优化（As Transpose）

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NjZkOWJmMTM2MWU3OWE2M2M0NTYwMWMzOGQ3MDhlOGJfYzI1NzAzZjA1YmRlZDVmNTI5MTcwOGM5ODlhMWFhMjVfSUQ6NzY1ODU4OTc1MzA2MzI0NzA4OF8xNzg0ODk5ODI4OjE3ODQ5ODYyMjhfVjM)

As 的 SMEM 存储格式需要转置的原因：在 dotIdx 内循环中，线程访问 As\[dotIdx\*BM \+ threadRow\*TM \+ i\]，即沿 BM 方向跳跃。这导致同一 Warp 内线程访问的 As 地址不连续 → **无法被硬件自动合并为 LDS\.128**。

**解决方案：**在 GMEM→SMEM 加载时同步转置：float4 tmp 从 A 加载后，4 个 float 分别写入 As 的 4 个不同行而非连续列——即 **As\[\(innerColA\*4\+0\)\*BM \+ innerRowA\] = tmp\.x**。这样在 dotIdx 循环中，线程访问的是 **As\[dotIdx\*BM \+ contiguous\_range\]**，地址连续 → 硬件自动生成 LDS\.128。

**代价：**加载时多了一些索引计算和分散写入。但 SMEM 写入速度远快于 SMEM 读取在关键路径上的影响，且只在每次外层迭代（K/BK 次）做一次转置写入，换取内层循环中所有 LDS 都变成 128\-bit。

代码对比：**❌ 转置前**：As\[threadRow\*BK \+ innerColA\*4\+N\] → LDS（32\-bit 分散）；**✅ 转置后**：As\[dotIdx\*BM \+ threadRow\*TM \+ i\] → LDS\.128（连续合并）。

## 6\.3 Kernel 6：完整向量化实现（Kernel 6: Full Vectorization）

```cuda
// 每个线程加载 128bit/32bit = 4 个元素
const uint innerRowA = threadIdx.x / (BK / 4);
const uint innerColA = threadIdx.x % (BK / 4);

for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
  // ✅ float4 从 A 加载 128-bit (LDG.E.128)，存储时同步转置
  float4 tmp = reinterpret_cast<float4*>(&A[innerRowA * K + innerColA * 4])[0];
  As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
  As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
  As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
  As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;

  // ✅ float4 从 B 加载 (LDG.E.128)，B 不需要转置（列存储天然连续）
  reinterpret_cast<float4*>(&Bs[innerRowB * BN + innerColB * 4])[0] =
      reinterpret_cast<float4*>(&B[innerRowB * N + innerColB * 4])[0];
  __syncthreads();

  A += BK; B += BK * N;

  // dotIdx 循环中，As 转置后读取连续 → LDS.128 自动向量化
  for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    for (uint i = 0; i < TM; ++i)
      regM[i] = As[dotIdx * BM + threadRow * TM + i];  // ✅ 连续 → LDS.128
    for (uint i = 0; i < TN; ++i)
      regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
    for (uint resM = 0; resM < TM; ++resM)
      for (uint resN = 0; resN < TN; ++resN)
        threadResults[resM * TN + resN] += regM[resM] * regN[resN];
  }
  __syncthreads();
}

// ✅ C 写回也用 float4 (STG.E.128)
for (uint resM = 0; resM < TM; ++resM) {
  for (uint resN = 0; resN < TN; resN += 4) {
    float4 tmp = reinterpret_cast<float4*>(
        &C[(threadRow * TM + resM) * N + threadCol * TN + resN])[0];
    tmp.x = alpha * threadResults[resM * TN + resN + 0] + beta * tmp.x;
    tmp.y = alpha * threadResults[resM * TN + resN + 1] + beta * tmp.y;
    tmp.z = alpha * threadResults[resM * TN + resN + 2] + beta * tmp.z;
    tmp.w = alpha * threadResults[resM * TN + resN + 3] + beta * tmp.w;
    reinterpret_cast<float4*>(
        &C[(threadRow * TM + resM) * N + threadCol * TN + resN])[0] = tmp;
  }
}
```

Kernel 6 集成了 float4 向量化 \+ As 转置 \+ C 写回向量化，性能从 15971 GFLOPs 提升至 **18237 GFLOPs**（14% 提升）。具体的向量化效果：GMEM 加载：LDG\.E（32\-bit）→ **LDG\.E\.128**。As SMEM 加载：LDS（32\-bit）→ **LDS\.128**。GMEM 写回：STG\.E → **STG\.E\.128**C 矩阵的写回也用 float4，减少写事务。

**完整数据流：**① float4 从 GMEM 加载 A → 转置写入 As（SMEM）；② float4 从 GMEM 加载 B → 直接写入 Bs（SMEM，B 不需要转置因为列存储天然连续）；③ \_\_syncthreads；④ dotIdx 循环：LDS\.128 从 As/Bs 加载到 regM/regN → FMA；⑤ float4 从寄存器写回 C（GMEM）。

**Profiler 验证：**Nsight Compute 中 LDG\.E\.128 和 STG\.E\.128 指令占比显著增加，GMEM 吞吐量进一步提升。但此时新的瓶颈浮现：**shared memory bank conflicts**——下一章的主题。

## 6\.4 向量化检查清单

||检查项|参考|
|---|---|---|
|□|GMEM 加载是否使用 float4（128\-bit）？|§6\.1|
|□|As 是否转置存储以保证 SMEM 读取连续？|§6\.2|
|□|GMEM 写回是否也向量化？|§6\.3|
|□|是否在 SASS 中确认 LDG\.E\.128 和 LDS\.128 指令？|§6\.1|

---

# 7\. 共享内存 Bank Conflict（Shared Memory Bank Conflicts）

**TL;DR：**SMEM 由 32 个 Bank 组成，同一 Warp 内多线程访问同一 Bank 的不同地址会导致 Bank Conflict（串行化）。Kernel 7 通过地址线性化重排 Bs 的存储布局消除 Conflict；Kernel 8 用更简单的"额外列填充"方案（BN\+5 列）。两种方法均 \~16\.2\-16\.4 TFLOPs，比未优化略有改善，但收益不如其他优化显著。

本章递进链：Bank Conflict 原理 → Kernel 6 的 Bank Conflict 检测 → 两种消除方案对比 → 为什么收益有限。

## 7\.1 共享内存 Bank 结构（SMEM Bank Structure）

SMEM 被物理组织为 **32 个 Bank**，每个 Bank 4 字节宽。每个时钟周期，每个 Bank 可以服务一个地址。关键规则：**同一 Warp 内多个线程访问同一 Bank 的不同地址 → Bank Conflict → 串行化访问**（n\-way conflict 意味着需要 n 个周期）。同一 Bank 的相同地址则触发广播机制（无 Conflict）。

**Bank 映射公式：**Bank\# = \(byte\_address / 4\) % 32。对于 float 数组（4B），Bank\# = index % 32。这意味着：**步长恰好为 32 的访问模式会产生严重的 32\-way Conflict**——每 32 个 float 就会回到同一个 Bank。

**Kernel 6 的 Bank Conflict 来源：**Bs 的大小是 BK×BN = 8×128 = 1024 floats，其中每一行有 128 个 float = 32×4。当 dotIdx 循环中线程访问 Bs\[dotIdx\*BN \+ threadCol\*TN \+ i\] 时，TN=8、BN=128 → 连续线程的地址差为 TN×4B = 32B = **8 个 Bank**。虽然不会 32\-way，但仍有可能出现 2\-way 或 4\-way Conflict，具体取决于线程索引和 TN 值。ncu 中 "shared\_store\_transactions" 和 "shared\_load\_transactions" 指标可以量化 Conflict 严重程度。

## 7\.2 Kernel 7：地址线性化（Kernel 7: Address Linearization）

```cuda
// ❌ 线性存储：Bs[row * BN + col]
//    当 BN=128, TN=8 时，连续线程 col 差 8 → bank 差 8
//    32 个线程覆盖 32×8=256 floats → 某些 bank 被多次访问 → Conflict

// ✅ 线性化存储公式（打散 bank 分配）：
float4 tmp = reinterpret_cast<float4*>(&B[innerRowB * N + innerColB * 4])[0];
// 将 B 的每行按交织模式写入 Bs，使连续线程映射到不同 bank
Bs[((innerColB % 2) * 4 + innerRowB * 8 + 0) * 16 + innerColB / 2] = tmp.x;
Bs[((innerColB % 2) * 4 + innerRowB * 8 + 1) * 16 + innerColB / 2] = tmp.y;
Bs[((innerColB % 2) * 4 + innerRowB * 8 + 2) * 16 + innerColB / 2] = tmp.z;
Bs[((innerColB % 2) * 4 + innerRowB * 8 + 3) * 16 + innerColB / 2] = tmp.w;

// 对应的读取公式（dotIdx 循环中）：
for (uint i = 0; i < TN; ++i)
  regN[i] = Bs[(dotIdx * 8 + i) * 16 + threadCol];  // 交织读取 → bank 均匀分布

```

Kernel 7 用**地址线性化**策略消除 Bs 的 Bank Conflict：将 Bs 的二维索引 \(row, col\) 映射到一个**非线性函数**，使得同一 Warp 内线程访问的不同 col 落在不同 Bank 上。

**核心公式：**Bs 存储时：Bs\[\(\(innerColB%2\)\*4 \+ innerRowB\*8 \+ n\)\*16 \+ innerColB/2\] = tmp。读取时：Bs\[\(dotIdx\*8 \+ i\)\*16 \+ threadCol\]。这个复杂的索引变换的效果是将 Bs 的逻辑布局从"行优先"转换成"交织（Interleaved）"模式，使连续线程映射到不同 Bank。

**代价：**索引计算更复杂，代码可读性下降。收益：Bank Conflict 基本消除。但由于额外计算开销，在 A6000 上实测性能仅为 **16213 GFLOPs**，甚至略低于 Kernel 6（18237 GFLOPs）——这说明在当前瓶颈结构下，Bank Conflict 的消除收益被索引开销抵消了。

## 7\.3 Kernel 8：额外列填充（Kernel 8: Extra Column Padding）

```cuda
// ✅ 核心改动：Bs 声明时多加 5 列，改变每行的 bank 偏移
const int extraCols = 5;
__shared__ float Bs[BK * (BN + extraCols)];  // vs 原来的 BK*BN

// 存储时 (col + 5 偏移使相邻行同一列号映射到不同 bank)
tmp = reinterpret_cast<float4*>(&B[innerRowB * N + innerColB * 4])[0];
Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 0] = tmp.x;
Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 1] = tmp.y;
Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 2] = tmp.z;
Bs[innerRowB * (BN + extraCols) + innerColB * 4 + 3] = tmp.w;

// 读取时同样加 extraCols 偏移
for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
  for (uint i = 0; i < TN; ++i)
    regN[i] = Bs[dotIdx * (BN + extraCols) + threadCol * TN + i];
  ...
}

// 对比 Kernel 7: 代码几乎只需搜索替换 BN → (BN+extraCols)
// 性能: 16459 vs 16213 GFLOPs, 轻微优势
// 代价: 多占 5*BK*4B = 160B SMEM (可忽略)
```

Kernel 8 用更简单的方法：**在 Bs 的每行末尾加 5 个额外列**（BN → BN\+5），改变行宽使同一列号在不同行映射到不同 Bank。原理：Bank\# = \(row\*\(BN\+5\) \+ col\) % 32。当 row 变化时，额外的 5 列产生偏移，打散 Bank 分配。

**实现：**只需声明 **\_\_shared\_\_ float Bs\[BK \* \(BN \+ 5\)\]**，加载和读取时的索引都改为 **innerRowB \* \(BN \+ 5\) \+ \.\.\.**。代码改动量极小。

**与 Kernel 7 对比：**✅ Kernel 8 更简洁（\+5 列 vs 复杂非线性映射），性能略高（16459 vs 16213 GFLOPs）。❌ 浪费 5×BK×4B 的 SMEM 空间。两者的 **共同问题**：在当前优化阶段（\~16\-18 TFLOPs），Bank Conflict 不是主要瓶颈——SMEM 访存/FMA 比例才是。这就是为什么消除 Bank Conflict 只带来微小的性能提升（\~1%），而 Block Tiling（第 5 章）和向量化（第 6 章）的收益更大。

## 7\.4 Bank Conflict 检查清单

||检查项|参考|
|---|---|---|
|□|是否使用 ncu 检查 shared memory bank conflicts 指标？|§7\.1|
|□|Bs 的列数是否为 32 的倍数（导致 Conflict）？|§7\.2|
|□|是否通过地址线性化或 Padding 消除了 Conflict？|§7\.2\-7\.3|
|□|Padding 方案是否比原生方案有可测量的性能提升？|§7\.3|

---

# 8\. 参数自动调优（Autotuning）

**TL;DR：**Kernel 6 累积了 5 个模板参数（BM, BN, BK, TM, TN），手动猜测最优组合不现实。通过脚本遍历 \~400 组合参数空间，在 A6000 上发现 BM=BN=128, BK=16, TM=TN=8 最优，性能从 18237 GFLOPs 提升至 19721 GFLOPs（\~5%）。最优参数因 GPU 型号而异——A100 上相同配置反而不佳。

本章递进链：参数空间定义 → 约束条件筛选 → 自动调优脚本 → 跨 GPU 结果对比。

## 8\.1 超参数空间（Hyperparameter Space）

到 Kernel 6 为止，已累积 **5 个模板参数**：BM, BN — Block Tile 的 M/N 维度（决定 SMEM 用量和 Block 大小）；BK — Block Tile 的 K 维度（决定每次 SMEM 加载量）；TM, TN — 每线程计算的 M/N 子块大小（决定算术强度）。

**有效组合约束：**向量化要求每个线程每次加载 4 个 float → BM×BK 和 BK×BN 必须能被 **4 × NUM\_THREADS** 整除。BM×BN/\(TM×TN\) = NUM\_THREADS \(每 Block 的线程数等于结果总数/每线程结果数\)。BK 需要适中：太小则外层循环多 \+ SMEM 利用率低；太大则 SMEM 不够用。

**搜索空间：**BM ∈ \{64, 128\}，BN ∈ \{64, 128\}，BK ∈ \{8, 16\}，TM ∈ \{4, 8\}，TN ∈ \{4, 8\}。经过约束过滤后约 **\~400 个有效组合**。每个组合编译成一个独立 Kernel 实例（模板实例化），运行时测试 4096² 矩阵的性能。

## 8\.2 Kernel 9：自动调优实现（Kernel 9: Autotuned）

```cuda
// 五模板参数: BM, BN, BK, TM, TN (通过 autotuner 确定最优值)
// A6000 最优: BM=BN=128, BK=16, TM=TN=8

// ✅ Warptile 分段计算（预演 Warp Tiling）
constexpr int WM = TM * 16;            // 每个 warp 沿 M 维计算 128 个结果
constexpr int WN = TN * 16;            // 每个 warp 沿 N 维计算 128 个结果
constexpr int WMITER = CEIL_DIV(BM, WM);  // Block Tile 内 Warptile 的迭代次数
constexpr int WNITER = CEIL_DIV(BN, WN);

// ✅ 动态 rowStride — 适配任意 BK/BN 组合
constexpr uint rowStrideA = (K9_NUM_THREADS * 4) / BK;
constexpr uint rowStrideB = K9_NUM_THREADS / (BN / 4);

// SMEM 加载循环 — rowStride 保证覆盖整个 Tile
for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
  float4 tmp = reinterpret_cast<float4*>(
      &A[(innerRowA + offset) * K + innerColA * 4])[0];
  As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
  // ... 转置写入 tmp.y, tmp.z, tmp.w
}
// Bs 加载同理，用 rowStrideB

// ✅ 三重 FMA 循环：Warptile → dotIdx → Thread Tile
for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx)
  for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx)
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // regM/regN 加载...
      for (uint rM = 0; rM < TM; ++rM)
        for (uint rN = 0; rN < TN; ++rN)
          threadResults[(wmIdx*TM+rM)*(WNITER*TN) + wnIdx*TN + rN] +=
              regM[rM] * regN[rN];
    }
```

**工程优化经验**：Kernel 9 在 Kernel 6 的基础上做了三处增强以适应任意参数组合：**Warptile 分段计算**：将 Block Tile 按 WM×WN 大小进一步划分为 Warptile（预演了第 9 章的 Warp Tiling），每个 Warp 迭代计算 WMITER×WNITER 个 Warptile。**动态 rowStride**：加载 SMEM 时的步长不再硬编码，而是通过 **rowStrideA = \(NUM\_THREADS \* 4\) / BK** 动态计算，使任何 BK 值都能正确覆盖整个 Tile。**\_\_launch\_bounds\_\_**：通过 **\_\_launch\_bounds\_\_\(NUM\_THREADS\)** 提示编译器线程数，帮助寄存器分配优化。

threadResults 数组大小也变为动态的 **\[WMITER \* WNITER \* TM \* TN\]**。代码的复杂度显著增加，但换来了对任意参数组合的正确性和高效性。

## 8\.3 跨 GPU 调优结果（Cross\-GPU Results）

自动调优的关键发现：**最优参数因 GPU 型号而异，不能跨平台通用。**

**实验结果：**A6000（Ampere, 84 SMs）：最优 BM=BN=128, BK=16, TM=TN=8 → **19721 GFLOPs**（cuBLAS 的 84\.8%），相比默认参数提升 \~5%。A100 SXM4 40GB（Ampere, 108 SMs）：最优 BM=BN=64, BK=16, TM=TN=4 → **12\.6 TFLOPs**，而用 A6000 参数只有 12\.0 TFLOPs（差 6%）。A100 的 fp32 吞吐不如 A6000（架构侧重不同：A100 的 Tensor Core 更强、fp32 CUDA Core 更少）。

**为什么不同？**SM 数量不同 → 最优 Occupancy 不同。Register File 和 SMEM 的 trade\-off 不同。Block Tile 大小影响 L2 Cache 利用率。这就是为什么 cuBLAS 包含 **数百个 Kernel 变体**，在运行时根据 GPU 型号 \+ 矩阵尺寸选择最优实现。

## 8\.4 自动调优检查清单

||检查项|参考|
|---|---|---|
|□|是否系统性地搜索了 BM/BN/BK/TM/TN 的合理组合？|§8\.1|
|□|参数组合是否满足向量化对齐约束？|§8\.1|
|□|是否为不同 GPU 型号分别调优？|§8\.3|
|□|是否验证了 Kernel 9 在不同矩阵尺寸下的正确性？|§8\.2|

---

# 9\. Warp Tiling — 三级 Tiling 层次（Warp Tiling）

**TL;DR：**在 Block Tiling 和 Thread Tiling 之间插入 Warp Tiling 层，显式利用 Warp 调度器的并行性。每个 Warp 负责计算 Block Tile 内的一块矩形子区域（Warptile），线程在 Warptile 内进一步细分。三级 Tiling（Block → Warp → Thread）将并行度显式映射到硬件层次，配合自动调优后性能从 \~19\.7 TFLOPs 提升至 21\.8 TFLOPs（cuBLAS 的 93\.7%）。

本章递进链：Warp 调度器硬件对应 → Warptile 设计（WM/WN/WMITER/WNITER）→ 三级 Tiling 数据流 → 性能随矩阵尺寸变化。

## 9\.1 Warp 调度器与 Warp Tiling 动机（Warp Scheduler \& Motivation）

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MWRiMDU4NjAwMWJhYTMxZDBhYzZjZWZmOTM4ZTdmNjRfMjliYzgyYTVmYmRkMmJiOTBmZjJmMmZhMGFhZjI4ZTFfSUQ6NzY1ODU4OTgwNzAxMDMxOTU2NV8xNzg0ODk5ODI4OjE3ODQ5ODYyMjhfVjM)

到此为止的优化层次：Block Tiling（不同 Block 在不同 SM 上并行）→ Thread Tiling（同一 Block 内每个线程独立计算）。这中间缺失了一个关键层次：**Warp**。

**为什么需要 Warp Tiling？**Warp 是 SM 内的调度单元（4 Schedulers/SM），同一 Warp 内线程有寄存器缓存局部性，Warp 间共享 SMEM。显式管理 Warp 级 Tiling 可以：① 减少 SMEM 访问——Warp 内的数据更多放在寄存器中；② 减少 Bank Conflict——Warp 内线程访问模式有规律可优化；③ 更好地映射到硬件调度层次。

**Warp Tiling 的分层设计：**Block Tile（BM×BN）→ 按 2D 网格划分给各 Warp → 每个 Warp 负责一个 **Warptile（WM×WN）** → 每个 Warp 内迭代多个 **Sub\-Warptile（WSUBM×WSUBN）** → 每个线程计算 TM×TN 个结果。这形成了 **Block → Warp → Sub\-Warp → Thread** 四级嵌套。

## 9\.2 Kernel 10：Warp Tiling 实现（Kernel 10: Warptiling）

```cuda
// 参数: BM=BN=128, BK=16, WM=WN=64, TM=8, TN=4, NUM_THREADS=128

// ✅ Warp 分解：按 warpIdx 定位每个 Warp 在 Block Tile 中的位置
const uint warpIdx = threadIdx.x / WARPSIZE;
const uint warpCol  = warpIdx % (BN / WN);  // Warp 在 N 维的索引
const uint warpRow  = warpIdx / (BN / WN);  // Warp 在 M 维的索引

// ✅ Sub-Warptile 划分
constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
constexpr uint WSUBM = WM / WMITER;  // 每个 Sub-Warptile 的 M 维度
constexpr uint WSUBN = WN / WNITER;  // 每个 Sub-Warptile 的 N 维度

// 线程在 Sub-Warptile 内的位置
const uint threadIdxInWarp  = threadIdx.x % WARPSIZE;
const uint threadColInWarp  = threadIdxInWarp % (WSUBN / TN);
const uint threadRowInWarp  = threadIdxInWarp / (WSUBN / TN);

// C 指针直接定位到 Warp 的输出区域
C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

// processFromSmem: dotIdx 外循环 → regM/regN 加载整 Warptile → 双重 FMA
for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
  // 加载 Warptile 对应的 As/Bs 到 regM[WMITER*TM], regN[WNITER*TN]
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
    for (uint i = 0; i < TM; ++i)
      regM[wSubRowIdx * TM + i] =
          As[dotIdx * BM + warpRow * WM + wSubRowIdx * WSUBM +
             threadRowInWarp * TM + i];
  // ... regN 加载同理 ...

  // FMA: WMITER×WNITER 个 Sub-Warptile, 每 Thread 算 TM×TN
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx)
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx)
      for (uint rM = 0; rM < TM; ++rM)
        for (uint rN = 0; rN < TN; ++rN)
          threadResults[(wSubRowIdx*TM+rM)*(WNITER*TN) + wSubColIdx*TN + rN] +=
              regM[wSubRowIdx*TM+rM] * regN[wSubColIdx*TN+rN];
}
```

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MGQ2MmEyNWFkNTcyZTA2YjkyZDNkNDliNzc3NzA3ODhfOTI0ODg4YmNhNzk2YjZjZWYzMjE5Mzg4MjE2ZDZkNjFfSUQ6NzY1ODU4OTg2MDEwNTk2NDc2N18xNzg0ODk5ODI4OjE3ODQ5ODYyMjhfVjM)

Kernel 10 的核心结构：**warpIdx = threadIdx\.x / WARPSIZE** → warpRow/warpCol 定位 Warp 在 Block Tile 中的位置 → threadIdxInWarp 定位线程在 Warp 内的位置 → WMITER/WNITER 控制 Sub\-Warptile 迭代。

**关键函数拆分：**loadFromGmem\(\)：使用 float4 向量化加载 A 和 B 到 SMEM，As 加载时同步转置。与 Kernel 6/9 的加载逻辑一致。processFromSmem\(\)：对当前 SMEM 内容做 Warptile 级计算。dotIdx 外循环 → 加载 regM\[WMITER\*TM\] 和 regN\[WNITER\*TN\] → 双重 FMA 循环计算 threadResults。

**参数示例（A6000 最优）：**BM=128, BN=128, BK=8, WM=64, WN=32, WMITER=2, WNITER=2, WSUBM=32, WSUBN=16, TM=8, TN=4。这意味着每个 Block 有 \(128×128\)/\(64×32\)=8 个 Warp，每个 Warp 计算 64×32 的子块。

**性能：**21779 GFLOPs（cuBLAS 的 **93\.7%**），是 12 个 Kernel 中性能最高的。从 Kernel 9 的 19721 GFLOPs 提升 \~10%。

## 9\.3 三级 Tiling 数据流（Three\-Level Tiling Dataflow）

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MzEyN2NjY2JhYjRhYTBiZWNjNDVjOWMzMTU4ZjIyZTBfYjQyYmU3ZTdjZmIzNjk0YjAzNmJhNjhjY2EzOGQzZTNfSUQ6NzY0ODEwNTY0MDYxNjU3ODI3Nl8xNzg0ODk5ODI4OjE3ODQ5ODYyMjhfVjM)

将所有优化层次放在一起看数据流：

**Level 1 — GMEM → SMEM（Block Tiling）：**完整 Block Tile（BM×BK \+ BK×BN）从 HBM 加载到 SMEM。所有 Block 内线程协作加载，float4 向量化（LDG\.E\.128），As 加载时转置。数据传输：每外层迭代一次。

**Level 2 — SMEM → Register（Thread Tiling）：**每个线程从其 Sub\-Warptile 对应的 SMEM 区域加载 TM×TN 个 float 到寄存器（regM/regN）。LDS\.128 向量化加载。数据复用：同一 Block 的多次内层迭代复用同一批 SMEM 数据。

**Level 3 — Register → FMA（ILP）：**regM\[TM\] 和 regN\[TN\] 之间做 TM×TN 次 FMA。所有数据已在寄存器中，延迟仅 \~4 cycles。数据复用：regM 的元素在 TN 次 FMA 中复用，regN 在 TM 次中复用。

**三级复用的量化效果：**以 TM=TN=8 为例——每 8×8=64 次 FMA 只需 8\+8=16 次 LDS 加载，LDS:FMA = 1:4。而 Naive Kernel 每次 FMA 需要 2 次 GMEM 加载（LDS:FMA 甚至不适用，因为根本没有 SMEM），这就是 \~1000× 的算术强度提升。

## 9\.4 Warp Tiling 检查清单

||检查项|参考|
|---|---|---|
|□|是否显式将 Block Tile 按 Warp 划分？|§9\.2|
|□|Warptile 大小是否匹配 Warp 调度器数量？|§9\.1|
|□|WMITER/WNITER 是否正确计算子 Warptile 步进？|§9\.2|
|□|是否通过调优找到最佳 WM/WN/TM/TN 组合？|§9\.4|

---

# 10\. 双缓冲 — 计算与访存重叠（Double Buffering）

**TL;DR：**之前的 Kernel 串行执行"加载 GMEM → 计算 → 加载下一块"，计算单元在加载期间空闲。双缓冲通过分配双倍 SMEM（前后台缓冲区），让一半线程加载下一块数据的同时另一半线程计算当前块。Kernel 11 用手动同步实现双缓冲；Kernel 12 用 CUDA 11\+ 的 async memcpy（cuda::memcpy\_async）\+ cuda::barrier 实现更优雅的异步流水线。

本章递进链：串行加载\-计算的空闲气泡 → 手动双缓冲（Kernel 11）→ async memcpy \+ barrier（Kernel 12）→ Hopper 架构展望。

## 10\.1 计算与访存重叠的原理（Computation\-Memory Overlap）

之前所有 Kernel 的共同模式：**串行交替**——加载 GMEM → \_\_syncthreads → 计算 → \_\_syncthreads → 加载下一块 → \.\.\.。这意味着在加载期间，计算单元**完全空闲**（反之亦然）。

**双缓冲（Double Buffering）**的思想：分配双倍 SMEM（Buffer 0 和 Buffer 1），让"加载下一块"和"计算当前块"**同时进行**。当计算单元在 Buffer 0 上工作时，内存单元在 Buffer 1 上加载下一批数据；下一轮交换。这本质上是将**串行流水线变成并行流水线**，目标是消除内存加载带来的计算停顿。

**前提条件：**SMEM 足够容纳两份缓冲区（如 2×BM×BK \+ 2×BK×BN）。这在较小 Tile 尺寸时是可行的。需要将线程分为两组（Loader 和 Computer），或使用异步加载指令（cuda::memcpy\_async）让所有线程既加载又计算。

## 10\.2 Kernel 11：手动双缓冲（Kernel 11: Manual Double Buffering）

```cuda
// ✅ 双倍 SMEM 分配：前后台缓冲区
__shared__ float As[2 * BM * BK];  // Buffer 0 + Buffer 1
__shared__ float Bs[2 * BK * BN];

// ✅ 线程按 ID 分组：前一半=加载+计算 B0，后一半=加载 B1
bool doubleBufferIdx = threadIdx.x >= (NUM_THREADS / 2);

// 加载索引只用一半线程数计算（因为两组分开加载不同 Buffer）
const uint innerRowA = (threadIdx.x % (NUM_THREADS / 2)) / (BK / 4);
// ...

if (doubleBufferIdx == 0) {
  // 组 0: 加载 B0
  loadFromGmem(..., As, Bs, ...);  // → Buffer 0
}
__syncthreads();

// ✅ 外层循环以 2*BK 步进，每次处理两个 Buffer
for (uint bkIdx = 0; bkIdx < K; bkIdx += 2 * BK) {
  if (doubleBufferIdx == 0) {
    processFromSmem(..., As, Bs, ...);      // 计算 Buffer 0
    __syncthreads();
    if (bkIdx + BK < K)
      processFromSmem(..., As+BM*BK, Bs+BK*BN, ...); // 计算 Buffer 1
    __syncthreads();
    if (bkIdx + 2*BK < K)
      loadFromGmem(..., A+2*BK, B+2*BK*N, As, Bs, ...); // 加载下一轮 B0
  } else {
    // 组 1: 对称的逻辑，先加载 B1 再计算 B0→B1
    if (bkIdx + BK < K)
      loadFromGmem(..., A+BK, B+BK*N, As+BM*BK, Bs+BK*BN, ...);
    __syncthreads();
    processFromSmem(..., As, Bs, ...);
    __syncthreads();
    if (bkIdx + BK < K)
      processFromSmem(..., As+BM*BK, Bs+BK*BN, ...);
  }
  A += 2 * BK; B += 2 * BK * N;
}
```

Kernel 11 用 **doubleBufferIdx** 将线程分为两组（前一半/后一半），手动编排加载和计算的交替：

**策略：**doubleBufferIdx=0 的线程负责加载 B0 和计算；doubleBufferIdx=1 的线程负责加载 B1。外层循环以 **bkIdx \+= 2\*BK** 步进（每次处理两个 Buffer）。

**流程：**加载 B0 → \_\_syncthreads → 计算 B0 \+ 加载 B1（两组线程同时做不同事）→ \_\_syncthreads → 计算 B1 \+ 加载下一轮的 B0 → \.\.\.最后一个迭代只计算不加载。

**实际性能：**17278 GFLOPs（cuBLAS 的 74\.3%），**低于** Kernel 10 的 21779。原因是手动双缓冲的线程分工不均——一半线程做加载时另一半可能闲置，且额外的 \_\_syncthreads 增加了同步开销。这反直觉地说明：**在这个规模的 Kernel 上，简单的串行执行因为 Occupancy 已然足够，流水线的收益不如 Warp Tiling。**

## 10\.3 Kernel 12：Async Memcpy \+ Barrier（Kernel 12: Async Double Buffering）

```cuda
// ✅ CUDA 11+ 异步机制
#include <cooperative_groups.h>
#include <cuda/barrier>

auto block = cooperative_groups::this_thread_block();
__shared__ cuda::barrier<cuda::thread_scope::thread_scope_block> frontBarrier, backBarrier;
if (block.thread_rank() == 0) {
  init(&frontBarrier, block.size());  // 初始化 barrier (arrive=block.size)
  init(&backBarrier, block.size());
}
__syncthreads();

// 加载第一个 Tile (B0) 到 SMEM
loadFromGmem<...>(N, K, A, B, As, Bs, ..., (*frontBarrierPtr));
//  loadFromGmem 内部用 cuda::memcpy_async 异步拷贝：
//    cuda::memcpy_async(&As[...], &A[...],
//                       cuda::aligned_size_t<sizeof(float)>(sizeof(float)), barrier);

int As_offset = 0, Bs_offset = 0;

// ✅ 流水线循环：异步加载下一个 Tile 的同时计算当前 Tile
for (uint bkIdx = 0; bkIdx < K - BK; bkIdx += BK) {
  // 异步加载 next Tile (Buffer 1-As_offset)
  loadFromGmem<...>(N, K, A + BK, B + BK * N,
                     As + (1-As_offset)*BM*BK, Bs + (1-Bs_offset)*BK*BN,
                     ..., (*backBarrierPtr));

  (*frontBarrierPtr).arrive_and_wait();  // 等待当前 Tile 加载完成
  processFromSmem<...>(regM, regN, threadResults,
                        As + As_offset*BM*BK, Bs + Bs_offset*BK*BN, ...);

  A += BK; B += BK * N;
  As_offset = 1 - As_offset;  // ✅ 交换前后台 Buffer
  Bs_offset = 1 - Bs_offset;
  auto tmp = frontBarrierPtr;  // ✅ 交换 barrier
  frontBarrierPtr = backBarrierPtr;
  backBarrierPtr = tmp;
  __syncthreads();
}
// 最后一个 Tile: 只等 + 算
(*frontBarrierPtr).arrive_and_wait();
processFromSmem<...>(..., As + As_offset*BM*BK, Bs + Bs_offset*BK*BN, ...);
```

Kernel 12 用 CUDA 11\+ 的现代异步机制重写双缓冲：**cuda::memcpy\_async**：将 GMEM→SMEM 的拷贝提交为异步操作，不阻塞当前线程。配合 **cuda::barrier**（到达\-等待模型）：线程完成当前计算后 arrive 在 barrier 上，所有线程到达后交换 front/back barrier。

**核心流程：**初始化 frontBarrier 和 backBarrier。加载 B0 到 Buffer 0（异步）。循环：加载下一块到 Buffer 1（异步，backBarrier）→ **frontBarrier\.arrive\_and\_wait\(\)** 等待 B0 加载完成 → 计算 B0 → 交换 front/back barrier → 推进指针 → 下一个迭代。最后一个迭代：只需等待和计算，不需加载。

**与 Kernel 11 的对比：**Kernel 11 手动分配线程角色（一半加载一半计算），Kernel 12 所有线程统一执行加载\+计算，由硬件调度器自然重叠。Barrier 的使用也更安全——不需要手动 \_\_syncthreads 配对。但截至目前该 Kernel 的实测性能并未显著超越 Kernel 10，说明在当前规模下，**SMEM 带宽已经足够快，加载\-计算的重叠收益有限**。真正受益的场景是更大 Tile 或更小 K 维度（加载占比更高的情况）。

## 10\.4 双缓冲检查清单

||检查项|参考|
|---|---|---|
|□|SMEM 是否分配了双倍空间用于前后台缓冲？|§10\.2|
|□|加载和计算是否在不同缓冲区上并发执行？|§10\.2|
|□|是否使用 cuda::memcpy\_async 而非同步加载？|§10\.3|
|□|Barrier 的 arrive\_and\_wait 是否正确配对？|§10\.3|
|□|双缓冲是否带来了可测量的性能提升（ncu 验证 SMEM 吞吐量）？|§10\.1|

---

# 11\. 总结 — 定位、量化、优化、验证

**核心闭环：**定位瓶颈（ncu 采样 Stall 原因）→ 量化目标（Roofline 模型/Occupancy 计算）→ 应用技术（Tiling/向量化/双缓冲）→ 验证收益（GFLOPs 和 GMEM 吞吐量对比）。

本章回应导读章提出的三个核心问题，串联全文公式，并提供完整的性能对比总表。

## 11\.1 回答三个核心问题

**问题 1：如何通过逐层 Tiling 最大化算术强度？**核心策略是**数据复用金字塔**——每一层 Tiling 将数据拉到更快的存储，然后尽可能多地复用：GMEM→SMEM（Block Tiling，复用 BK 次计算）；SMEM→Register（Thread Tiling，复用 TM×TN 次 FMA）；Register→FMA（ILP，无额外加载）。从 Naive Kernel（AI \~0\.25 FLOPs/Byte）到 Kernel 10（AI \~245 FLOPs/Byte），增长了 **\~1000×**。

**问题 2：各级内存特征如何决定优化策略？**GMEM 带宽小、延迟高 → 必须合并访存（128B 事务）\+ 减少访问次数（Tiling）。SMEM 带宽大、延迟低、Bank 结构 → 精确控制数据布局（转置、Padding）以消除 Bank Conflict。Register 零延迟、线程私有 → 最终计算尽可能在寄存器中完成（regM/regN 缓存）。L2 Cache 被动管理 → 优化 Block Tile 大小的空间局部性。

**问题 3：每个优化步骤的瓶颈与收益？**见 §11\.2 性能全景表。最重要的三个步骤：合并访存（6\.4×）、2D Blocktiling（1\.9×）、Warp Tiling（1\.1×，但将性能推至 93\.7% cuBLAS）。Bank Conflict 消除和双缓冲的收益较小，说明在这些阶段瓶颈已从 SMEM 转向其他方面。

## 11\.2 性能演进全景

|Kernel|优化技术|GFLOPs/s|vs cuBLAS|关键参数/变化|
|---|---|---|---|---|
|1|Naive|309|1\.3%|2D Grid\+Block, 每线程 1 结果|
|2|GMEM Coalescing|1987|8\.5%|1D Block, BLOCKSIZE 映射, 15→110 GB/s|
|3|SMEM Caching|2980|12\.8%|\_\_shared\_\_ As/Bs, BLOCKSIZE=32, 8KB SMEM|
|4|1D Blocktiling|8475|36\.5%|TM=8, BM=BN=64, BK=8|
|5|2D Blocktiling|15972|68\.7%|TM=TN=8, BM=BN=128, 64 结果/线程|
|6|Vectorized Mem Access|18237|78\.4%|float4, As 转置, LDG\.128\+LDS\.128|
|7|Bank Conflict \(Linearize\)|16213|69\.7%|Bs 地址交织映射|
|8|Bank Conflict \(Padding\)|16459|70\.8%|Bs BK×\(BN\+5\), 额外列填充|
|9|Autotuning|19721|84\.8%|BM=128,BN=128,BK=16,TM=TN=8|
|10|Warptiling|**21779**|**93\.7%**|WM=64,WN=32, 三级 Tiling 层次|
|11|Double Buffering \(Manual\)|17278|74\.3%|双倍 SMEM, 线程分工|
|12|Double Buffering \(Async\)|—|—|cuda::memcpy\_async \+ barrier|
|cuBLAS|NVIDIA cuBLAS|**23250**|**100%**|500MB 二进制, 数百 Kernel 变体|

## 11\.3 关键公式串联

全文档的核心公式串联：**Roofline：**Perf ≤ min\(Peak FLOPs, BW × AI\)。Peak FLOPs = 30 TFLOPs \(A6000 fp32\)。BW × AI = 768 GB/s × AI。AI\<sub\>ridge\</sub\> = 39 FLOPs/Byte。→ 优化就是把 AI 从 0\.25 推到 245，从斜线区推入水平区。**Arithmetic Intensity：**AI = 2MNK / Bytes。Bytes = f\(Block Tile size, Thread Tile size, Double Buffering\)。→ 每个 Tiling 参数都直接影响 AI。**Occupancy：**Occ = min\(65536/regs\_per\_block, 100KB/smem\_per\_block, 1536/threads\_per\_block\) / 48 warps。→ 寄存器是 A6000 上最紧的限制。**Latency Hiding：**隐藏延迟需要的 Warp 数 = 内存延迟 / FMA 延迟 = 200/4 = 50 Warps → 接近 A6000 的 48 Warps/SM 最大值。→ 解释了为什么 \~66% Occupancy 就足够——不需要更多。**性能上限：**最终 GFLOPs = Peak FLOPs × 实际利用率，其中利用率受 Occupancy、Stall 周期、Bank Conflict、指令混合等因素影响。

## 11\.4 未覆盖的优化方向

以下优化方向在本项目中尚未实现，但值得探索：**Tensor Core / WMMA：**利用 A6000 的 Tensor Core（fp16/BF16/TF32）可获得 2\.5\-3\.5× 吞吐提升。但 fp32 SGEMM 的 Tensor Core 路径需要精度折中。**Split\-K：**cuBLAS 在中等矩阵尺寸（256\-512）时自动启用 Split\-K，沿 K 维拆分到多个 Block 并行计算后做 Reduction。提升小矩阵的 Occupancy。**Thread Swizzling：**优化 Block 分配到 SM 的顺序，提高 L2 Cache 命中率。本项目测试后无明显收益（L2 命中率已达 80%）。**Warp Specialization（Hopper\+）：**让部分 Warp 专职加载（使用更少寄存器），另一部分专职计算。配合 TMA（Tensor Memory Accelerator）直接 GMEM→SMEM 传输，跳过寄存器中转，降低寄存器压力。**Mixed Precision：**实际深度学习 workload 中，SGEMM（fp32）逐渐被 BF16/FP8 GEMM 取代，需要不同的优化路径。

## 11\.5 全文档检查清单汇总

||检查项|参考|
|---|---|---|
|□|是否理解 GPU 内存层次（GMEM/L2/SMEM/Register）及各级特征？|§2|
|□|能否用 Roofline 模型判断 kernel 瓶颈类型？|§2\.2|
|□|是否确保全局内存合并访存（Coalesced）？|§3|
|□|是否利用共享内存缓存可复用数据？|§4|
|□|是否通过多结果/线程提高算术强度？|§5|
|□|是否使用 float4 向量化 GMEM 和 SMEM 访存？|§6|
|□|是否检查并消除共享内存 Bank Conflict？|§7|
|□|是否为当前 GPU 自动调优了 Block/Warp/Thread 参数？|§8\-9|
|□|是否尝试双缓冲将计算与访存重叠？|§10|

---

# 12\. 参考文献

1. Boehm, S\. \(2022\)\. **How to Optimize a CUDA Matmul Kernel for cuBLAS\-like Performance: a Worklog\.** siboehm\.com\. [https://siboehm\.com/articles/22/CUDA\-MMM](https://siboehm.com/articles/22/CUDA-MMM) — 全文核心参考

2. Wang, Z\. \(2022\)\. **NVIDIA SGEMM Practice\.** GitHub\. [https://github\.com/wangzyon/NVIDIA\_SGEMM\_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE) — Benchmark 框架来源

3. NVIDIA\. \(2025\)\. **CUDA C\+\+ Programming Guide\.**[https://docs\.nvidia\.com/cuda/cuda\-c\-programming\-guide/](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — §5\.3 Max Memory Throughput; §3\.2\.2 Shared Memory

4. NVIDIA\. \(2025\)\. **CUDA C\+\+ Best Practices Guide\.**[https://docs\.nvidia\.com/cuda/cuda\-c\-best\-practices\-guide/](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/) — §9\.2 Coalesced Access; §9\.2\.2 Shared Memory

5. NVIDIA\. \(2025\)\. **Kernel Profiling Guide\.**[https://docs\.nvidia\.com/nsight\-compute/ProfilingGuide/](https://docs.nvidia.com/nsight-compute/ProfilingGuide/) — Warp State Metrics

6. Williams, S\., Waterman, A\., \& Patterson, D\. \(2009\)\. **Roofline: An Insightful Visual Performance Model for Multicore Architectures\.** Communications of the ACM, 52\(4\)\. [DOI: 10\.1145/1498765\.1498785](https://doi.org/10.1145/1498765.1498785) — §2\.2 Roofline 模型

7. Little, J\. D\. C\. \(1961\)\. **A Proof for the Queuing Formula: L = λW\.** Operations Research, 9\(3\)\. — §2\.5 Little's Law

8. Volkov, V\. \(2016\)\. **Understanding Latency Hiding on GPUs\.** Ph\.D\. Thesis, UC Berkeley\. [EECS\-2016\-143](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-143.html) — §2\.4 Occupancy

9. NVIDIA\. \(2022\)\. **CUTLASS: CUDA Templates for Linear Algebra Subroutines\.** GitHub\. [https://github\.com/NVIDIA/cutlass](https://github.com/NVIDIA/cutlass) — §9 Warp Tiling; §10 双缓冲

> （注：部分内容可能由 AI 生成）
