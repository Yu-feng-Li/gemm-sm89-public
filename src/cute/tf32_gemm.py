# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""独立的 SM89 CuTe DSL TF32 GEMM。

launch_gemm 和 launch_split_k 由主机按形状与配置实例化，复用同一个 GPU 主算子。
调用方提供连续行主序 FP32 A[M,K]、B[K,N]、C[M,N]，地址需 16 字节对齐。
输入在寄存器中舍入成 TF32，乘积以 FP32 累加，覆盖 C 的原值。
"""
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

CTA_TILE = (128, 256, 16)
WARPS = (2, 4, 1)
STAGES = 3


@cute.jit
def launch_gemm(a: cute.Pointer, b: cute.Pointer, c: cute.Pointer,
                shape: cutlass.Constexpr, stream: cuda.CUstream,
                tile: cutlass.Constexpr = None, warps: cutlass.Constexpr = WARPS,
                stages: cutlass.Constexpr = STAGES, cache_all: cutlass.Constexpr = False,
                raster_group: cutlass.Constexpr = 0, schedule: cutlass.Constexpr = "default",
                raster_m: cutlass.Constexpr = False):
    """在 stream 上覆盖 C，整个 K 由每个输出块的单个 CTA 累加。

    公共形状要求 M > 0、N 为 4 的正倍数、K >= 32 且为 16 的倍数。
    显式 tile 还须满足 launch_partitions 的 K 段和拷贝布局约束。
    """
    if cutlass.const_expr(tile is None):
        # tile=None 保留旧入口的方阵和矩形特例。常规 auto 由主机提供 tile。
        tile = CTA_TILE
        if cutlass.const_expr(shape == (4096, 4096, 4096)):
            raster_group = 1
            schedule = "copy_after"
        elif cutlass.const_expr(shape == (1024, 1024, 1024)):
            tile = (64, 128, 16)
            warps = (1, 4, 1)
            stages = 3
            raster_group = 0
            schedule = 'steady_copy'
        elif cutlass.const_expr(shape == (2048, 2048, 2048)):
            tile = (128, 256, 16)
            warps = (2, 4, 1)
            stages = 3
            raster_group = 0
            schedule = 'steady_copy'
        elif cutlass.const_expr(shape == (512, 5120, 17408)):
            tile = (128, 160, 32)
            warps = (4, 2, 1)
            stages = 2
            raster_group = 0
            schedule = 'copy_after'
        elif cutlass.const_expr(shape == (4096, 4096, 5120)):
            tile = (128, 256, 32)
            warps = (2, 4, 1)
            stages = 2
            raster_group = 0
            schedule = 'static_raster'
        elif cutlass.const_expr(shape == (4096, 4096, 17408)):
            tile = (128, 256, 32)
            warps = (2, 4, 1)
            stages = 2
            raster_group = 0
            schedule = 'static_raster'
        elif cutlass.const_expr(shape == (256, 256, 256)):
            tile = (32, 64, 16)
            warps = (1, 2, 1)
        elif cutlass.const_expr(shape == (128, 12288, 5120)):
            tile = (128, 96, 16)
            warps = (2, 2, 1)
    launch_partitions(a, b, c, shape, tile, warps, stages, 1, stream, cache_all, raster_group, schedule, raster_m)


@cute.jit
def launch_partitions(a: cute.Pointer, b: cute.Pointer, c: cute.Pointer,
                      shape: cutlass.Constexpr, tile: cutlass.Constexpr,
                      warps: cutlass.Constexpr, stages: cutlass.Constexpr,
                      split: cutlass.Constexpr, stream: cuda.CUstream,
                      cache_all: cutlass.Constexpr = False, raster_group: cutlass.Constexpr = 0,
                      schedule: cutlass.Constexpr = "default", raster_m: cutlass.Constexpr = False):
    """构造普通 GEMM 和 Split-K 共用的网格及内存布局。

    每个 CTA 的 warps[0] × warps[1] 个 warp 沿输出的 M/N 维分工，
    warp 内执行 16×8×8 TF32 MMA。每个 K 段至少有两个完整 BK 块。
    c 指向 C 或调用方分配的分段 FP32 输出，由 split 决定其布局。
    """
    m, n, k = shape
    bm, bn, bk = tile
    assert m > 0 and n > 0 and n % 4 == 0
    assert k % split == 0 and k // split >= 2 * bk and (k // split) % bk == 0
    part_k = k // split
    assert bk in (16, 32) and bm >= 16 and bn >= 32 and stages >= 2
    assert warps[2] == 1 and bm % (16 * warps[0]) == 0 and bn % (16 * warps[1]) == 0
    threads = 32 * warps[0] * warps[1]
    a_vectors = bk // 4
    b_vectors = math.gcd(128, bn) // 4
    # 拷贝的单次线程覆盖必须能完整嵌入 shared tile。
    assert bm % (threads // a_vectors) == 0 and bk % (threads // b_vectors) == 0, (
        "CTA tile is incompatible with the vector-copy thread layout"
    )
    mA = cute.make_tensor(a, cute.make_layout((m, part_k, split), stride=(k, 1, part_k if split > 1 else m*k)))
    # CuTe 用 (M,K) 和 (N,K) 表达 MMA 的两个操作数，让第二维都是 K。
    # B 的步长 (1,N) 使逻辑 B[j,kk] 仍指向 b[kk*N+j]，底层字节无须转置。
    mB = cute.make_tensor(b, cute.make_layout((n, part_k, split), stride=(1, n, n*part_k)))
    mC = cute.make_tensor(c, cute.make_layout((m, n, split), stride=(n, 1, m*n)))

    sA_layout = cute.tile_to_shape(cute.make_layout((8, 16), stride=(16, 1)),
                                  (bm, bk, stages), (0, 1, 2))
    b_atom_n = math.gcd(32, bn)
    sB_layout = cute.tile_to_shape(cute.make_layout((b_atom_n, 8), stride=(1, b_atom_n)),
                                  (bn, bk, stages), (0, 1, 2))
    sA_swizzle, sB_swizzle = cute.make_swizzle(2, 4, 3), cute.make_swizzle(2, 5, 2)
    # A 的连续轴是 K，B 的连续轴是 N。每次 cp.async 搬 4 个连续 FP32，
    # 因此 A/B 的线程布局和向量轴分别构造。shared 中的 swizzle 减少 bank 冲突。
    cache_mode = cute.nvgpu.LoadCacheMode.ALWAYS if cache_all else cute.nvgpu.LoadCacheMode.GLOBAL
    copy = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cache_mode),
        cutlass.Float32, num_bits_per_copy=128,
    )
    tiled_copy_A = cute.make_tiled_copy_tv(
        copy, cute.make_layout((threads // a_vectors, a_vectors), stride=(a_vectors, 1)), cute.make_layout((1, 4))
    )
    tiled_copy_B = cute.make_tiled_copy_tv(
        copy, cute.make_layout((b_vectors, threads // b_vectors), stride=(1, b_vectors)),
        cute.make_layout((4, 1)),
    )
    warp_layout = cute.make_layout(warps)
    tiled_mma = cute.make_tiled_mma(
        cute.nvgpu.warp.MmaTF32Op((16, 8, 8)), warp_layout,
        permutation_mnk=(16 * warps[0], 16 * warps[1], 8),
    )

    # 将组内 tile 编号放进 blockIdx.x 的低位，便于相近 CTA 复用 L2。
    # raster_m 沿 M 分组，此时相邻的 M tile 先复用 B。
    grid_m, grid_n = (m + bm - 1) // bm, (n + bn - 1) // bn
    grouped_tiles = grid_m if raster_m else grid_n
    outer_tiles = grid_n if raster_m else grid_m
    raster = 8 if grouped_tiles > 5 else 4 if grouped_tiles > 2 else 2 if grouped_tiles > 1 else 1
    raster = raster_group or raster
    tf32_kernel(
        mA, mB, mC, sA_layout, sA_swizzle, sB_layout, sB_swizzle,
        tiled_copy_A, tiled_copy_B, tiled_mma, raster, tile, stages, schedule, raster, raster_m,
    ).launch(grid=(outer_tiles * raster, (grouped_tiles + raster - 1) // raster, split),
             block=(threads, 1, 1), stream=stream, smem=None)


@cute.jit
def launch_split_k(a: cute.Pointer, b: cute.Pointer, c: cute.Pointer, partial: cute.Pointer,
                   shape: cutlass.Constexpr, tile: cutlass.Constexpr,
                   warps: cutlass.Constexpr, stages: cutlass.Constexpr,
                   split: cutlass.Constexpr, stream: cuda.CUstream,
                   cache_all: cutlass.Constexpr = False, raster_group: cutlass.Constexpr = 0,
                   schedule: cutlass.Constexpr = "default", raster_m: cutlass.Constexpr = False):
    """将分段乘积写到调用方提供的 partial，再覆盖 C。

    partial 的连续布局是 [split,M,N]，占 4*split*M*N 字节。
    CTA 的 blockIdx.z 选 K 段和输出平面，partial[part,m,n] 只由该段的 CTA 写入。
    主算子和 reduce_partitions 在同一 stream 上依次启动，归约看到所有段写完的结果。
    """
    launch_partitions(a, b, partial, shape, tile, warps, stages, split, stream, cache_all, raster_group, schedule, raster_m)
    elements = shape[0] * shape[1]
    partitions = cute.make_tensor(partial, cute.make_layout((elements, split), stride=(1, elements)))
    out = cute.make_tensor(c, cute.make_layout(elements))
    reduce_partitions(partitions, out, elements, split).launch(
        grid=((elements + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream
    )


@cute.kernel
def reduce_partitions(partial: cute.Tensor, out: cute.Tensor,
                      elements: cutlass.Constexpr, split: cutlass.Constexpr):
    # 每个线程归约 C 的一个元素。所有局部结果保留 FP32。
    lane, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    i = block * 256 + lane
    if i < elements:
        total = cutlass.Float32(0)
        for part in cutlass.range_constexpr(split):
            total += partial[i, part]
        out[i] = total


@dsl_user_op
def sm89_tf32_mma_bits(value, *, loc=None, ip=None):
    # SM89 HMMA ignores the low 13 bits. Keep this encoding in MMA registers.
    return llvm.inline_asm(
        cutlass.Int32.mlir_type,
        [cutlass.Float32(value).ir_value(loc=loc, ip=ip)],
        "{ .reg .f32 magnitude; .reg .pred nonfinite; "
        "abs.f32 magnitude, $1; setp.geu.f32 nonfinite, magnitude, 0f7F800000; "
        "mov.b32 $0, $1; @!nonfinite add.u32 $0, $0, 0x1000; }",
        "=r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def round_tf32(src: cute.Tensor, dst_bits: cute.Tensor, schedule: cutlass.Constexpr):
    # MMA fragment 通过 Int32 视图接收 TF32 的位编码。
    for i in cutlass.range(cute.size(src), unroll_full=True):
        if cutlass.const_expr(schedule == "copy_after_unmasked"):
            dst_bits[i] = cutlass.Int32(sm89_tf32_mma_bits(src[i]))
        else:
            dst_bits[i] = cutlass.Int32(cute.arch.cvt_f32_tf32(src[i]))


@cute.jit
def issue_next_tile(copy_A, copy_B, gA, gB, sA, sB, pred_A, pred_B,
                    index, stage, valid):
    # 即使到达 K 尾部也提交空组，维持 wait_group 的顺序。
    if valid:
        cute.copy(copy_A, gA[None, None, None, index],
                  sA[None, None, None, stage], pred=pred_A)
        cute.copy(copy_B, gB[None, None, None, index],
                  sB[None, None, None, stage], pred=pred_B)
    cute.arch.cp_async_commit_group()


@cute.kernel
def tf32_kernel(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    sA_layout: cute.Layout,
    sA_swizzle: cute.Swizzle,
    sB_layout: cute.Layout,
    sB_swizzle: cute.Swizzle,
    tiled_copy_A: cute.TiledCopy,
    tiled_copy_B: cute.TiledCopy,
    tiled_mma: cute.TiledMma,
    rasterization_factor: cutlass.Int32,
    tile: cutlass.Constexpr,
    stages: cutlass.Constexpr,
    schedule: cutlass.Constexpr,
    static_raster: cutlass.Constexpr,
    raster_m: cutlass.Constexpr,
):
    """用 FP32 输入视图 mA[M,part_k,S]、mB[N,part_k,S] 计算一个输出块。

    输出 mC[M,N,S] 在 S=1 时是 C，在 S>1 时是 partial。以 32×1024×5120
    的 hot、2 MiB 配置为例，tile=(32,64,16)，单 CTA 的两个 warp 沿 N 分工。
    整个输出有 16 个块，16 个 K 段把它扩成 256 个计算 CTA。z 对应的 CTA
    只遍历 K[z*320:(z+1)*320]，共 20 个 BK 块，并写出该段的 32×64。
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()
    if cutlass.const_expr(raster_m or schedule == "static_raster"):
        rasterization_factor = static_raster
    grid_dim = cute.ceil_div(mC.shape, (tile[0], tile[1], 1))
    offset_tile_x, offset_tile_y = (
        bidx // rasterization_factor,
        (bidx % rasterization_factor) + bidy * rasterization_factor,
    )
    if cutlass.const_expr(raster_m):
        offset_tile_x, offset_tile_y = offset_tile_y, offset_tile_x
    # raster 分组可能补出多余的 block；组内编号换回 (M tile, N tile) 后去掉它们。
    if grid_dim[0] <= offset_tile_x or grid_dim[1] <= offset_tile_y:
        pass
    else:
        tiler_coord = (offset_tile_x, offset_tile_y, None)

        # (m_tile,n_tile,None) 保留整条 K 循环。A 只取 m_tile，B 只取 n_tile，
        # C 用这两个坐标定位。例如 N tile 3 的输出列是 [192,256)。
        gA = cute.local_tile(
            mA[None, None, bidz],
            tiler=tile,
            coord=tiler_coord,
            proj=(1, None, 1),
        )
        gB = cute.local_tile(
            mB[None, None, bidz],
            tiler=tile,
            coord=tiler_coord,
            proj=(None, 1, 1),
        )
        gC = cute.local_tile(
            mC[None, None, bidz],
            tiler=tile,
            coord=tiler_coord,
            proj=(1, 1, None),
        )

        # 公共分派要求每个 K 段整除 BK，此时 residual_k 为 0。这里保留通用的
        # 首块对齐逻辑，若有 K 余数，数据视图向前移，首批拷贝屏蔽负 K 坐标。
        residual_k = cute.size(mA, mode=[1]) - cutlass.Int32(tile[2]) * cute.size(
            gA, mode=[2]
        )

        gA = cute.domain_offset((0, residual_k, 0), gA)
        gB = cute.domain_offset((0, residual_k, 0), gB)
        gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
        gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

        # 坐标张量只参与边界判断，和实际数据使用相同的 tile 分法。
        mcA = cute.make_identity_tensor(mA.layout.shape)
        mcB = cute.make_identity_tensor(mB.layout.shape)
        cA = cute.local_tile(
            mcA[None, None, bidz],
            tiler=tile,
            coord=tiler_coord,
            proj=(1, None, 1),
        )
        cB = cute.local_tile(
            mcB[None, None, bidz],
            tiler=tile,
            coord=tiler_coord,
            proj=(None, 1, 1),
        )

        cA = cute.domain_offset((0, residual_k, 0), cA)
        cB = cute.domain_offset((0, residual_k, 0), cB)

        @cute.struct
        class SharedStorageAB:
            a: cute.struct.Align[
                cute.struct.MemRange[mA.element_type, cute.cosize(sA_layout)],
                16,
            ]
            b: cute.struct.Align[
                cute.struct.MemRange[mB.element_type, cute.cosize(sB_layout)],
                16,
            ]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorageAB.size_in_bytes(), byte_alignment=16)
        sA = SharedStorageAB(storage).a.get_tensor(sA_layout, swizzle=sA_swizzle)
        sB = SharedStorageAB(storage).b.get_tensor(sB_layout, swizzle=sB_swizzle)

        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)
        tBgB = thr_copy_B.partition_S(gB)
        tBsB = thr_copy_B.partition_D(sB)

        tAcA = thr_copy_A.partition_S(cA)
        tBcB = thr_copy_B.partition_S(cB)

        # A 按 M 行判定，B 按连续 N 向量判定。N 为 4 的倍数，4 元素向量
        # 要么都有效，要么都在尾块之外。谓词在后续 K 块复用。
        tApA = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tAgA.shape[0][1],
                    cute.size(tAgA, mode=[1]),
                    cute.size(tAgA, mode=[2]),
                ),
                stride=(cute.size(tAgA, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        tBpB = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tBsB.shape[0][1],
                    cute.size(tBsB, mode=[1]),
                    cute.size(tBsB, mode=[2]),
                ),
                stride=(cute.size(tBsB, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        for rest_v in range(tApA.shape[0]):
            for m in range(tApA.shape[1]):
                tApA[rest_v, m, 0] = cute.elem_less(
                    tAcA[(0, rest_v), m, 0, 0][0], mA.shape[0]
                )
        for rest_v in range(tBpB.shape[0]):
            for n in range(tBpB.shape[1]):
                tBpB[rest_v, n, 0] = cute.elem_less(
                    tBcB[(0, rest_v), n, 0, 0][0], mB.shape[0]
                )

        # 完整形状的预取槽会被真实数据覆盖，可略过初始化。若有 M/N 边界，
        # 先清零 shared，令被屏蔽的输入在 MMA 中贡献 0。
        full_tiles = (mA.shape[0] % tile[0] == 0 and
                      mB.shape[0] % tile[1] == 0 and
                      mA.shape[1] % tile[2] == 0 and
                      mA.shape[1] // tile[2] >= stages - 1)
        if cutlass.const_expr(not full_tiles):
            tAsA.fill(0)
            tBsB.fill(0)
            cute.arch.sync_threads()
        num_smem_stages = cute.size(tAsA, mode=[3])
        k_tile_count = cute.size(tAgA, mode=[3])
        k_tile_index = cutlass.Int32(0)

        # 预取 stages-1 个 K 块（例如三个 stage 时先填 stage 0、1），
        # 最后一个槽留给主循环发出的下一组 global → shared 拷贝。
        for k in range(tApA.shape[2]):
            if cute.elem_less(cutlass.Int32(-1), tAcA[0, 0, k, 0][1]):
                cute.copy(
                    tiled_copy_A,
                    tAgA[None, None, k, k_tile_index],
                    tAsA[None, None, k, 0],
                    pred=tApA[None, None, k],
                )
        for k in range(tBpB.shape[2]):
            if cute.elem_less(cutlass.Int32(-1), tBcB[0, 0, k, 0][1]):
                cute.copy(
                    tiled_copy_B,
                    tBgB[None, None, k, k_tile_index],
                    tBsB[None, None, k, 0],
                    pred=tBpB[None, None, k],
                )
        k_tile_index = k_tile_index + 1
        cute.arch.cp_async_commit_group()

        for k_tile in range(1, num_smem_stages - 1):
            if k_tile == k_tile_count:
                tApA.fill(0)
                tBpB.fill(0)
            cute.copy(
                tiled_copy_A,
                tAgA[None, None, None, k_tile_index],
                tAsA[None, None, None, k_tile],
                pred=tApA,
            )
            cute.copy(
                tiled_copy_B,
                tBgB[None, None, None, k_tile_index],
                tBsB[None, None, None, k_tile],
                pred=tBpB,
            )
            k_tile_index = k_tile_index + 1
            cute.arch.cp_async_commit_group()

        # tCrC 是当前 lane 所持有的 FP32 输出累加寄存器，整个 K 段复用它。
        # tCgC 是同一批元素的全局目标视图。raw_A/B 先保存 shared 读出的
        # FP32，再通过 bits_A/B 给 MMA 的操作数寄存器填入 TF32 编码。
        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrC = tiled_mma.make_fragment_C(tCgC)
        raw_A = cute.make_fragment_like(tCrA, cutlass.Float32)
        raw_B = cute.make_fragment_like(tCrB, cutlass.Float32)
        bits_A = cute.recast_tensor(tCrA, cutlass.Int32)
        bits_B = cute.recast_tensor(tCrB, cutlass.Int32)
        tCrC.fill(0.0)

        # shared → register 与 global → shared 使用不同分工。前者按 MMA lane
        # 布局取数，A 用 ldmatrix，B 用 32 位读取，落入当前 lane 的操作数片段。
        atom_copy_s2r_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), cutlass.Float32
        )
        atom_copy_s2r_B = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=32
        )

        tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma)
        tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_copy_s2r_B, tiled_mma)

        thr_copy_ldmatrix_A = tiled_copy_s2r_A.get_slice(tidx)
        thr_copy_ldmatrix_B = tiled_copy_s2r_B.get_slice(tidx)
        tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
        tCrA_copy_view = thr_copy_ldmatrix_A.retile(raw_A)
        tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
        tCrB_copy_view = thr_copy_ldmatrix_B.retile(raw_B)

        # 起步时读 stage 0，异步拷贝写最后一个 stage。当前 shared 读取视图
        # tCsA_p/tCsB_p 保持到本块末尾，即使读索引提前指向了下一个 stage。
        smem_pipe_read = 0
        smem_pipe_write = num_smem_stages - 1

        tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
        tCsB_p = tCsB_copy_view[None, None, None, smem_pipe_read]

        num_k_block = cute.size(tCrA, mode=[2])
        copy_after_mma = schedule in ("copy_after", "copy_after_unmasked")
        if num_k_block > 1:
            cute.arch.cp_async_wait_group(num_smem_stages - 2)
            cute.arch.sync_threads()
            cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, 0], tCrA_copy_view[None, None, 0])
            cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, 0], tCrB_copy_view[None, None, 0])
            round_tf32(raw_A[None, None, 0], bits_A[None, None, 0], schedule)
            round_tf32(raw_B[None, None, 0], bits_B[None, None, 0], schedule)

        # 寄存器预读与当前 MMA 交叠。每个 BK 由多个 K=8 的 MMA 子块消耗。
        # steady_copy 将循环拆为有预取和无预取两段，减少稳态判断。
        phases = 2 if schedule == "steady_copy" else 1
        for phase in cutlass.range_constexpr(phases):
            begin, end = 0, k_tile_count
            if cutlass.const_expr(schedule == "steady_copy"):
                steady_tiles = k_tile_count - num_smem_stages + 1
                if cutlass.const_expr(phase == 0):
                    end = steady_tiles
                else:
                    begin = steady_tiles
            for k_tile in range(begin, end):
                copy_valid = k_tile + num_smem_stages - 1 < k_tile_count
                if cutlass.const_expr(schedule == "steady_copy"):
                    copy_valid = phase == 0
                for k_block in cutlass.range(num_k_block, unroll_full=True):
                    if k_block == num_k_block - 1:
                        # 即将预读下一个 shared 槽的第 0 个 MMA 子块。wait_group
                        # 等所需拷贝组完成，CTA barrier 同步所有生产者和读取者。
                        # 到下一轮才能复用旧槽，届时各 warp 已完成对它的读取。
                        tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
                        tCsB_p = tCsB_copy_view[None, None, None, smem_pipe_read]
                        cute.arch.cp_async_wait_group(num_smem_stages - 2)
                        cute.arch.sync_threads()

                    k_block_next = (k_block + 1) % num_k_block
                    cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, k_block_next], tCrA_copy_view[None, None, k_block_next])
                    cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, k_block_next], tCrB_copy_view[None, None, k_block_next])

                    # 两种 schedule 只移动发出异步拷贝的时机，读写槽的推进相同。
                    if k_block == 0 and cutlass.const_expr(not copy_after_mma):
                        issue_next_tile(
                            tiled_copy_A, tiled_copy_B, tAgA, tBgB, tAsA, tBsB,
                            tApA, tBpB, k_tile_index, smem_pipe_write, copy_valid,
                        )
                        k_tile_index = k_tile_index + 1
                        # 相邻 K 块依次占用环形槽，刚读过的槽成为下一轮写槽。
                        smem_pipe_write = smem_pipe_read
                        smem_pipe_read = smem_pipe_read + 1
                        if smem_pipe_read == num_smem_stages:
                            smem_pipe_read = 0

                    if k_block != 0:
                        round_tf32(raw_A[None, None, k_block], bits_A[None, None, k_block], schedule)
                        round_tf32(raw_B[None, None, k_block], bits_B[None, None, k_block], schedule)
                    cute.gemm(tiled_mma, tCrC, tCrA[None, None, k_block],
                              tCrB[None, None, k_block], tCrC)
                    if k_block == 0 and cutlass.const_expr(copy_after_mma):
                        issue_next_tile(
                            tiled_copy_A, tiled_copy_B, tAgA, tBgB, tAsA, tBsB,
                            tApA, tBpB, k_tile_index, smem_pipe_write, copy_valid,
                        )
                        k_tile_index = k_tile_index + 1
                        smem_pipe_write = smem_pipe_read
                        smem_pipe_read = smem_pipe_read + 1
                        if smem_pipe_read == num_smem_stages:
                            smem_pipe_read = 0

                    if k_block == num_k_block - 1:
                        round_tf32(raw_A[None, None, 0], bits_A[None, None, 0], schedule)
                        round_tf32(raw_B[None, None, 0], bits_B[None, None, 0], schedule)

        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        # 完整形状走向量写回。边界用 MMA 的坐标布局逐元素判定 M/N，
        # 避免把累加寄存器里超出矩阵的元素写到下一行或越界地址。
        if cutlass.const_expr(full_tiles):
            cute.autovec_copy(tCrC, tCgC)
        else:
            coords = cute.make_identity_tensor(mC.shape)
            c_tile = cute.local_tile(coords[None, None, bidz],
                tiler=tile, coord=tiler_coord, proj=(1, 1, None))
            t_coords = thr_mma.partition_C(c_tile)
            for i in cutlass.range(cute.size(tCrC), unroll_full=True):
                xy = t_coords[i]
                if xy[0] < mC.shape[0] and xy[1] < mC.shape[1]:
                    tCgC[i] = tCrC[i]
