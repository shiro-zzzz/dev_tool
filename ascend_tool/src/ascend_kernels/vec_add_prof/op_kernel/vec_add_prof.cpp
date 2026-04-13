/*
 * vec_add_prof.cpp — Ascend C kernel: vector add with profiling
 *
 * Demonstrates kernel_tool.h profiling instrumentation.
 * Profiling tags:
 *   0         : init timestamp (auto from PROF_INIT)
 *   100+tile  : CopyIn  complete for tile
 *   200+tile  : Compute complete for tile
 *   300+tile  : CopyOut complete for tile
 *   999       : all tiles finished
 */

#include "kernel_operator.h"
// Enable profiling — comment out to disable (zero overhead)
#define ASCEND_PROFILE_ENABLE
#include "kernel_tool.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // double buffer

class KernelVecAddProf {
public:
    __aicore__ inline KernelVecAddProf() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR profBuf, GM_ADDR z,
                                uint32_t totalLength, uint32_t tileNum)
    {
        uint32_t blockNum = GetBlockNum();
        uint32_t blockIdx = GetBlockIdx();

        this->blockLength = totalLength / blockNum;
        this->tileNum     = tileNum;
        this->tileLength  = this->blockLength / tileNum;

        uint32_t offset = blockIdx * this->blockLength;
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x) + offset, this->blockLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(y) + offset, this->blockLength);
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(z) + offset, this->blockLength);
        profGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(profBuf));

        pipe.InitBuffer(inQueueX,  BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(inQueueY,  BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        PROF_INIT(profGm);

        int32_t loops = static_cast<int32_t>(this->tileNum);
        for (int32_t i = 0; i < loops; i++) {
            // SyncAll before each tile — creates visible alignment points
            PROF_SYNC_ALL();
            PROF_RECORD_TIME(10 + i);   // tag 10+tile: sync point (对齐标记)
            PROF_SLEEP_US(10);          // 10 μs gap for visual separation
            PROF_RECORD_TIME(20 + i);   // tag 20+tile: after sleep (间隔结束)
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }

        PROF_SYNC_ALL();       // final alignment point
        PROF_RECORD_TIME(998); // final sync point
        PROF_RECORD_TIME(999); // all tiles done
        PROF_TO_GM(profGm);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
        PROF_SLEEP_US(5);  // 5 μs gap after CopyIn
        PROF_RECORD_TIME(100 + progress);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        Add(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
        PROF_SLEEP_US(5);  // 5 μs gap after Compute
        PROF_RECORD_TIME(200 + progress);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
        PROF_SLEEP_US(5);  // 5 μs gap after CopyOut
        PROF_RECORD_TIME(300 + progress);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM>  inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<half>    xGm, yGm, zGm;
    GlobalTensor<int64_t> profGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void vec_add_prof(
    GM_ADDR x, GM_ADDR y, GM_ADDR prof_buf, GM_ADDR z,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelVecAddProf op;
    op.Init(x, y, prof_buf, z, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}
