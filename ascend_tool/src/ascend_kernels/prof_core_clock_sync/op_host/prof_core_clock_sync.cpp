/*
 * prof_core_clock_sync.cpp — host-side tiling and operator registration
 *
 * Operator A: ProfCoreClockSync — Single-card multi-core clock synchronization
 *
 * Synchronizes all AICore clocks within a single NPU card through multi-round
 * barriers. Captures timestamps at each sync point so the host can compute
 * per-core clock offsets for aligning kernel_tool.h profiling data.
 *
 * Host offset calculation:
 *   offset_core_i = mean(ts[core_i][r] - ts[core_0][r]) for r in [0, syncRounds)
 */

#include "prof_core_clock_sync_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

constexpr uint32_t DEFAULT_SYNC_ROUNDS = 16; // default sync rounds for averaging

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    ProfCoreClockSyncTilingData tiling;

    // Get sync_rounds from output shape dimension (shape[1] encodes rounds)
    // or use default. For simplicity, use the output tensor shape to infer.
    const gert::StorageShape *outShape = context->GetOutputShape(0);
    uint32_t totalElements = 1;
    for (int i = 0; i < outShape->GetStorageShape().GetDimNum(); i++) {
        totalElements *= static_cast<uint32_t>(outShape->GetStorageShape().GetDim(i));
    }

    // Query the platform for the actual number of AI cores
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t blockDim = static_cast<uint32_t>(ascendcPlatform.GetCoreNumAiv());
    uint32_t syncRounds = DEFAULT_SYNC_ROUNDS;

    tiling.set_syncRounds(syncRounds);
    tiling.set_blockDim(blockDim);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    context->SetBlockDim(blockDim);

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x1_shape = context->GetInputShape(0);
    gert::Shape *y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    return ge::GRAPH_SUCCESS;
}
} // namespace ge

namespace ops {

class ProfCoreClockSync : public OpDef {
public:
    explicit ProfCoreClockSync(const char *name) : OpDef(name)
    {
        this->Input("sync_buf")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND});

        this->Output("sync_timestamps")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910_93");
    }
};

OP_ADD(ProfCoreClockSync);

} // namespace ops
