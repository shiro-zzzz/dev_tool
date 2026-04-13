/*
 * vec_add_prof.cpp — host-side tiling and operator registration for VecAddProf
 */

#include "vec_add_prof_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {

constexpr uint32_t BLOCK_DIM    = 8;   // number of AI Cores
constexpr uint32_t TILE_LENGTH  = 256; // elements per tile (must be ≥16 for 32B alignment with half)

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    VecAddProfTilingData tiling;

    // Compute total number of elements from input[0] shape
    const gert::StorageShape *shape = context->GetInputShape(0);
    uint32_t totalLength = 1;
    for (int i = 0; i < shape->GetStorageShape().GetDimNum(); i++) {
        totalLength *= static_cast<uint32_t>(shape->GetStorageShape().GetDim(i));
    }

    // Calculate tile count per core
    uint32_t blockLength = totalLength / BLOCK_DIM;
    uint32_t tileNum = blockLength / TILE_LENGTH;
    if (tileNum == 0) {
        tileNum = 1;
    }

    tiling.set_totalLength(totalLength);
    tiling.set_tileNum(tileNum);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    context->SetBlockDim(BLOCK_DIM);

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

class VecAddProf : public OpDef {
public:
    explicit VecAddProf(const char *name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        this->Input("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        this->Input("prof_buf")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND});

        this->Output("z")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape)
            .SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910_93");
    }
};

OP_ADD(VecAddProf);

} // namespace ops
