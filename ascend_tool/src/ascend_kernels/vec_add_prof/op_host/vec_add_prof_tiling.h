/*
 * vec_add_prof_tiling.h — shared tiling data between host and kernel
 */

#ifndef VEC_ADD_PROF_TILING_H
#define VEC_ADD_PROF_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(VecAddProfTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(VecAddProf, VecAddProfTilingData)
} // namespace optiling

#endif // VEC_ADD_PROF_TILING_H
