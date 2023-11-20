/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file matrix_calculation_ops.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_PROTO_INC_MATRIX_CALCULATION_OPS_H_
#define OPS_BUILT_IN_OP_PROTO_INC_MATRIX_CALCULATION_OPS_H_

#include "graph/operator_reg.h"

namespace ge {
/**
* @brief scatterList op \n
* @par Inputs:
* Four inputs, including:
* @li var: A TensorList. Must be one of the following types: float16, bf16, float32, int8, 
    int16, int32, int64, uint8, uint16, uint32, uint64.
* @li indices: A Tensor. Must be one of the following types: int32, int64.
* @li updates: A Tensor. Must be one of the following types: float16, bf16, float32, int8, 
    int16, int32, int64, uint8, uint16, uint32, uint64.
* @li mask: A Optional Tensor. Must be one of the following types: DT_UINT8.

* @par Attributes:
* @li reduce: A attribute, the type is string, default value is "update".
* @li axis: A attribute, int32，default value is -2\n

* @par Outputs:
* one outputs, including:
* @li var: A Tensor. Must be one of the following types: float16, bf16, float32, int8, 
    int16, int32, int64, uint8, uint16, uint32, uint64. \n

* @par Restrictions:
* Warning: THIS FUNCTION IS EXPERIMENTAL. Please do not use. \n
*/
REG_OP(ScatterList)
    .DYNAMIC_INPUT(var, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                                    DT_UINT16, DT_UINT32, DT_UINT64}))
    .INPUT(indices, TensorType::IndexNumberType())
    .INPUT(updates, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                                DT_UINT16, DT_UINT32, DT_UINT64}))
    .OPTIONAL_INPUT(mask, TensorType({DT_UINT8}))
    .DYNAMIC_OUTPUT(var, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT, DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                                     DT_UINT16, DT_UINT32, DT_UINT64}))
    .ATTR(reduce, String, "update")
    .ATTR(axis, Int, -2)
    .OP_END_FACTORY_REG(ScatterList)

#endif  // OPS_BUILT_IN_OP_PROTO_INC_MATRIX_CALCULATION_OPS_H_
