import math
import unittest
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import create_common_tensor

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestFFN(TestCase):
    def deq_scale_generate(self, deq_scale_shape):
        fp32_deq_scale = np.random.uniform(low=2, high=3, size=deq_scale_shape).astype(np.float32)
        uint32_deq_scale = np.frombuffer(fp32_deq_scale, np.uint32).reshape(deq_scale_shape)
        #与高19位运算，模拟硬件
        uint32_deq_scale &= 0XFFFFE000
        fp32_deq_scale = np.frombuffer(uint32_deq_scale, np.float32)
        uint64_deq_scale = np.zeros(deq_scale_shape, np.uint64)
        uint64_deq_scale |= np.uint64(uint32_deq_scale)

        return fp32_deq_scale, uint64_deq_scale

    def supported_op_exec(self, x, weight1, weight2, activation, quant=False):
        if not quant:
            mm1_res = torch.matmul(x, weight1)
            activation_res = torch.nn.functional.relu(mm1_res)
            mm2_res = torch.matmul(activation_res, weight2)
        else: # quant
            scale = self.scale
            offset = self.offset
            deq_scale1 = self.deq_scale1_fp32
            deq_scale2 = self.deq_scale2_fp32

            x = torch.from_numpy(x).to(torch.int32)
            weight1 = torch.from_numpy(weight1).to(torch.int32)
            weight2 = torch.from_numpy(weight2).to(torch.int32)

            # mm1
            mm1_res = torch.matmul(x, weight1)
            deq_scale1 = deq_scale1.reshape(1, -1)[:, :mm1_res.shape[-1]]
            deq_scale1 = torch.from_numpy(deq_scale1)
            mm1_res = (mm1_res * deq_scale1).to(torch.float16)
            # activation
            scale = torch.from_numpy(scale).to(torch.float16)
            offset = torch.from_numpy(offset).to(torch.float16)
            activation_res = torch.nn.functional.relu(mm1_res)
            activation_res = activation_res * scale + offset
            activation_res = torch.round(activation_res, 0)
            activation_res = activation_res.clamp(-128, 127).to(torch.int32)
            # mm2
            mm2_res = torch.matmul(activation_res, weight2)
            deq_scale2 = deq_scale2.reshape(1, -1)[:, :mm2_res.shape[-1]]
            deq_scale2 = torch.from_numpy(deq_scale2)    
            mm2_res = (mm2_res * deq_scale2).to(torch.float16)        
     
        return mm2_res

    def custom_op_exec(self, x, weight1, weight2, activation, quant=False):
        if not quant:
            return torch_npu.npu_ffn(x, weight1, weight2, activation, inner_precise=1)
        else:   # quant
            scale = self.scale_clone
            offset = self.offset_clone
            deq_scale1 = torch.from_numpy(self.deq_scale1_uint64.astype(np.int64))
            deq_scale2 = torch.from_numpy(self.deq_scale2_uint64.astype(np.int64))
            return torch_npu.npu_ffn(x.npu(), weight1.npu(), weight2.npu(), activation, scale=scale.npu(), offset=offset.npu(), deq_scale1=deq_scale1.npu(), deq_scale2=deq_scale2.npu(), inner_precise=1)

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B',
        "OP `FFN` is only supported on 910B, skip this ut for this device type!")
    def test_npu_ffn(self, device="npu"):
        torch.manual_seed(0)
        x = torch.randn(8192, 320, dtype=torch.float16).npu()
        weight1 = torch.randn(320, 2560, dtype=torch.float16).npu()
        weight2 = torch.randn(2560, 320, dtype=torch.float16).npu()
        x_clone = x.clone()
        weight1_clone = weight1.clone()
        weight2_clone = weight2.clone()
        activation = "relu"

        supported_output = self.supported_op_exec(x, weight1, weight2, activation)
        custom_output = self.custom_op_exec(x_clone, weight1_clone, weight2_clone, activation)
        self.assertRtolEqual(x, x_clone, 0.001)
        self.assertRtolEqual(supported_output, custom_output, 0.001)
 

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B',
        "OP `FFN` is only supported on 910B, skip this ut for this device type!")
    def test_npu_ffn_quant(self, device="npu"):
        torch.manual_seed(0)
        np.random.seed(0)
        x = np.randint(-128, 127, size=(8192, 320), dtype=np.int8)
        weight1 = np.randint(-128, 127, size=(320, 2560), dtype=np.int8)
        weight2 = np.randint(-128, 127, size=(2560, 320), dtype=np.int8)
        self.scale = np.ones(1, dtype=np.float32)
        self.offset = np.zeros(1, dtype=np.float32)
        self.deq_scale1_fp32, self.deq_scale1_uint64 = self.deq_scale_generate((1, 2560))
        self.deq_scale2_fp32, self.deq_scale2_uint64 = self.deq_scale_generate((1, 320))

        x_clone = torch.from_numpy(x)
        weight1_clone = torch.from_numpy(weight1)
        weight2_clone = torch.from_numpy(weight2)
        self.scale_clone = torch.from_numpy(self.scale)
        self.offset_clone = torch.from_numpy(self.offset)
        activation = "relu"

        supported_output = self.supported_op_exec(x, weight1, weight2, activation, quant=True)
        custom_output = self.custom_op_exec(x_clone, weight1_clone, weight2_clone, activation, quant=True)
        self.assertRtolEqual(x, x_clone, 0.001)
        self.assertRtolEqual(supported_output, custom_output, 0.001)


if __name__ == "__main__":
    run_tests()
