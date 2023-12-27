import unittest
import numpy as np

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestTransQuantParam(TestCase):

    def supported_op_exec(self, scale):
        scale_shape = scale.shape()
        uint32_scale = np.frombuffer(scale, np.uint32).reshape(scale_shape)
        # 与高19位运算，模拟硬件
        uint32_scale &= 0XFFFFE000
        # output dtype: fp16
        uint64_deq_scale = np.zeros(scale_shape, np.uint64)
        uint64_deq_scale |= np.uint64(uint32_scale)
        return uint64_deq_scale

    def custom_op_exec(self, scale):
        return torch_npu.npu_trans_quant_param(scale).cpu().numpy()

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B',
        "OP `TransQuantParam` is tested on 910B(support 910B/910C), skip this ut for other device type!")
    def test_npu_transquantparam(self, device="npu"):
        cpu_input_scale = np.random.randn(0, 10, [8192]).astype(np.float32)
        npu_input_scale = torch.from_numpy(cpu_input_scale).to(device)

        supported_output = self.supported_op_exec(cpu_input_scale)
        custom_output = self.custom_op_exec(npu_input_scale)
        self.assertRtolEqual(cpu_input_scale, npu_input_scale, 0.001)
        self.assertRtolEqual(supported_output, custom_output, 0.001)


if __name__ == "__main__":
    run_tests()
