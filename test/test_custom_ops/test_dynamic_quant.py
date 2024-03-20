import math
import unittest
import copy
import struct
from struct import pack, unpack
import numpy as np
import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestDynamicQuant(TestCase):
    def supported_op_exec(self, input):
        inputFp32 = input.float()
        inputAbs = inputFp32.abs()
        inputMax = inputAbs.max(dim=-1, keepdim=True)
        scaleNpu = inputMax / 127

        inputScaled = inputFp32 / scaleNpu
        outputNpu = inputScaled.round()

        return [outputNpu, scaleNpu]

    def custom_op_exec(self, input):
        return torch_npu.npu_dynamic_quant(input)

    def generate_input(self, input_shape, dtype="float16"):
        input = np.random.random(input_shape)
        inputNpu = None
        if dtype == "float16":
            input = torch.from_numpy(input).to(torch.float16).npu()
        else:
            input = torch.from_numpy(input).to(torch.bfloat16).npu()
        return [inputNpu]

    @SupportedDevices(['Ascend910B'])
    @SupportedDevices(['Ascend310P'])
    def test_npu_dynamic_quant(self, device="npu"):
        input = self.generate_input([4, 2048, 1024])

        supported_output = self.supported_op_exec(input.clone())
        custom_output = self.custom_op_exec(input.clone())
        self.assertTensorsSlowEqual(supported_output[0], custom_output[0], 1)
        self.assertRtolEqual(supported_output[1], custom_output[1], 0.0001)

if __name__ == "__main__":
    run_tests()