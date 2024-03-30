import math
import unittest
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import SupportedDevices


class TestLinspace(TestCase):
    def supported_op_exec(self, start, end):
        return torch.linspace(start, end, 5, device="cpu")

    def custom_op_exec(self, start, end):
        start = start.npu()
        end = end.npu()
        return torch.linspace(start, end, 5, device="npu")

    @SupportedDevices(['Ascend910B'])
    def test_npu_linspace(self, device="npu"):
        start = torch.randint(10, (1, ))
        end = torch.randint(11, 100, (1, ))

        supported_output = self.supported_op_exec(start, end)
        custom_output = self.custom_op_exec(start, end)
        custom_output = custom_output.cpu()
        self.assertRtolEqual(supported_output, custom_output)


if __name__ == "__main__":
    run_tests()
