import math
import unittest
import numpy as np
import torch

import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

torch.npu.set_compile_mode(jit_compile=False)
DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


class TestReplicationPad3dBackward(TestCase):

    def replication_pad3d_backward(self, grad_out, self_tensor, padding):
        padding_layer = torch.nn.ReplicationPad3d(padding)

        self_tensor.requires_grad = True
        output = padding_layer(self_tensor)
        output.backward(grad_out)

        grad_result = self_tensor.grad
        return grad_result

    def get_input_data(self, device):
        input_shape = [1, 1, 4, 4, 4]
        output_shape = [1, 1, 8, 8, 8]
        padding = [2, 2, 2, 2, 2, 2]

        self_tensor = torch.ones(input_shape, device=device, dtype=torch.float32)
        grad_out_tensor = torch.ones(output_shape, device=device, dtype=torch.float32)

        return grad_out_tensor, self_tensor, padding

    def test_replication_pad3d_backward(self):
        grad_out_tensor, self_tensor, padding = self.get_input_data('cpu')
        grad_out_tensor_npu, self_tensor_npu, padding_npu = self.get_input_data('npu')

        golden = self.replication_pad3d_backward(grad_out_tensor, self_tensor, padding)
        output = self.replication_pad3d_backward(grad_out_tensor_npu, self_tensor_npu, padding)

        self.assertRtolEqual(golden, output)


if __name__ == "__main__":
    run_tests()
