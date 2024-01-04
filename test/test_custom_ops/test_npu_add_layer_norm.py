import unittest
from collections import namedtuple
import numpy as np
import torch_npu
import torch
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import get_npu_device
torch.npu.set_compile_mode(jit_compile=False)

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]
Result = namedtuple('Result', ['y', 'mean', 'rstd', 'x'])


class TestNPUAddLayerNorm(TestCase):
    def supported_op_exec(self, x1, x2, gamma, beta, bias=None):
        epsilon = 1e-5
        x = x1 + x2
        x = x + bias
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        rstd = np.divide(1, np.sqrt(variance + epsilon))
        y = ((x - mean) * rstd) * gamma + beta
        return Result(y, mean, rstd, x)

    def custom_op_exec_default(self, x1, x2, gamma, beta):
        y, mean, rstd, x = torch_npu.npu_add_layer_norm(x1, x2, gamma, beta)
        return Result(y.cpu().numpy(), mean.cpu().numpy(), rstd.cpu().numpy(), x.cpu().numpy())

    def custom_op_exec_additional_output(self, x1, x2, gamma, beta):
        y, mean, rstd, x = torch_npu.npu_add_layer_norm(x1, x2, gamma, beta, None, 1e-5, True)
        return Result(y.cpu().numpy(), mean.cpu().numpy(), rstd.cpu().numpy(), x.cpu().numpy())

    def custom_op_exec_bias(self, x1, x2, gamma, beta, bias):
        y, mean, rstd, x = torch_npu.npu_add_layer_norm(x1, x2, gamma, beta, bias)
        return Result(y.cpu().numpy(), mean.cpu().numpy(), rstd.cpu().numpy(), x.cpu().numpy())

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B',
        "OP `AddLayerNorm` is only supported on 910B, skip this ut for this device type!")
    def test_add_layer_norm_default(self, device="npu"):
        if device is None:
            device = get_npu_device()
        cpu_input_x1 = np.random.uniform(0, 100, [1, 1024, 8192]).astype(np.float32)
        cpu_input_x2 = np.random.uniform(0, 100, [1, 1024, 8192]).astype(np.float32)
        cpu_input_gamma = np.random.uniform(0, 1, [8192]).astype(np.float32)
        cpu_input_beta = np.random.uniform(0, 1, [8192]).astype(np.float32)
        
        npu_input_x1 = torch.from_numpy(cpu_input_x1).to(device)
        npu_input_x2 = torch.from_numpy(cpu_input_x2).to(device)
        npu_input_gamma = torch.from_numpy(cpu_input_gamma).to(device)
        npu_input_beta = torch.from_numpy(cpu_input_beta).to(device)

        supported_result = self.supported_op_exec(cpu_input_x1, cpu_input_x2, cpu_input_gamma, cpu_input_beta)
        custom_result = self.custom_op_exec_default(npu_input_x1, npu_input_x2, npu_input_gamma, npu_input_beta)
        
        self.assertRtolEqual(supported_result.y, custom_result.y)
        self.assertRtolEqual(supported_result.mean, custom_result.mean)
        self.assertRtolEqual(supported_result.rstd, custom_result.rstd)

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B',
        "OP `AddLayerNorm` is only supported on 910B, skip this ut for this device type!")
    def test_add_layer_norm_additional_output(self, device="npu"):
        if device is None:
            device = get_npu_device()
        cpu_input_x1 = np.random.uniform(0, 100, [1, 1024, 8192]).astype(np.float32)
        cpu_input_x2 = np.random.uniform(0, 100, [1, 1024, 8192]).astype(np.float32)
        cpu_input_gamma = np.random.uniform(0, 1, [8192]).astype(np.float32)
        cpu_input_beta = np.random.uniform(0, 1, [8192]).astype(np.float32)
        
        npu_input_x1 = torch.from_numpy(cpu_input_x1).to(device)
        npu_input_x2 = torch.from_numpy(cpu_input_x2).to(device)
        npu_input_gamma = torch.from_numpy(cpu_input_gamma).to(device)
        npu_input_beta = torch.from_numpy(cpu_input_beta).to(device)

        supported_result = self.supported_op_exec(cpu_input_x1, cpu_input_x2, cpu_input_gamma, cpu_input_beta)
        custom_result = self.custom_op_exec_additional_output(npu_input_x1, npu_input_x2, npu_input_gamma, npu_input_beta)
        
        self.assertRtolEqual(supported_result.y, custom_result.y)
        self.assertRtolEqual(supported_result.mean, custom_result.mean)
        self.assertRtolEqual(supported_result.rstd, custom_result.rstd)
        self.assertRtolEqual(supported_result.x, custom_result.x)

    @unittest.skipIf(DEVICE_NAME != 'Ascend910B',
        "OP `AddLayerNorm` is only supported on 910B, skip this ut for this device type!")
    def test_add_layer_norm_bias(self, device="npu"):
        if device is None:
            device = get_npu_device()
        cpu_input_x1 = np.random.uniform(0, 100, [1, 1024, 8192]).astype(np.float32)
        cpu_input_x2 = np.random.uniform(0, 100, [1, 1024, 8192]).astype(np.float32)
        cpu_input_gamma = np.random.uniform(0, 1, [8192]).astype(np.float32)
        cpu_input_beta = np.random.uniform(0, 1, [8192]).astype(np.float32)
        cpu_input_bias = np.random.uniform(0, 1, [8192]).astype(np.float32)
        
        npu_input_x1 = torch.from_numpy(cpu_input_x1).to(device)
        npu_input_x2 = torch.from_numpy(cpu_input_x2).to(device)
        npu_input_gamma = torch.from_numpy(cpu_input_gamma).to(device)
        npu_input_beta = torch.from_numpy(cpu_input_beta).to(device)

        supported_result = self.supported_op_exec(cpu_input_x1, cpu_input_x2, cpu_input_gamma, cpu_input_beta, cpu_input_bias)
        custom_result = self.custom_op_exec_bias(npu_input_x1, npu_input_x2, npu_input_gamma, npu_input_beta, cpu_input_bias)
        
        self.assertRtolEqual(supported_result.y, custom_result.y)
        self.assertRtolEqual(supported_result.mean, custom_result.mean)
        self.assertRtolEqual(supported_result.rstd, custom_result.rstd)

if __name__ == "__main__":
    run_tests()