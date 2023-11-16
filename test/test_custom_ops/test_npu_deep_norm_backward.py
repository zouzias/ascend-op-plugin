import numpy as np
import torch_npu
import torch
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import get_npu_device
torch.npu.set_compile_mode(jit_compile=False)


class TestNPUDeepNormBackward(TestCase):
    def supported_op_exec(self, dy, x, gx, gamma, mean, rstd):
        alpha = 0.3
        epsilon = 1e-6

        dx = x
        dgx = gx
        dbeta = gamma
        dgamma = gamma

        return dx, dgx, dbeta, dgamma

    def custom_op_exec(self, dy, x, gx, gamma, mean, rstd):
        dx, dgx, dbeta, dgamma = torch_npu.npu_deep_norm_backward(dy, x, gx, gamma, mean, rstd, float(0.3))
        return dx.cpu().numpy(), dgx.cpu().numpy(), dbeta.cpu().numpy(), dgamma.cpu().numpy()

    def test_deep_norm_backward(self, device="npu"):
        if device is None:
            device = get_npu_device()

        cpu_input_dy = np.random.uniform(0, 100, [1024, 12288]).astype(np.float32)
        cpu_input_x = np.random.uniform(0, 100, [1024, 12288]).astype(np.float32)
        cpu_input_gx = np.random.uniform(0, 100, [1024, 12288]).astype(np.float32)
        cpu_input_gamma = np.random.uniform(0, 100, [12288]).astype(np.float32)
        cpu_input_mean = np.random.uniform(0, 100, [1024, 1]).astype(np.float32)
        cpu_input_rstd = np.random.uniform(0, 100, [1024, 1]).astype(np.float32)

        npu_input_dy = torch.from_numpy(cpu_input_dy).to(device)
        npu_input_x = torch.from_numpy(cpu_input_x).to(device)
        npu_input_gx = torch.from_numpy(cpu_input_gx).to(device)
        npu_input_gamma = torch.from_numpy(cpu_input_gamma).to(device)
        npu_input_mean = torch.from_numpy(cpu_input_mean).to(device)
        npu_input_rstd = torch.from_numpy(cpu_input_rstd).to(device)

        supported_dx, supported_dgx, supported_dbeta, supported_dgamma = self.supported_op_exec(cpu_input_dy,
                                                                                                cpu_input_x, 
                                                                                                cpu_input_gx, 
                                                                                                cpu_input_gamma,
                                                                                                cpu_input_mean, 
                                                                                                cpu_input_rstd)
        custom_dx, custom_dgx, custom_dbeta, custom_dgamma = self.custom_op_exec(cpu_input_dy,
                                                                                 cpu_input_x, 
                                                                                 cpu_input_gx, 
                                                                                 cpu_input_gamma,
                                                                                 cpu_input_mean, 
                                                                                 cpu_input_rstd)
        
        self.assertRtolEqual(supported_dx, custom_dx)
        self.assertRtolEqual(supported_dgx, custom_dgx)
        self.assertRtolEqual(supported_dbeta, custom_dbeta)
        self.assertRtolEqual(supported_dgamma, custom_dgamma)

if __name__ == "__main__":
    run_tests()
