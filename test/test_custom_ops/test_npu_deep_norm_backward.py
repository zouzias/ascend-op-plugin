import numpy as np
import torch_npu
import torch
from torch_npu.testing.testcase import TestCase, run_tests
from torch_npu.testing.common_utils import get_npu_device
torch.npu.set_compile_mode(jit_compile=False)


class DeepNormGradInputParams:
    def __init__(self, dy, x, gx, gamma, mean, rstd):
        self.dy = dy
        self.x = x
        self.gx = gx
        self.gamma = gamma
        self.mean = mean
        self.rstd = rstd


class DeepNormGradOutputParams:
    def __init__(self, dx, dgx, dbeta, dgamma):
        self.dx = dx
        self.dgx = dgx
        self.dbeta = dbeta
        self.dgamma = dgamma


class TestNPUDeepNormBackward(TestCase):
    def supported_op_exec(self, deepnormgrad_input: DeepNormGradInputParams):
        dy = deepnormgrad_input.dy
        x = deepnormgrad_input.x
        gx = deepnormgrad_input.gx
        gamma = deepnormgrad_input.gamma
        mean = deepnormgrad_input.mean
        rstd = deepnormgrad_input.rstd
        alpha = 0.3

        x_sum = alpha * x + gx
        reduce_axis = (2)
        value_D = x_sum.shape[-1]

        pd_xl = dy * gamma
        x2_tensor = x_sum - mean

        pd_var_first_part = (-0.5) * pd_xl * x2_tensor * np.power(rstd, 3)
        pd_var = np.sum(pd_var_first_part, reduce_axis, keepdims=True)

        pd_mean = np.sum((-1.0) * pd_xl * rstd, reduce_axis, keepdims=True)

        pd_x_first_part = pd_xl * rstd
        try:
            pd_x_second_part = pd_var * (2.0 / value_D) * x2_tensor
            pd_x_thrid_part = pd_mean * (1.0 / value_D)
        except ZeroDivisionError as err:
            raise err
        pd_gx = pd_x_first_part + pd_x_second_part + pd_x_thrid_part

        pd_x = alpha * pd_gx

        pd_gamma = np.sum(dy * x2_tensor * rstd, axis=0, keepdims=True)
        pd_beta = np.sum(dy, axis=0, keepdims=True)

        return DeepNormGradOutputParams(pd_x, pd_gx, pd_beta, pd_gamma)


    def custom_op_exec(self, deepnormgrad_input: DeepNormGradInputParams, device="npu"):
        dy = deepnormgrad_input.dy
        x = deepnormgrad_input.x
        gx = deepnormgrad_input.gx
        gamma = deepnormgrad_input.gamma
        mean = deepnormgrad_input.mean
        rstd = deepnormgrad_input.rstd

        npu_input_dy = torch.from_numpy(dy).to(device)
        npu_input_x = torch.from_numpy(x).to(device)
        npu_input_gx = torch.from_numpy(gx).to(device)
        npu_input_gamma = torch.from_numpy(gamma).to(device)
        npu_input_mean = torch.from_numpy(mean).to(device)
        npu_input_rstd = torch.from_numpy(rstd).to(device)

        dx, dgx, dbeta, dgamma = torch_npu.npu_deep_norm_backward(npu_input_dy,
                                                                  npu_input_x,
                                                                  npu_input_gx,
                                                                  npu_input_gamma,
                                                                  npu_input_mean,
                                                                  npu_input_rstd,
                                                                  float(0.3))
        return DeepNormGradOutputParams(dx.cpu().numpy(), dgx.cpu().numpy(),
                                        dbeta.cpu().numpy(), dgamma.cpu().numpy())

    def test_deep_norm_backward(self, device="npu"):
        if device is None:
            device = get_npu_device()

        cpu_input_dy = np.random.uniform(0, 100, [1024, 12288]).astype(np.float32)
        cpu_input_x = np.random.uniform(0, 100, [1024, 12288]).astype(np.float32)
        cpu_input_gx = np.random.uniform(0, 100, [1024, 12288]).astype(np.float32)
        cpu_input_gamma = np.random.uniform(0, 100, [12288]).astype(np.float32)
        cpu_input_mean = np.random.uniform(0, 100, [1024, 1]).astype(np.float32)
        cpu_input_rstd = np.random.uniform(0, 100, [1024, 1]).astype(np.float32)

        deepnormgrad_input = DeepNormGradInputParams(cpu_input_dy, cpu_input_x, cpu_input_gx,
                                                     cpu_input_gamma, cpu_input_mean, cpu_input_rstd)

        supported_output = self.supported_op_exec(deepnormgrad_input)
        custom_output = self.custom_op_exec(deepnormgrad_input)

        self.assertRtolEqual(supported_output.dx, custom_output.dx)
        self.assertRtolEqual(supported_output.dgx, custom_output.dgx)
        self.assertRtolEqual(supported_output.dbeta, custom_output.dbeta)
        self.assertRtolEqual(supported_output.dgamma, custom_output.dgamma)

if __name__ == "__main__":
    run_tests()

